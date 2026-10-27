# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Cascading SPICE refresh on child datasets.

When columns are removed from a parent dataset and the parent is refreshed, every child
dataset still holds SPICE data shaped to the parent's old column set. The next scheduled
child refresh — or a child transform that references a now-absent parent column — can then
fail or show errors. ``DatasetModifier.trigger_cascading_refresh`` refreshes the whole
discovered tree, parents before children, right after the parent's own ingestion.

Two properties matter and pull in opposite directions:

- It must refresh *every* refreshable descendant, in parent-before-child order, so each
  level re-resolves against the level above it.
- It must *never* raise. The parent change is already committed by the time this runs, so
  a child that cannot be refreshed (an uploaded-file source) or that fails transiently
  (throttling, access denied) is reported, not allowed to turn a successful optimisation
  into a failed one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import create_autospec

from lambda_source.modifier import DatasetModifier
from lambda_source.qs_client import (
    AccessDeniedError,
    QSClient,
    QSClientError,
    ThrottlingError,
)


def _child(dataset_id, depth=1, name=""):
    """A stand-in for a discovered ChildDataset — only the fields the cascade reads."""
    return SimpleNamespace(dataset_id=dataset_id, depth=depth, name=name or dataset_id)


def _modifier():
    """A DatasetModifier whose QSClient records create_ingestion calls.

    autospec rather than a bare MagicMock so a wrong call signature fails here, which is
    the project convention (and enforced by the AST guard over the test suite).
    """
    client = create_autospec(QSClient, instance=True)
    client.create_ingestion.return_value = {"Status": 201}
    return DatasetModifier(client), client


# No real waiting in tests. delay_seconds=0 is passed everywhere; a separate test asserts
# the default is a positive pause so the pacing is not accidentally removed.


class TestEveryRefreshableChildIsRefreshed:
    def test_one_ingestion_per_child(self):
        modifier, client = _modifier()
        children = [_child("c1"), _child("c2"), _child("c3")]

        results = modifier.trigger_cascading_refresh(children, delay_seconds=0)

        assert client.create_ingestion.call_count == 3
        assert all(r.refresh_started for r in results)
        assert {r.dataset_id for r in results} == {"c1", "c2", "c3"}

    def test_each_ingestion_id_carries_the_cascade_prefix(self):
        """A distinct prefix so a cascaded refresh is attributable to this tool and
        distinguishable from the parent's own `optimiser-` ingestion and a rollback's
        `optimiser-rollback-` in the QuickSight ingestion history."""
        modifier, client = _modifier()

        modifier.trigger_cascading_refresh([_child("c1")], delay_seconds=0)

        _, kwargs = client.create_ingestion.call_args
        # create_ingestion(dataset_id, ingestion_id) — positional in the real signature
        args = client.create_ingestion.call_args.args
        ingestion_id = args[1] if len(args) > 1 else kwargs.get("ingestion_id")
        assert ingestion_id.startswith("optimiser-cascade-")

    def test_ingestion_ids_are_unique_per_child(self):
        modifier, client = _modifier()
        children = [_child("c1"), _child("c2"), _child("c3")]

        modifier.trigger_cascading_refresh(children, delay_seconds=0)

        ids = [call.args[1] for call in client.create_ingestion.call_args_list]
        assert len(ids) == len(set(ids)), "ingestion ids collided across children"


class TestParentBeforeChildOrdering:
    def test_children_refresh_in_ascending_depth(self):
        """Topological order: a grandchild must refresh after its parent-level child, so
        it reads data already updated one level up. ``depth`` is the only ordering signal
        discovery exposes, and level order satisfies the parent-before-child guarantee."""
        modifier, client = _modifier()
        # Deliberately out of order on input.
        children = [
            _child("grandchild", depth=2),
            _child("direct", depth=1),
            _child("great-grand", depth=3),
        ]

        results = modifier.trigger_cascading_refresh(children, delay_seconds=0)

        refreshed_order = [call.args[0] for call in client.create_ingestion.call_args_list]
        assert refreshed_order == ["direct", "grandchild", "great-grand"]
        # The returned results preserve the order they were refreshed in.
        assert [r.dataset_id for r in results] == ["direct", "grandchild", "great-grand"]


class TestAChildRefreshNeverRaises:
    def test_unrefreshable_child_is_reported_not_raised(self):
        """A file-upload child cannot be refreshed. That is a fact about the child, not a
        failure — mark it unsupported and carry on."""
        modifier, client = _modifier()
        client.create_ingestion.side_effect = QSClientError(
            "Unhandled error calling create_ingestion: [InvalidParameterValueException] "
            "DataSet is not refreshable because of the source type",
            error_code="InvalidParameterValueException",
            api_name="create_ingestion",
        )

        results = modifier.trigger_cascading_refresh([_child("c1")], delay_seconds=0)

        assert len(results) == 1
        assert results[0].refresh_unsupported is True
        assert results[0].refresh_started is False
        assert results[0].error is None, "an unsupported source is not an error"

    def test_a_throttled_child_is_recorded_as_a_failure(self):
        modifier, client = _modifier()
        client.create_ingestion.side_effect = ThrottlingError(
            "Throttling retries exhausted",
            error_code="ThrottlingException",
            api_name="create_ingestion",
        )

        results = modifier.trigger_cascading_refresh([_child("c1")], delay_seconds=0)

        assert results[0].refresh_started is False
        assert results[0].refresh_unsupported is False
        assert "ThrottlingError" in results[0].error

    def test_one_failing_child_does_not_stop_the_rest(self):
        """The whole point of not raising: a failure on one descendant must not abandon
        the descendants after it. All three are attempted; the middle one fails."""
        modifier, client = _modifier()

        def side_effect(dataset_id, ingestion_id):
            if dataset_id == "c2":
                raise AccessDeniedError(
                    "Access denied calling create_ingestion",
                    error_code="AccessDeniedException",
                    api_name="create_ingestion",
                )
            return {"Status": 201}

        client.create_ingestion.side_effect = side_effect
        children = [_child("c1", 1), _child("c2", 2), _child("c3", 3)]

        results = modifier.trigger_cascading_refresh(children, delay_seconds=0)

        assert client.create_ingestion.call_count == 3, "every child was attempted"
        by_id = {r.dataset_id: r for r in results}
        assert by_id["c1"].refresh_started is True
        assert by_id["c2"].error is not None and by_id["c2"].refresh_started is False
        assert by_id["c3"].refresh_started is True, "the child after the failure still ran"


class TestEmptyAndEdgeCases:
    def test_no_children_returns_empty_and_calls_nothing(self):
        modifier, client = _modifier()

        results = modifier.trigger_cascading_refresh([], delay_seconds=0)

        assert results == []
        client.create_ingestion.assert_not_called()

    def test_a_child_without_an_id_is_skipped(self):
        modifier, client = _modifier()
        children = [_child("c1"), SimpleNamespace(dataset_id="", depth=1, name="")]

        results = modifier.trigger_cascading_refresh(children, delay_seconds=0)

        assert client.create_ingestion.call_count == 1
        assert [r.dataset_id for r in results] == ["c1"]


class TestPacingIsNotSilentlyRemoved:
    def test_the_default_delay_is_a_positive_pause(self):
        """CreateIngestion is rate-limited and the cascade fires one per descendant with
        nothing between them. The default pacing guards a wide tree from burning through
        the per-call throttle retries on its first few children."""
        assert DatasetModifier.CASCADE_REFRESH_DELAY_SECONDS > 0
