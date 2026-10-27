# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""An apply must not raise an alert because a dataset has no refresh to start.

``execute_rollback`` was fixed to treat "DataSet is not refreshable because of the source
type" as a fact about the dataset rather than a failure. The apply path was left
inconsistent: it logged a warning and called ``publish_error``, so every removal against a
file-backed dataset raised an SNS error alert about a non-event.

Harmless today only because the topic has no subscribers. The moment anyone sets
``NotificationEmail`` it becomes a recurring alert for something nobody can act on, and an
alert that cries wolf stops being read — which costs more than the noise itself, because
the next alert is a real one.

QuickSight cannot refresh a dataset built from an uploaded file: it keeps no connection or
location metadata to re-read from, so CreateIngestion is rejected outright rather than
failing transiently. Documented, not transient:
https://docs.aws.amazon.com/quicksight/latest/user/refreshing-data.html
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest

from lambda_source.modifier import DatasetModifier, ModificationError
from lambda_source.notifications import NotificationPublisher
from lambda_source.reporter import ReportGenerator
from lambda_source.snapshot import SnapshotManager

ACCOUNT = "123456789012"

UNREFRESHABLE = ModificationError(
    "CreateIngestion failed for dataset ds-1: Unhandled error calling create_ingestion: "
    "[InvalidParameterValueException] DataSet is not refreshable because of the source type",
    dataset_id="ds-1",
)


class TestRefreshUnsupportedIsNotAnAlert:
    """Exercised through the classifier the apply path now shares with rollback.

    Reaching _execute_apply directly means standing up the whole pipeline; the branch
    under test is the classification, and that is worth testing on its own terms rather
    than through nine mocks.
    """

    def test_the_quicksight_wording_is_recognised(self):
        from lambda_source.handler import _refresh_is_unsupported

        assert _refresh_is_unsupported(UNREFRESHABLE)

    def test_it_sees_through_the_modificationerror_wrapper(self):
        """modifier.py wraps the QSClientError, so the marker is nested one level.

        Worth asserting explicitly: the classifier was written against the rollback path,
        where the exception arrives unwrapped. If wrapping had hidden the marker, the
        apply path would have gone on alerting and the shared helper would have looked
        like it covered both.
        """
        from lambda_source.handler import _refresh_is_unsupported

        assert isinstance(UNREFRESHABLE, ModificationError)
        assert "not refreshable" in str(UNREFRESHABLE)
        assert _refresh_is_unsupported(UNREFRESHABLE)

    @pytest.mark.parametrize(
        "message",
        [
            "CreateIngestion failed for dataset ds-1: ThrottlingException: rate exceeded",
            "CreateIngestion failed for dataset ds-1: IngestionId already exists",
            "CreateIngestion failed for dataset ds-1: AccessDeniedException",
        ],
        ids=["throttled", "duplicate_id", "access_denied"],
    )
    def test_a_genuine_failure_is_still_a_failure(self, message):
        """The point of matching the phrase and not the exception type.

        These are all things an operator can act on, and all of them must still alert.
        Suppressing them as "no refresh needed" would be the same mistake pointed the
        other way.
        """
        from lambda_source.handler import _refresh_is_unsupported

        assert not _refresh_is_unsupported(ModificationError(message, dataset_id="ds-1"))


def _apply(ingestion_raises=None):
    """Drive the real ``_execute_apply`` with its six collaborators stubbed.

    Built with ``object.__new__`` rather than the constructor, which wants config and
    live clients. The branch under test is a handful of lines deep in a long method, and
    the alternative — asserting on source text or re-implementing the control flow in the
    test — checks something other than the code that runs.
    """
    from lambda_source.handler import Orchestrator

    orchestrator = object.__new__(Orchestrator)

    modifier = create_autospec(DatasetModifier, instance=True)
    modifier.build_optimized_definition.return_value = {"PhysicalTableMap": {}}
    modifier.validate_modification.return_value = None
    modifier.apply_modification.return_value = {"Status": 200}
    if ingestion_raises:
        modifier.trigger_ingestion.side_effect = ingestion_raises

    notifier = create_autospec(NotificationPublisher, instance=True)
    reporter = create_autospec(ReportGenerator, instance=True)
    reporter.store_report.return_value = "p/reports/ds-1.json"
    snapshot_manager = create_autospec(SnapshotManager, instance=True)
    snapshot_manager.capture_snapshot.return_value = "p/snapshots/ds-1/now.json"

    orchestrator._modifier = modifier
    orchestrator._notifier = notifier
    orchestrator._reporter = reporter
    orchestrator._snapshot_manager = snapshot_manager
    orchestrator._config = SimpleNamespace(
        execution_mode="apply",
        safety_threshold_pct=0.7,
        quarantine_period_days=0,
        aws_account_id=ACCOUNT,
        aws_region="us-east-1",
    )

    classification = SimpleNamespace(
        column_name="col_a", classification="unused", data_type="STRING"
    )
    result = orchestrator._execute_apply(
        dataset_id="ds-1",
        dataset_name="sales_orders_SO.json",
        dataset_def={"PhysicalTableMap": {}, "OutputColumns": []},
        classifications=[classification],
        unused_classifications=[classification],
        unused_column_names=["col_a"],
        used_columns=set(),
        consumers=[],
    )
    return result, notifier, modifier


class TestTheApplyPathDoesNotAlertOnAnUnrefreshableDataset:
    def test_no_error_notification_is_published(self, caplog):
        with caplog.at_level(logging.INFO, logger="lambda_source.handler"):
            result, notifier, _ = _apply(ingestion_raises=UNREFRESHABLE)

        # assert_not_called() raises on its own; the message goes in the assert below so
        # it is not a dead string in a tuple expression.
        assert notifier.publish_error.call_count == 0, (
            "raised an SNS error alert because the dataset has no refreshable source. "
            "Nothing went wrong and nobody can act on it"
        )
        assert result.status == "optimized", "the removal itself still succeeded"

    def test_it_is_logged_as_information_not_a_warning(self, caplog):
        with caplog.at_level(logging.INFO, logger="lambda_source.handler"):
            _apply(ingestion_raises=UNREFRESHABLE)

        matching = [
            r for r in caplog.records if "none available for this source type" in r.getMessage()
        ]
        assert matching, "no line explains why no refresh started"
        assert matching[0].levelno == logging.INFO
        assert getattr(matching[0], "refresh_unsupported", None) is True

    def test_the_success_line_does_not_also_appear(self, caplog):
        """It was outside the try block, so it claimed success right after the failure.

        Two adjacent contradictory lines is worse than either alone, and now that
        structured logging actually reaches CloudWatch these lines are being read.
        """
        with caplog.at_level(logging.INFO, logger="lambda_source.handler"):
            _apply(ingestion_raises=UNREFRESHABLE)

        messages = [r.getMessage() for r in caplog.records]
        assert "SPICE ingestion triggered" not in messages


class TestGenuineFailuresStillAlert:
    def test_a_throttled_refresh_still_publishes_an_error(self, caplog):
        throttled = ModificationError(
            "CreateIngestion failed for dataset ds-1: ThrottlingException: rate exceeded",
            dataset_id="ds-1",
        )
        with caplog.at_level(logging.INFO, logger="lambda_source.handler"):
            result, notifier, _ = _apply(ingestion_raises=throttled)

        notifier.publish_error.assert_called_once()
        assert "ThrottlingException" in notifier.publish_error.call_args.kwargs["error_message"]
        assert result.status == "optimized", "a failed refresh must not fail the removal"

    def test_the_success_line_appears_when_the_refresh_actually_starts(self, caplog):
        with caplog.at_level(logging.INFO, logger="lambda_source.handler"):
            _, notifier, modifier = _apply()

        modifier.trigger_ingestion.assert_called_once_with("ds-1")
        notifier.publish_error.assert_not_called()
        assert "SPICE ingestion triggered" in [r.getMessage() for r in caplog.records]
