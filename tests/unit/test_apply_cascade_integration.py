# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The apply path drives the cascade only when asked, and reports what it did.

Unit coverage for the cascade lives in test_cascade_refresh.py. This file covers the
wiring in Orchestrator._execute_apply: that the config flag gates it, that a cascade's
per-child outcomes reach both the DatasetResult and the stored report, and that a child
refresh failure is surfaced without failing the parent optimisation.

Built the same way test_apply_refresh_reporting.py builds it — object.__new__ plus six
autospec'd collaborators — because the branch under test is a handful of lines deep in a
long method and standing up the whole pipeline would test everything except the wiring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import create_autospec

from lambda_source.modifier import ChildRefreshResult, DatasetModifier
from lambda_source.notifications import NotificationPublisher
from lambda_source.reporter import ReportGenerator
from lambda_source.snapshot import SnapshotManager

ACCOUNT = "123456789012"


def _child(dataset_id, depth=1, name=""):
    return SimpleNamespace(dataset_id=dataset_id, depth=depth, name=name or dataset_id)


def _apply(*, cascade, child_datasets, cascade_results=None):
    """Drive _execute_apply with cascade config and a stubbed cascade outcome.

    ``cascade_results`` is what the modifier's ``trigger_cascading_refresh`` returns; the
    default is a single successful child so the common path needs no setup.
    """
    from lambda_source.handler import Orchestrator

    orchestrator = object.__new__(Orchestrator)

    modifier = create_autospec(DatasetModifier, instance=True)
    modifier.build_optimized_definition.return_value = {"PhysicalTableMap": {}}
    modifier.validate_modification.return_value = None
    modifier.apply_modification.return_value = {"Status": 200}
    modifier.trigger_ingestion.return_value = {"Status": 201}
    if cascade_results is None:
        cascade_results = [
            ChildRefreshResult(
                dataset_id="c1", name="c1", depth=1, refresh_started=True,
                ingestion_id="optimiser-cascade-abc",
            )
        ]
    modifier.trigger_cascading_refresh.return_value = cascade_results

    notifier = create_autospec(NotificationPublisher, instance=True)
    reporter = create_autospec(ReportGenerator, instance=True)
    reporter.generate_report.return_value = {"report": True}
    reporter.store_report.return_value = "p/reports/ds-1.json"
    reporter.estimate_spice_savings.return_value = 1234
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
        cascade_child_refresh=cascade,
    )

    classification = SimpleNamespace(
        column_name="col_a", classification="unused", data_type="STRING"
    )
    result = orchestrator._execute_apply(
        dataset_id="ds-1",
        dataset_name="orders",
        dataset_def={"PhysicalTableMap": {}, "OutputColumns": []},
        classifications=[classification],
        unused_classifications=[classification],
        unused_column_names=["col_a"],
        used_columns=set(),
        consumers=[],
        child_datasets=child_datasets,
    )
    return result, modifier, reporter, notifier


class TestTheFlagGatesTheCascade:
    def test_cascade_runs_when_enabled_and_children_exist(self):
        _, modifier, _, _ = _apply(cascade=True, child_datasets=[_child("c1")])
        modifier.trigger_cascading_refresh.assert_called_once()

    def test_cascade_does_not_run_when_disabled(self):
        _, modifier, _, _ = _apply(cascade=False, child_datasets=[_child("c1")])
        modifier.trigger_cascading_refresh.assert_not_called()

    def test_cascade_does_not_run_without_children(self):
        _, modifier, _, _ = _apply(cascade=True, child_datasets=[])
        modifier.trigger_cascading_refresh.assert_not_called()

    def test_the_parent_optimisation_still_succeeds_either_way(self):
        result, _, _, _ = _apply(cascade=True, child_datasets=[_child("c1")])
        assert result.status == "optimized"


class TestOutcomesAreReported:
    def test_results_land_on_the_dataset_result(self):
        result, _, _, _ = _apply(cascade=True, child_datasets=[_child("c1")])
        assert len(result.child_refresh_results) == 1
        assert result.child_refresh_results[0]["dataset_id"] == "c1"
        assert result.child_refresh_results[0]["refresh_started"] is True

    def test_results_are_passed_to_the_report(self):
        _, _, reporter, _ = _apply(cascade=True, child_datasets=[_child("c1")])
        kwargs = reporter.generate_report.call_args.kwargs
        assert "child_refresh_results" in kwargs
        assert len(kwargs["child_refresh_results"]) == 1

    def test_no_cascade_means_an_empty_results_list(self):
        result, _, _, _ = _apply(cascade=False, child_datasets=[_child("c1")])
        assert result.child_refresh_results == []


class TestAFailingChildIsSurfacedButNotFatal:
    def test_a_failed_child_refresh_publishes_one_summary_alert(self):
        failed = [
            ChildRefreshResult(dataset_id="c1", refresh_started=True),
            ChildRefreshResult(
                dataset_id="c2", error="ThrottlingError: rate exceeded"
            ),
        ]
        result, _, _, notifier = _apply(
            cascade=True, child_datasets=[_child("c1"), _child("c2")],
            cascade_results=failed,
        )

        # One summary alert naming the children to refresh by hand — not one per child.
        notifier.publish_error.assert_called_once()
        message = notifier.publish_error.call_args.kwargs["error_message"]
        assert "c2" in message
        assert result.status == "optimized", "a child failure must not fail the parent"

    def test_all_children_succeeding_raises_no_alert(self):
        ok = [
            ChildRefreshResult(dataset_id="c1", refresh_started=True),
            ChildRefreshResult(dataset_id="c2", refresh_unsupported=True),
        ]
        _, _, _, notifier = _apply(
            cascade=True, child_datasets=[_child("c1"), _child("c2")],
            cascade_results=ok,
        )
        notifier.publish_error.assert_not_called()
