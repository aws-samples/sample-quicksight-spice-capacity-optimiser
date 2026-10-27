# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the unreachable-consumer guard.

``ConsumerDiscovery.discover_consumers`` collected unreadable consumers into a local
list and, on the ``skip_unreachable=True`` path, logged a warning and returned
``list[Consumer]`` with those entries simply missing. The caller had no way to tell a
dataset whose only dashboard returned ``AccessDenied`` from a dataset that genuinely has
no dashboards.

The consequence ran in the dangerous direction. The unreadable dashboard's columns were
never scanned, so they were classified unused and became removal candidates — removing
them breaks the dashboard nobody could read. The method's own docstring promised the
opposite all along: "columns found only in unreachable consumers will be conservatively
marked as used". Nothing implemented it.

``ChildDatasetDiscovery`` had the identical defect for the same reason.

The load-bearing test is
``TestRemovalCandidates::test_an_unreadable_consumer_yields_zero_removal_candidates`` —
the property that was actually violated.
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from lambda_source.attribution import UsageAttribution
from lambda_source.discovery import ConsumerDiscovery, DiscoveryError
from lambda_source.models import ColumnStatus, ColumnUsageSource, OptimiserConfig
from lambda_source.qs_client import AccessDeniedError, QSClient
from lambda_source.reporter import ReportGenerator

DASHBOARD = {
    "DashboardId": "dash-1",
    "Arn": "arn:aws:quicksight:us-east-1:123456789012:dashboard/dash-1",
    "Name": "The Only Dashboard",
    "Status": "CREATION_SUCCESSFUL",
}

COLUMNS = [
    {"Name": "col_a", "Type": "STRING"},
    {"Name": "col_b", "Type": "STRING"},
    {"Name": "col_c", "Type": "INTEGER"},
]


def unreadable_dashboard_client() -> MagicMock:
    qs = create_autospec(QSClient, instance=True)
    qs.list_analyses.return_value = []
    qs.list_dashboards.return_value = [DASHBOARD]
    qs.describe_dashboard_definition.side_effect = AccessDeniedError(
        "User is not authorized to access this dashboard",
        error_code="AccessDeniedException",
    )
    return qs


class TestAccumulator:
    def test_unreachable_consumers_reach_the_caller(self):
        discovery = ConsumerDiscovery(
            qs_client=unreadable_dashboard_client(), account_id="9" * 12
        )
        unreachable: list[DiscoveryError] = []

        consumers = discovery.discover_consumers(
            "ds-1", skip_unreachable=True, unreachable=unreachable
        )

        assert consumers == []
        assert len(unreachable) == 1
        assert unreachable[0].error_code == "AccessDeniedException"
        assert unreachable[0].arn == DASHBOARD["Arn"]

    def test_accumulator_is_populated_even_when_the_run_halts(self):
        """So a caller that catches ConsumerDiscoveryError can still see what failed."""
        from lambda_source.discovery import ConsumerDiscoveryError

        discovery = ConsumerDiscovery(
            qs_client=unreadable_dashboard_client(), account_id="9" * 12
        )
        unreachable: list[DiscoveryError] = []

        with pytest.raises(ConsumerDiscoveryError):
            discovery.discover_consumers(
                "ds-1", skip_unreachable=False, unreachable=unreachable
            )

        assert len(unreachable) == 1

    def test_omitting_the_accumulator_still_works(self):
        """Backward compatible: existing callers pass nothing."""
        discovery = ConsumerDiscovery(
            qs_client=unreadable_dashboard_client(), account_id="9" * 12
        )
        assert discovery.discover_consumers("ds-1", skip_unreachable=True) == []

    def test_a_clean_run_leaves_the_accumulator_empty(self):
        qs = create_autospec(QSClient, instance=True)
        qs.list_analyses.return_value = []
        qs.list_dashboards.return_value = []
        discovery = ConsumerDiscovery(qs_client=qs, account_id="9" * 12)
        unreachable: list[DiscoveryError] = []

        discovery.discover_consumers(
            "ds-1", skip_unreachable=True, unreachable=unreachable
        )

        assert unreachable == []


class TestRemovalCandidates:
    """End-to-end through the orchestrator, which is where the guard lives."""

    def _orchestrator(self, qs_client: MagicMock):
        from lambda_source.handler import Orchestrator

        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="dry_run",
            s3_bucket="bucket",
            s3_prefix="prefix/",
            aws_account_id="9" * 12,
            aws_region="us-east-1",
            # The flag that makes the bug reachable. An operator turns this on right
            # after hitting a permissions error, which is when the risk is highest.
            skip_unreachable_consumers=True,
        )
        with patch("lambda_source.handler.QSClient", return_value=qs_client), patch(
            "lambda_source.handler.boto3"
        ), patch("lambda_source.reporter.boto3"), patch(
            "lambda_source.snapshot.boto3"
        ), patch(
            "lambda_source.notifications.boto3"
        ), patch(
            "lambda_source.dataset_lock.boto3"
        ):
            orch = Orchestrator(config)
            orch._reporter._s3_client = MagicMock()
            captured: dict = {}
            original = orch._reporter.generate_report

            def spy(**kwargs):
                report = original(**kwargs)
                captured["report"] = report
                return report

            orch._reporter.generate_report = spy
            orch._reporter.store_report = lambda report, dsid: "key.json"
            result = orch.process_dataset("ds-1")
        return result, captured.get("report")

    def _client_with_unreadable_dashboard(self) -> MagicMock:
        qs = unreadable_dashboard_client()
        qs.describe_dataset.return_value = {
            "DataSetId": "ds-1",
            "Name": "orders",
            "ImportMode": "SPICE",
            "OutputColumns": COLUMNS,
            "PhysicalTableMap": {"pt": {"RelationalTable": {"InputColumns": COLUMNS}}},
            "LogicalTableMap": {},
        }
        qs.list_datasets.return_value = []
        return qs

    def test_an_unreadable_consumer_yields_zero_removal_candidates(self):
        """The property that was violated. Without the guard all three columns are
        unreferenced, classified unused, and proposed for removal."""
        result, report = self._orchestrator(self._client_with_unreadable_dashboard())

        assert result.unused_columns_count == 0
        assert report is not None
        unused = [
            c for c in report["column_details"] if c["status"] == ColumnStatus.UNUSED.value
        ]
        assert unused == []

    def test_every_column_is_marked_used(self):
        _, report = self._orchestrator(self._client_with_unreadable_dashboard())

        statuses = {c["column_name"]: c["status"] for c in report["column_details"]}
        assert set(statuses) == {"col_a", "col_b", "col_c"}
        assert all(s == ColumnStatus.USED.value for s in statuses.values())

    def test_columns_are_flagged_as_retained_without_proof(self):
        """They are protected, but nothing demonstrated they are needed."""
        _, report = self._orchestrator(self._client_with_unreadable_dashboard())

        assert all(
            c["retained_without_proof"] for c in report["column_details"]
        )

    def test_the_reason_names_the_unreadable_consumer_and_error(self):
        _, report = self._orchestrator(self._client_with_unreadable_dashboard())

        reason = report["column_details"][0]["reason"]
        assert "could not be read" in reason
        assert "AccessDeniedException" in reason

    def test_risk_indicator_is_distinct_from_no_consumers_found(self):
        """The whole point. These two were indistinguishable, and the remediation is a
        permissions fix rather than an optimisation."""
        _, report = self._orchestrator(self._client_with_unreadable_dashboard())

        joined = " | ".join(report["risk_indicators"])
        assert "could not be read" in joined
        assert "Check permissions" in joined


class TestRiskIndicatorWording:
    def test_unreachable_and_absent_consumers_read_differently(self):
        reporter = ReportGenerator("b", "p/", s3_client=MagicMock())

        attribution = UsageAttribution()
        attribution.record(
            "col_a", ColumnUsageSource.UNREACHABLE_CONSUMER.value, requester_name="dash-1"
        )
        from lambda_source.analyzer import UsageAnalyzer

        classifications = UsageAnalyzer().classify_columns(
            all_columns=[{"Name": "col_a", "Type": "STRING"}],
            used_columns={"col_a"},
            join_key_columns=set(),
            calc_field_input_columns=set(),
            attribution=attribution,
        )

        unreachable_case = reporter._get_risk_indicators(
            0, classifications=classifications
        )
        absent_case = reporter._get_risk_indicators(0, classifications=[])

        assert unreachable_case != absent_case
        # "no consumers found" is still reported, because it is also true - but it is
        # no longer the only thing said.
        assert "no consumers found" in unreachable_case
        assert any("could not be read" in i for i in unreachable_case)
        assert not any("could not be read" in i for i in absent_case)


class TestChildDiscoveryParity:
    """The same defect existed in ChildDatasetDiscovery. Audit item from the backlog."""

    def test_child_discovery_exposes_unreachable_ids(self):
        from lambda_source.child_discovery import ChildDatasetDiscovery

        qs = create_autospec(QSClient, instance=True)
        qs.list_datasets.return_value = [
            {"DataSetId": "child-1", "Name": "child", "ImportMode": "SPICE"}
        ]
        qs.describe_dataset.side_effect = AccessDeniedError(
            "nope", error_code="AccessDeniedException"
        )

        discovery = ChildDatasetDiscovery(qs_client=qs, account_id="9" * 12, region="us-east-1")
        unreachable: list[str] = []

        discovery.discover_child_datasets(
            target_dataset_id="ds-1",
            skip_unreachable=True,
            unreachable_out=unreachable,
        )

        assert unreachable == ["child-1"]

    def test_omitting_the_accumulator_still_works(self):
        from lambda_source.child_discovery import ChildDatasetDiscovery

        qs = create_autospec(QSClient, instance=True)
        qs.list_datasets.return_value = []
        discovery = ChildDatasetDiscovery(qs_client=qs, account_id="9" * 12, region="us-east-1")

        assert discovery.discover_child_datasets(target_dataset_id="ds-1") == []
