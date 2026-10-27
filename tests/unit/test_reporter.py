# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the ReportGenerator module.

Tests dry run report exclusion of UpdateDataSet response, apply mode report
inclusion of snapshot key and API response, and zero unused columns producing
a no-opportunity statement.

Requirements: 5.2, 5.4, 10.1, 10.3
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lambda_source.models import ColumnClassification, ColumnStatus, Consumer
from lambda_source.reporter import ReportGenerator


@pytest.fixture
def reporter() -> ReportGenerator:
    """Create a ReportGenerator with a mock S3 client."""
    mock_s3 = MagicMock()
    return ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
        s3_client=mock_s3,
        average_rows=1_000_000,
    )


@pytest.fixture
def sample_classifications() -> list[ColumnClassification]:
    """Sample column classifications with used and unused columns."""
    return [
        ColumnClassification(
            column_name="customer_id",
            data_type="STRING",
            status=ColumnStatus.USED.value,
            reasons=["referenced by consumer"],
        ),
        ColumnClassification(
            column_name="order_id",
            data_type="INTEGER",
            status=ColumnStatus.JOIN_KEY.value,
            reasons=["retained: join key"],
        ),
        ColumnClassification(
            column_name="internal_code",
            data_type="STRING",
            status=ColumnStatus.UNUSED.value,
            reasons=["no consumer references"],
        ),
        ColumnClassification(
            column_name="legacy_flag",
            data_type="INTEGER",
            status=ColumnStatus.UNUSED.value,
            reasons=["no consumer references"],
        ),
    ]


@pytest.fixture
def sample_consumers() -> list[Consumer]:
    """Sample consumer list."""
    return [
        Consumer(
            arn="arn:aws:quicksight:us-east-1:123456789012:analysis/abc",
            consumer_type="analysis",
            name="Sales Analysis",
            dataset_identifier="ds_sales",
            status="CREATION_SUCCESSFUL",
        ),
    ]


# ---------------------------------------------------------------------------
# Dry Run Mode — excludes UpdateDataSet response and snapshot key
# ---------------------------------------------------------------------------


class TestDryRunReportExcludesUpdateResponse:
    """Req 5.4: Dry run mode SHALL NOT include UpdateDataSet response or snapshot key."""

    def test_dry_run_report_has_null_update_api_response(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Dry run report update_api_response field is None."""
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key="some/key.json",
            update_response={"ResponseMetadata": {"HTTPStatusCode": 200}},
        )

        assert report["update_api_response"] is None

    def test_dry_run_report_has_null_snapshot_s3_key(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Dry run report snapshot_s3_key field is None even when provided."""
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key="snapshots/ds-123/2024-01-15T10:30:00Z.json",
            update_response={"Status": 200},
        )

        assert report["snapshot_s3_key"] is None

    def test_dry_run_report_still_includes_required_fields(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Dry run report includes all required reporting fields (Req 10.1)."""
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        assert report["dataset_id"] == "ds-123"
        assert report["dataset_name"] == "Sales Transactions"
        assert report["execution_mode"] == "dry_run"
        assert report["total_columns"] == 4
        assert report["used_columns_count"] == 2  # used + join_key
        assert report["unused_columns_count"] == 2
        assert report["join_key_columns_count"] == 1
        assert report["estimated_spice_savings_bytes"] > 0
        assert "execution_id" in report
        assert "execution_timestamp" in report
        assert "column_details" in report


# ---------------------------------------------------------------------------
# Apply Mode — includes snapshot key and API response
# ---------------------------------------------------------------------------


class TestApplyModeReportIncludesSnapshotAndResponse:
    """Req 10.3: Apply mode report includes snapshot S3 key and UpdateDataSet response."""

    def test_apply_mode_report_includes_snapshot_key(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Apply mode report includes the snapshot_s3_key."""
        snapshot_key = "test-prefix/snapshots/ds-123/2024-01-15T10:30:00Z.json"
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="apply",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key=snapshot_key,
            update_response={"ResponseMetadata": {"HTTPStatusCode": 200}},
        )

        assert report["snapshot_s3_key"] == snapshot_key

    def test_apply_mode_report_includes_update_api_response(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Apply mode report includes the UpdateDataSet API response."""
        update_resp = {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "Status": 200,
            "DataSetId": "ds-123",
        }
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="apply",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key="prefix/snapshots/ds-123/ts.json",
            update_response=update_resp,
        )

        assert report["update_api_response"] == update_resp

    def test_apply_mode_report_with_none_snapshot_key(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Apply mode report with None snapshot_s3_key keeps None."""
        report = reporter.generate_report(
            dataset_id="ds-123",
            dataset_name="Sales Transactions",
            execution_mode="apply",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key=None,
            update_response={"Status": 200},
        )

        assert report["snapshot_s3_key"] is None

    def test_apply_mode_report_includes_all_required_fields(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Apply mode report contains all required fields including per-column detail."""
        report = reporter.generate_report(
            dataset_id="ds-apply",
            dataset_name="Apply Dataset",
            execution_mode="apply",
            classifications=sample_classifications,
            consumer_count=2,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
            snapshot_s3_key="key.json",
            update_response={"Status": 200},
        )

        # Core fields (Req 10.1)
        assert report["dataset_id"] == "ds-apply"
        assert report["total_columns"] == 4
        assert report["used_columns_count"] == 2
        assert report["unused_columns_count"] == 2
        assert report["estimated_spice_savings_bytes"] >= 0
        assert report["execution_mode"] == "apply"

        # Per-column detail (Req 10.2)
        assert len(report["column_details"]) == 4
        for detail in report["column_details"]:
            assert "column_name" in detail
            assert "data_type" in detail
            assert "status" in detail
            assert "reason" in detail


# ---------------------------------------------------------------------------
# Zero Unused Columns — no-opportunity statement
# ---------------------------------------------------------------------------


class TestZeroUnusedColumnsReport:
    """Req 5.2: Zero unused columns produces a report showing no optimization opportunity."""

    def test_zero_unused_columns_has_zero_savings(
        self, reporter: ReportGenerator, sample_consumers
    ):
        """When all columns are used, estimated savings is zero."""
        all_used_classifications = [
            ColumnClassification(
                column_name="col_a",
                data_type="STRING",
                status=ColumnStatus.USED.value,
                reasons=["referenced by consumer"],
            ),
            ColumnClassification(
                column_name="col_b",
                data_type="INTEGER",
                status=ColumnStatus.USED.value,
                reasons=["referenced by consumer"],
            ),
            ColumnClassification(
                column_name="col_c",
                data_type="STRING",
                status=ColumnStatus.JOIN_KEY.value,
                reasons=["retained: join key"],
            ),
        ]

        report = reporter.generate_report(
            dataset_id="ds-no-op",
            dataset_name="Fully Used Dataset",
            execution_mode="dry_run",
            classifications=all_used_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        assert report["unused_columns_count"] == 0
        assert report["estimated_spice_savings_bytes"] == 0

    def test_zero_unused_columns_no_unused_in_details(
        self, reporter: ReportGenerator, sample_consumers
    ):
        """Column details contain no entries with status 'unused'."""
        all_used_classifications = [
            ColumnClassification(
                column_name="col_x",
                data_type="STRING",
                status=ColumnStatus.USED.value,
                reasons=["referenced by consumer"],
            ),
            ColumnClassification(
                column_name="col_y",
                data_type="INTEGER",
                status=ColumnStatus.USED.value,
                reasons=["referenced by consumer"],
            ),
        ]

        report = reporter.generate_report(
            dataset_id="ds-full",
            dataset_name="Full Dataset",
            execution_mode="dry_run",
            classifications=all_used_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        unused_details = [
            d for d in report["column_details"] if d["status"] == "unused"
        ]
        assert len(unused_details) == 0

    def test_zero_unused_columns_threshold_not_breached(
        self, reporter: ReportGenerator, sample_consumers
    ):
        """Report with zero unused columns should not show threshold breach."""
        all_used_classifications = [
            ColumnClassification(
                column_name=f"col_{i}",
                data_type="STRING",
                status=ColumnStatus.USED.value,
                reasons=["referenced by consumer"],
            )
            for i in range(5)
        ]

        report = reporter.generate_report(
            dataset_id="ds-safe",
            dataset_name="Safe Dataset",
            execution_mode="dry_run",
            classifications=all_used_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        assert report["threshold_breached"] is False
        assert report["unused_columns_count"] == 0


# ---------------------------------------------------------------------------
# Report storage to S3
# ---------------------------------------------------------------------------


class TestReportStorage:
    """Test report storage to S3 with correct key pattern (Req 10.4)."""

    def test_store_report_calls_s3_put_object(self, reporter: ReportGenerator):
        """store_report calls S3 put_object with correct parameters."""
        report = {"execution_id": "test-id", "dataset_id": "ds-123"}

        reporter.store_report(report, "ds-123")

        reporter._s3_client.put_object.assert_called_once()
        call_kwargs = reporter._s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert "reports/" in call_kwargs["Key"]
        assert "ds-123.json" in call_kwargs["Key"]
        assert call_kwargs["ContentType"] == "application/json"

    def test_store_report_key_pattern(self, reporter: ReportGenerator):
        """S3 key follows pattern: {prefix}reports/{date}/{dataset_id}.json."""
        report = {"execution_id": "test-id"}

        s3_key = reporter.store_report(report, "ds-456")

        assert s3_key.startswith("test-prefix/reports/")
        assert s3_key.endswith("/ds-456.json")

    def test_store_report_body_is_valid_json(self, reporter: ReportGenerator):
        """Report body stored in S3 is valid JSON."""
        report = {"execution_id": "abc", "total_columns": 10}

        reporter.store_report(report, "ds-789")

        call_kwargs = reporter._s3_client.put_object.call_args[1]
        body = call_kwargs["Body"]
        parsed = json.loads(body)
        assert parsed["execution_id"] == "abc"
        assert parsed["total_columns"] == 10


# ---------------------------------------------------------------------------
# SPICE savings estimation
# ---------------------------------------------------------------------------


class TestSpiceSavingsEstimation:
    """Test SPICE savings estimation logic."""

    def test_empty_unused_columns_returns_zero(self, reporter: ReportGenerator):
        """No unused columns means zero savings."""
        result = reporter.estimate_spice_savings([])
        assert result == 0

    def test_single_string_column_savings(self, reporter: ReportGenerator):
        """Single STRING column estimates 100 bytes/row * 1M rows."""
        unused = [
            ColumnClassification(
                column_name="col",
                data_type="STRING",
                status=ColumnStatus.UNUSED.value,
            )
        ]
        result = reporter.estimate_spice_savings(unused)
        assert result == 100 * 1_000_000

    def test_multiple_columns_accumulate_savings(self, reporter: ReportGenerator):
        """Multiple unused columns accumulate savings."""
        unused = [
            ColumnClassification(
                column_name="str_col",
                data_type="STRING",
                status=ColumnStatus.UNUSED.value,
            ),
            ColumnClassification(
                column_name="int_col",
                data_type="INTEGER",
                status=ColumnStatus.UNUSED.value,
            ),
        ]
        result = reporter.estimate_spice_savings(unused)
        expected = (100 * 1_000_000) + (8 * 1_000_000)
        assert result == expected

    def test_unknown_data_type_uses_default(self, reporter: ReportGenerator):
        """Unknown data type defaults to 100 bytes/row."""
        unused = [
            ColumnClassification(
                column_name="unknown_col",
                data_type="CUSTOM_TYPE",
                status=ColumnStatus.UNUSED.value,
            )
        ]
        result = reporter.estimate_spice_savings(unused)
        assert result == 100 * 1_000_000


# ---------------------------------------------------------------------------
# Risk indicators
# ---------------------------------------------------------------------------


class TestRiskIndicators:
    """Test risk indicator generation in reports."""

    def test_shared_dataset_risk_indicator(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Consumer count > 1 produces 'shared dataset' risk indicator."""
        report = reporter.generate_report(
            dataset_id="ds-shared",
            dataset_name="Shared Dataset",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=3,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        assert "shared dataset" in report["risk_indicators"]

    def test_no_consumers_risk_indicator(
        self, reporter: ReportGenerator, sample_classifications
    ):
        """Consumer count = 0 produces 'no consumers found' risk indicator."""
        report = reporter.generate_report(
            dataset_id="ds-orphan",
            dataset_name="Orphan Dataset",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=0,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=[],
        )

        assert "no consumers found" in report["risk_indicators"]

    def test_single_consumer_no_risk_indicator(
        self, reporter: ReportGenerator, sample_classifications, sample_consumers
    ):
        """Single consumer produces no risk indicators."""
        report = reporter.generate_report(
            dataset_id="ds-solo",
            dataset_name="Solo Dataset",
            execution_mode="dry_run",
            classifications=sample_classifications,
            consumer_count=1,
            safety_threshold_pct=0.7,
            threshold_breached=False,
            consumers=sample_consumers,
        )

        assert report["risk_indicators"] == []
