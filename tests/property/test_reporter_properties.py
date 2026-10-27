# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for reporter module.

Feature: spice-dataset-optimiser
Property 15: Report field completeness

Validates: Requirements 5.2, 10.1, 10.2
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from lambda_source.models import ColumnClassification, ColumnStatus, Consumer, ExecutionMode
from lambda_source.reporter import ReportGenerator
from tests.property.strategies import (
    st_column_name,
    st_data_type,
    st_dataset_id,
    st_s3_bucket_name,
    st_arn,
)


# ---------------------------------------------------------------------------
# Strategies for report generation inputs
# ---------------------------------------------------------------------------


@st.composite
def st_column_classification(draw: st.DrawFn) -> ColumnClassification:
    """Generate a valid ColumnClassification instance."""
    name = draw(st_column_name())
    data_type = draw(st_data_type())
    status = draw(st.sampled_from([s.value for s in ColumnStatus]))
    num_reasons = draw(st.integers(min_value=0, max_value=3))
    reasons = [
        draw(st.text(min_size=3, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz _-:"))
        for _ in range(num_reasons)
    ]
    return ColumnClassification(
        column_name=name,
        data_type=data_type,
        status=status,
        reasons=reasons,
    )


@st.composite
def st_consumer_obj(draw: st.DrawFn) -> Consumer:
    """Generate a valid Consumer dataclass instance."""
    consumer_type = draw(st.sampled_from(["analysis", "dashboard"]))
    return Consumer(
        arn=draw(st_arn(resource_type=consumer_type)),
        consumer_type=consumer_type,
        name=draw(st.text(min_size=3, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-")),
        dataset_identifier=draw(st.text(min_size=3, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789_")),
        status="CREATION_SUCCESSFUL",
    )


@st.composite
def st_report_inputs(draw: st.DrawFn) -> dict:
    """Generate a complete set of inputs for ReportGenerator.generate_report()."""
    dataset_id = draw(st_dataset_id())
    dataset_name = draw(st.text(min_size=3, max_size=40, alphabet="abcdefghijklmnopqrstuvwxyz0123456789 _-"))
    execution_mode = draw(st.sampled_from([m.value for m in ExecutionMode]))

    num_classifications = draw(st.integers(min_value=1, max_value=20))
    classifications = [draw(st_column_classification()) for _ in range(num_classifications)]

    num_consumers = draw(st.integers(min_value=0, max_value=5))
    consumers = [draw(st_consumer_obj()) for _ in range(num_consumers)]

    safety_threshold_pct = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    threshold_breached = draw(st.booleans())

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "execution_mode": execution_mode,
        "classifications": classifications,
        "consumer_count": num_consumers,
        "safety_threshold_pct": safety_threshold_pct,
        "threshold_breached": threshold_breached,
        "consumers": consumers,
    }


# ---------------------------------------------------------------------------
# Property 15: Report field completeness
# ---------------------------------------------------------------------------


@given(inputs=st_report_inputs())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_report_contains_all_required_top_level_fields(inputs: dict) -> None:
    """Property 15: Report field completeness - top-level required fields.

    **Validates: Requirements 5.2, 10.1, 10.2**

    For any execution result, the Optimisation_Report SHALL contain all required
    fields: dataset_id, total_columns, used_columns_count, unused_columns_count,
    estimated_spice_savings_bytes, and execution_mode.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    report = generator.generate_report(**inputs)

    # Verify all required top-level fields are present
    required_fields = [
        "dataset_id",
        "total_columns",
        "used_columns_count",
        "unused_columns_count",
        "estimated_spice_savings_bytes",
        "execution_mode",
    ]

    for field in required_fields:
        assert field in report, (
            f"Required field '{field}' missing from Optimisation_Report. "
            f"Report keys: {list(report.keys())}"
        )


@given(inputs=st_report_inputs())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_report_contains_column_details(inputs: dict) -> None:
    """Property 15: Report field completeness - column_details present.

    **Validates: Requirements 5.2, 10.1, 10.2**

    For any execution result, the Optimisation_Report SHALL contain
    per-column detail entries in a 'column_details' field.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    report = generator.generate_report(**inputs)

    assert "column_details" in report, (
        "Required field 'column_details' missing from Optimisation_Report."
    )
    assert isinstance(report["column_details"], list), (
        f"'column_details' should be a list, got {type(report['column_details'])}"
    )
    # Number of column details should match number of classifications
    assert len(report["column_details"]) == len(inputs["classifications"]), (
        f"Expected {len(inputs['classifications'])} column detail entries, "
        f"got {len(report['column_details'])}"
    )


@given(inputs=st_report_inputs())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_report_column_details_have_required_fields(inputs: dict) -> None:
    """Property 15: Report field completeness - per-column detail fields.

    **Validates: Requirements 5.2, 10.1, 10.2**

    For any execution result, each per-column detail entry SHALL contain
    column_name, data_type, status, and reason.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    report = generator.generate_report(**inputs)

    required_column_fields = ["column_name", "data_type", "status", "reason"]

    for i, detail in enumerate(report["column_details"]):
        for field in required_column_fields:
            assert field in detail, (
                f"Required field '{field}' missing from column_details[{i}]. "
                f"Detail keys: {list(detail.keys())}"
            )
        # Verify values are non-empty strings
        assert isinstance(detail["column_name"], str) and len(detail["column_name"]) > 0, (
            f"column_details[{i}]['column_name'] should be a non-empty string, "
            f"got: {detail['column_name']!r}"
        )
        assert isinstance(detail["data_type"], str) and len(detail["data_type"]) > 0, (
            f"column_details[{i}]['data_type'] should be a non-empty string, "
            f"got: {detail['data_type']!r}"
        )
        assert isinstance(detail["status"], str) and len(detail["status"]) > 0, (
            f"column_details[{i}]['status'] should be a non-empty string, "
            f"got: {detail['status']!r}"
        )
        assert isinstance(detail["reason"], str) and len(detail["reason"]) > 0, (
            f"column_details[{i}]['reason'] should be a non-empty string, "
            f"got: {detail['reason']!r}"
        )


@given(inputs=st_report_inputs())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_report_field_values_match_inputs(inputs: dict) -> None:
    """Property 15: Report field completeness - field values are correct.

    **Validates: Requirements 5.2, 10.1, 10.2**

    For any execution result, the report field values SHALL accurately reflect
    the input data (dataset_id, execution_mode, column counts).
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    report = generator.generate_report(**inputs)

    # Verify dataset_id matches input
    assert report["dataset_id"] == inputs["dataset_id"], (
        f"dataset_id mismatch: expected {inputs['dataset_id']!r}, got {report['dataset_id']!r}"
    )

    # Verify execution_mode matches input
    assert report["execution_mode"] == inputs["execution_mode"], (
        f"execution_mode mismatch: expected {inputs['execution_mode']!r}, "
        f"got {report['execution_mode']!r}"
    )

    # Verify total_columns matches number of classifications
    assert report["total_columns"] == len(inputs["classifications"]), (
        f"total_columns mismatch: expected {len(inputs['classifications'])}, "
        f"got {report['total_columns']}"
    )

    # Verify estimated_spice_savings_bytes is non-negative
    assert report["estimated_spice_savings_bytes"] >= 0, (
        f"estimated_spice_savings_bytes should be non-negative, "
        f"got {report['estimated_spice_savings_bytes']}"
    )

    # Verify column counts add up correctly
    # used_columns_count includes join_key columns
    assert report["used_columns_count"] + report["unused_columns_count"] <= report["total_columns"], (
        f"Column counts don't add up: used={report['used_columns_count']}, "
        f"unused={report['unused_columns_count']}, total={report['total_columns']}"
    )


# ---------------------------------------------------------------------------
# Property 12: Shared dataset risk indicator
# ---------------------------------------------------------------------------


@given(consumer_count=st.integers(min_value=2, max_value=1000))
@settings(max_examples=100)
def test_shared_dataset_indicator_present_when_multiple_consumers(
    consumer_count: int,
) -> None:
    """Property 12: Shared dataset risk indicator - present when consumer count > 1.

    **Validates: Requirements 8.3**

    For any dataset with consumer count > 1, report SHALL include
    "shared dataset" risk indicator.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    risk_indicators = generator._get_risk_indicators(consumer_count)

    assert "shared dataset" in risk_indicators, (
        f"Expected 'shared dataset' risk indicator for consumer_count={consumer_count}, "
        f"but got: {risk_indicators}"
    )


@given(consumer_count=st.integers(min_value=0, max_value=1))
@settings(max_examples=100)
def test_shared_dataset_indicator_absent_when_one_or_zero_consumers(
    consumer_count: int,
) -> None:
    """Property 12: Shared dataset risk indicator - absent when consumer count <= 1.

    **Validates: Requirements 8.3**

    For datasets with exactly one or zero consumers, "shared dataset"
    SHALL NOT be present.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    risk_indicators = generator._get_risk_indicators(consumer_count)

    assert "shared dataset" not in risk_indicators, (
        f"Expected 'shared dataset' risk indicator to be absent for "
        f"consumer_count={consumer_count}, but got: {risk_indicators}"
    )


@given(consumer_count=st.integers(min_value=0, max_value=1000))
@settings(max_examples=100)
def test_shared_dataset_indicator_via_generate_report(
    consumer_count: int,
) -> None:
    """Property 12: Shared dataset risk indicator - validated through generate_report.

    **Validates: Requirements 8.3**

    When generating a full report, the risk_indicators field SHALL contain
    "shared dataset" if and only if consumer_count > 1.
    """
    generator = ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
    )

    # Create minimal classifications for report generation
    classifications = [
        ColumnClassification(
            column_name="col_a",
            data_type="STRING",
            status="used",
            reasons=["referenced by consumer"],
        ),
    ]

    report = generator.generate_report(
        dataset_id="test-dataset-001",
        dataset_name="Test Dataset",
        execution_mode="dry_run",
        classifications=classifications,
        consumer_count=consumer_count,
        safety_threshold_pct=0.7,
        threshold_breached=False,
        consumers=[],
    )

    risk_indicators = report["risk_indicators"]

    if consumer_count > 1:
        assert "shared dataset" in risk_indicators, (
            f"Expected 'shared dataset' in report risk_indicators for "
            f"consumer_count={consumer_count}, but got: {risk_indicators}"
        )
    else:
        assert "shared dataset" not in risk_indicators, (
            f"Expected 'shared dataset' NOT in report risk_indicators for "
            f"consumer_count={consumer_count}, but got: {risk_indicators}"
        )


# ---------------------------------------------------------------------------
# Property 8: SPICE savings estimation consistency
# ---------------------------------------------------------------------------


@st.composite
def st_unused_column_classification(draw: st.DrawFn) -> ColumnClassification:
    """Generate a single unused ColumnClassification with a known data type."""
    name = draw(st_column_name())
    data_type = draw(st_data_type())
    return ColumnClassification(
        column_name=name,
        data_type=data_type,
        status="unused",
        reasons=["no consumer references"],
    )


@st.composite
def st_unused_column_list(
    draw: st.DrawFn, min_size: int = 0, max_size: int = 20
) -> list[ColumnClassification]:
    """Generate a list of unused ColumnClassification objects."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    columns = [draw(st_unused_column_classification()) for _ in range(size)]
    return columns


class TestSpiceSavingsEstimationConsistency:
    """Property 8: SPICE savings estimation consistency.

    **Validates: Requirements 5.3**

    For any list of unused columns with known data types, the estimated SPICE
    savings SHALL be non-negative, monotonically increasing with additional
    unused columns (adding an unused column never decreases the estimate), and
    zero when the unused column list is empty.
    """

    def _make_generator(self, average_rows: int = 1_000_000) -> ReportGenerator:
        """Create a ReportGenerator instance for testing (no S3 interaction needed)."""
        return ReportGenerator(
            s3_bucket="test-bucket",
            s3_prefix="test-prefix/",
            s3_client=None,  # Not used for estimate_spice_savings
            average_rows=average_rows,
        )

    @settings(max_examples=100)
    @given(columns=st_unused_column_list(min_size=0, max_size=20))
    def test_savings_non_negative(self, columns: list[ColumnClassification]) -> None:
        """Estimated SPICE savings SHALL be non-negative for any input."""
        generator = self._make_generator()
        savings = generator.estimate_spice_savings(columns)
        assert savings >= 0, f"Savings must be non-negative, got {savings}"

    @settings(max_examples=100)
    @given(data=st.data())
    def test_savings_monotonically_increasing(self, data: st.DataObject) -> None:
        """Adding an unused column SHALL never decrease the estimate."""
        generator = self._make_generator()

        # Generate a base list of columns
        base_columns = data.draw(st_unused_column_list(min_size=0, max_size=15))
        # Generate one additional column
        extra_column = data.draw(st_unused_column_classification())

        base_savings = generator.estimate_spice_savings(base_columns)
        extended_savings = generator.estimate_spice_savings(base_columns + [extra_column])

        assert extended_savings >= base_savings, (
            f"Savings must be monotonically increasing: "
            f"base={base_savings}, extended={extended_savings}"
        )

    @settings(max_examples=100)
    @given(average_rows=st.integers(min_value=0, max_value=100_000_000))
    def test_savings_zero_when_empty(self, average_rows: int) -> None:
        """Estimated SPICE savings SHALL be zero when the unused column list is empty."""
        generator = self._make_generator(average_rows=average_rows)
        savings = generator.estimate_spice_savings([])
        assert savings == 0, f"Savings must be zero for empty list, got {savings}"
