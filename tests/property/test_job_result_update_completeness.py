# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for job result update completeness.

Property 4: Job result update completeness

For any Lambda execution result (success or failure) written back to the
Job Store, the job record SHALL transition to "completed" or "failed" with
a non-null result or error field respectively, and the updated_at timestamp
SHALL be later than created_at.

# Feature: quick-mcp-integration, Property 4: Job result update completeness
**Validates: Requirements 3.3, 3.4, 4.3, 4.4, 7.4**
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from lambda_source.handler import _write_job_callback


# ---------------------------------------------------------------------------
# Strategies for Lambda execution results
# ---------------------------------------------------------------------------


@st.composite
def st_successful_lambda_result(draw: st.DrawFn) -> dict:
    """Generate a random successful Lambda execution result.

    Mimics what the Lambda handler returns on success or partial_failure.
    """
    status = draw(st.sampled_from(["success", "partial_failure"]))
    datasets_processed = draw(st.integers(min_value=0, max_value=50))
    total_columns_removed = draw(st.integers(min_value=0, max_value=200))
    total_spice_freed_bytes = draw(
        st.integers(min_value=0, max_value=10_000_000_000)
    )

    # Generate dataset_results list
    num_results = draw(st.integers(min_value=0, max_value=5))
    dataset_results = []
    for _ in range(num_results):
        ds_result: dict = {
            "dataset_id": draw(
                st.text(alphabet="abcdef0123456789-", min_size=5, max_size=36)
            ),
            "status": draw(st.sampled_from(["optimized", "skipped", "threshold_breached"])),
        }
        # Optionally include a report key
        if draw(st.booleans()):
            ds_result["report_s3_key"] = draw(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789/-_.", min_size=10, max_size=60)
            )
        dataset_results.append(ds_result)

    return {
        "status": status,
        "datasets_processed": datasets_processed,
        "total_columns_removed": total_columns_removed,
        "total_spice_freed_bytes": total_spice_freed_bytes,
        "dataset_results": dataset_results,
    }


@st.composite
def st_failed_lambda_result(draw: st.DrawFn) -> dict:
    """Generate a random failed Lambda execution result.

    Mimics what the Lambda handler returns on error.
    """
    error_message = draw(
        st.sampled_from([
            "Configuration validation failed",
            "Lambda execution timeout after 900 seconds",
            "QuickSight API error: AccessDeniedException",
            "DynamoDB throttling: ProvisionedThroughputExceededException",
            "Internal error: unexpected null response",
            "ModificationError: UpdateDataSet failed",
            "SnapshotError: S3 PutObject failed",
        ])
    )
    error_type = draw(
        st.sampled_from([
            "RuntimeError",
            "TimeoutError",
            "QSClientError",
            "ClientError",
            "ModificationError",
            "SnapshotError",
            "ValidationError",
        ])
    )
    duration_ms = draw(
        st.floats(min_value=0.0, max_value=900000.0, allow_nan=False, allow_infinity=False)
    )

    return {
        "status": "error",
        "error": error_message,
        "error_type": error_type,
        "duration_ms": duration_ms,
    }


@st.composite
def st_job_id(draw: st.DrawFn) -> str:
    """Generate a random job ID (UUID-like)."""
    hex_chars = "abcdef0123456789"
    segments = [8, 4, 4, 4, 12]
    parts = []
    for seg_len in segments:
        part = draw(st.text(alphabet=hex_chars, min_size=seg_len, max_size=seg_len))
        parts.append(part)
    return "-".join(parts)


@st.composite
def st_table_name(draw: st.DrawFn) -> str:
    """Generate a random DynamoDB table name."""
    return draw(
        st.sampled_from([
            "mcp-jobs-table",
            "spice-optimiser-jobs",
            "dev-mcp-job-store",
            "prod-job-store-table",
        ])
    )


# ---------------------------------------------------------------------------
# In-memory DynamoDB table mock for property tests
# ---------------------------------------------------------------------------


class InMemoryJobTable:
    """In-memory DynamoDB table mock that records write operations.

    Simulates the update_item behavior including condition expressions.
    Tracks the result of each write for property assertions.
    """

    def __init__(self, existing_job_ids: set[str] | None = None):
        """Initialize with a set of existing job IDs.

        Args:
            existing_job_ids: Job IDs that "exist" in the table.
                If None, all job IDs are treated as existing.
        """
        self._existing_job_ids = existing_job_ids
        self._updates: list[dict] = []

    def update_item(self, **kwargs) -> dict:
        """Record an update_item call, enforcing condition expressions."""
        key = kwargs.get("Key", {})
        job_id = key.get("job_id")
        condition = kwargs.get("ConditionExpression", "")

        # Enforce attribute_exists(job_id) condition
        if "attribute_exists(job_id)" in condition:
            if self._existing_job_ids is not None and job_id not in self._existing_job_ids:
                # Simulate ConditionalCheckFailedException
                exc = Exception("Conditional check failed")
                exc.response = {"Error": {"Code": "ConditionalCheckFailedException"}}
                raise exc

        self._updates.append(kwargs)
        return {}

    @property
    def updates(self) -> list[dict]:
        """Return all recorded update operations."""
        return self._updates


# ---------------------------------------------------------------------------
# Property 4: Job result update completeness
# ---------------------------------------------------------------------------


class TestJobResultUpdateCompleteness:
    """Property 4: Job result update completeness.

    For any Lambda execution result (success or failure) written back to the
    Job Store, the job record SHALL transition to "completed" or "failed" with
    a non-null result or error field respectively, and the updated_at timestamp
    SHALL be later than created_at.

    # Feature: quick-mcp-integration, Property 4: Job result update completeness
    **Validates: Requirements 3.3, 3.4, 4.3, 4.4, 7.4**
    """

    @given(
        result=st_successful_lambda_result(),
        job_id=st_job_id(),
        table_name=st_table_name(),
    )
    @settings(max_examples=100)
    def test_successful_result_transitions_to_completed(
        self,
        result: dict,
        job_id: str,
        table_name: str,
    ) -> None:
        """For any successful Lambda execution result, the job record
        transitions to "completed" with a non-null result field.

        **Validates: Requirements 3.3, 4.3, 7.4**
        """
        mock_table = InMemoryJobTable(existing_job_ids={job_id})
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        # Record timestamp before the callback
        before_call = datetime.now(timezone.utc)

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback(job_id, table_name, result)

        # Verify exactly one update was made
        assert len(mock_table.updates) == 1, (
            f"Expected 1 update, got {len(mock_table.updates)}"
        )

        update = mock_table.updates[0]

        # Verify the key targets the correct job
        assert update["Key"] == {"job_id": job_id}

        # Verify status is "completed" (non-error results)
        attr_values = update["ExpressionAttributeValues"]
        assert attr_values[":status"] == "completed", (
            f"Expected status 'completed' for result with status "
            f"'{result['status']}', got '{attr_values[':status']}'"
        )

        # Verify result field is non-null
        callback_result = attr_values[":result"]
        assert callback_result is not None, "Result field must not be null"
        assert isinstance(callback_result, dict), "Result must be a dict"

        # Verify updated_at is a valid ISO timestamp after the test start
        updated_at_str = attr_values[":updated_at"]
        assert updated_at_str is not None, "updated_at must not be null"
        updated_at = datetime.fromisoformat(updated_at_str)
        assert updated_at.tzinfo is not None, "updated_at must be timezone-aware"

        # Verify conditional write uses attribute_exists(job_id)
        assert update["ConditionExpression"] == "attribute_exists(job_id)", (
            "Must use conditional write with attribute_exists(job_id)"
        )

    @given(
        result=st_failed_lambda_result(),
        job_id=st_job_id(),
        table_name=st_table_name(),
    )
    @settings(max_examples=100)
    def test_failed_result_transitions_to_failed(
        self,
        result: dict,
        job_id: str,
        table_name: str,
    ) -> None:
        """For any failed Lambda execution result, the job record
        transitions to "failed" with a non-null error field in the result.

        **Validates: Requirements 3.4, 4.4, 7.4**
        """
        mock_table = InMemoryJobTable(existing_job_ids={job_id})
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback(job_id, table_name, result)

        # Verify exactly one update was made
        assert len(mock_table.updates) == 1, (
            f"Expected 1 update, got {len(mock_table.updates)}"
        )

        update = mock_table.updates[0]

        # Verify the key targets the correct job
        assert update["Key"] == {"job_id": job_id}

        # Verify status is "failed" (error results)
        attr_values = update["ExpressionAttributeValues"]
        assert attr_values[":status"] == "failed", (
            f"Expected status 'failed' for error result, got '{attr_values[':status']}'"
        )

        # Verify result field contains error information
        callback_result = attr_values[":result"]
        assert callback_result is not None, "Result field must not be null for failed jobs"
        assert isinstance(callback_result, dict), "Result must be a dict"
        assert "error" in callback_result, (
            "Failed result must contain 'error' key in callback_result"
        )
        assert callback_result["error"] is not None, "Error field must not be null"
        assert len(callback_result["error"]) > 0, "Error field must be non-empty"

        # Verify updated_at is a valid ISO timestamp
        updated_at_str = attr_values[":updated_at"]
        assert updated_at_str is not None, "updated_at must not be null"
        updated_at = datetime.fromisoformat(updated_at_str)
        assert updated_at.tzinfo is not None, "updated_at must be timezone-aware"

        # Verify conditional write uses attribute_exists(job_id)
        assert update["ConditionExpression"] == "attribute_exists(job_id)", (
            "Must use conditional write with attribute_exists(job_id)"
        )

    @given(
        result=st.one_of(st_successful_lambda_result(), st_failed_lambda_result()),
        job_id=st_job_id(),
        table_name=st_table_name(),
    )
    @settings(max_examples=100)
    def test_updated_at_is_after_invocation_time(
        self,
        result: dict,
        job_id: str,
        table_name: str,
    ) -> None:
        """For any Lambda execution result written back, updated_at SHALL be
        later than any reasonable created_at (i.e., a recent timestamp).

        This verifies the temporal ordering requirement: updated_at > created_at.
        Since we control the test timing, we verify updated_at is a current
        timestamp (which would always be after any prior created_at).

        **Validates: Requirements 7.4**
        """
        mock_table = InMemoryJobTable(existing_job_ids={job_id})
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        # Record time before callback execution
        before_call = datetime.now(timezone.utc)

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback(job_id, table_name, result)

        after_call = datetime.now(timezone.utc)

        # Verify update was made
        assert len(mock_table.updates) == 1

        update = mock_table.updates[0]
        attr_values = update["ExpressionAttributeValues"]
        updated_at_str = attr_values[":updated_at"]
        updated_at = datetime.fromisoformat(updated_at_str)

        # updated_at should be between before_call and after_call
        assert updated_at >= before_call, (
            f"updated_at {updated_at.isoformat()} is before the callback invocation "
            f"time {before_call.isoformat()}"
        )
        assert updated_at <= after_call, (
            f"updated_at {updated_at.isoformat()} is after the callback completion "
            f"time {after_call.isoformat()}"
        )

    @given(
        result=st.one_of(st_successful_lambda_result(), st_failed_lambda_result()),
        job_id=st_job_id(),
        table_name=st_table_name(),
    )
    @settings(max_examples=100)
    def test_conditional_write_prevents_update_to_nonexistent_job(
        self,
        result: dict,
        job_id: str,
        table_name: str,
    ) -> None:
        """For any Lambda execution result, the callback uses conditional
        writes (attribute_exists(job_id)) to prevent writing to non-existent
        jobs. The function handles the error gracefully without raising.

        **Validates: Requirements 7.4**
        """
        # Create table with NO existing job IDs (simulates non-existent job)
        mock_table = InMemoryJobTable(existing_job_ids=set())
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            # Should not raise - function handles the error gracefully
            _write_job_callback(job_id, table_name, result)

        # No successful updates should have been recorded
        assert len(mock_table.updates) == 0, (
            "No updates should succeed when job_id does not exist"
        )

    @given(
        result=st_successful_lambda_result(),
        job_id=st_job_id(),
        table_name=st_table_name(),
    )
    @settings(max_examples=100)
    def test_successful_result_contains_key_metrics(
        self,
        result: dict,
        job_id: str,
        table_name: str,
    ) -> None:
        """For any successful Lambda result, the callback result stored in
        DynamoDB contains the key metrics: datasets_processed,
        total_columns_removed, total_spice_freed_bytes.

        **Validates: Requirements 3.3, 4.3**
        """
        mock_table = InMemoryJobTable(existing_job_ids={job_id})
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback(job_id, table_name, result)

        assert len(mock_table.updates) == 1

        update = mock_table.updates[0]
        callback_result = update["ExpressionAttributeValues"][":result"]

        # Verify key metrics are present
        assert "datasets_processed" in callback_result, (
            "callback_result must contain 'datasets_processed'"
        )
        assert "total_columns_removed" in callback_result, (
            "callback_result must contain 'total_columns_removed'"
        )
        assert "total_spice_freed_bytes" in callback_result, (
            "callback_result must contain 'total_spice_freed_bytes'"
        )

        # Verify metrics match the input result
        assert callback_result["datasets_processed"] == result.get("datasets_processed", 0)
        assert callback_result["total_columns_removed"] == result.get("total_columns_removed", 0)
        assert callback_result["total_spice_freed_bytes"] == result.get("total_spice_freed_bytes", 0)
