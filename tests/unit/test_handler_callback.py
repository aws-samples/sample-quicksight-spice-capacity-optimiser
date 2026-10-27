# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the Lambda handler callback_job_id support.

Tests the _write_job_callback function and the handler's integration with it.
Verifies:
- Callback is written on successful execution
- Callback is written on failed execution
- Callback is not written when callback fields are absent
- ConditionalCheckFailedException is handled gracefully
- Other DynamoDB errors are handled gracefully
- Callback result contains expected metrics

Requirements: 3.3, 3.4, 4.3, 4.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, ANY

import pytest

from lambda_source.handler import _write_job_callback, handler


class TestWriteJobCallback:
    """Tests for the _write_job_callback helper function."""

    def test_successful_execution_writes_completed_status(self):
        """A successful result writes status='completed' to DynamoDB."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {
            "status": "success",
            "datasets_processed": 3,
            "total_columns_removed": 5,
            "total_spice_freed_bytes": 1024000,
            "dataset_results": [
                {"report_s3_key": "reports/ds-1/report.json"},
            ],
        }

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-123", "mcp-jobs-table", result)

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]
        assert call_kwargs["Key"] == {"job_id": "job-123"}
        assert call_kwargs["ExpressionAttributeValues"][":status"] == "completed"
        assert call_kwargs["ExpressionAttributeValues"][":result"]["datasets_processed"] == 3
        assert call_kwargs["ExpressionAttributeValues"][":result"]["total_columns_removed"] == 5
        assert call_kwargs["ExpressionAttributeValues"][":result"]["total_spice_freed_bytes"] == 1024000
        assert call_kwargs["ExpressionAttributeValues"][":result"]["report_s3_key"] == "reports/ds-1/report.json"
        assert call_kwargs["ConditionExpression"] == "attribute_exists(job_id)"

    def test_callback_includes_dataset_outcomes(self):
        """Per-dataset outcomes are recorded so callers can explain a result."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {
            "status": "success",
            "datasets_processed": 1,
            "total_columns_removed": 0,
            "total_spice_freed_bytes": 0,
            "dataset_results": [
                {
                    "dataset_id": "ds-1",
                    "status": "threshold_breached",
                    "columns_removed_count": 0,
                },
            ],
        }

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-out", "mcp-jobs-table", result)

        stored = mock_table.update_item.call_args[1]["ExpressionAttributeValues"][":result"]
        assert stored["dataset_outcomes"] == [
            {
                "dataset_id": "ds-1",
                "status": "threshold_breached",
                "columns_removed_count": 0,
            }
        ]

    def test_error_result_writes_failed_status(self):
        """An error result writes status='failed' to DynamoDB."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {
            "status": "error",
            "error": "Something went wrong",
            "error_type": "RuntimeError",
        }

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-456", "mcp-jobs-table", result)

        call_kwargs = mock_table.update_item.call_args[1]
        assert call_kwargs["ExpressionAttributeValues"][":status"] == "failed"
        assert call_kwargs["ExpressionAttributeValues"][":result"]["error"] == "Something went wrong"

    def test_partial_failure_writes_completed_status(self):
        """A partial_failure result is treated as completed (not error)."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {
            "status": "partial_failure",
            "datasets_processed": 2,
            "total_columns_removed": 3,
            "total_spice_freed_bytes": 500,
            "dataset_results": [],
        }

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-789", "mcp-jobs-table", result)

        call_kwargs = mock_table.update_item.call_args[1]
        assert call_kwargs["ExpressionAttributeValues"][":status"] == "completed"

    def test_conditional_check_failed_logs_warning_no_raise(self):
        """If job_id doesn't exist, log warning but don't raise."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        # Simulate ConditionalCheckFailedException
        exc = Exception("Conditional check failed")
        exc.response = {"Error": {"Code": "ConditionalCheckFailedException"}}
        mock_table.update_item.side_effect = exc

        result = {"status": "success", "datasets_processed": 1}

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            # Should not raise
            _write_job_callback("nonexistent-job", "mcp-jobs-table", result)

    def test_other_dynamodb_error_logs_warning_no_raise(self):
        """Other DynamoDB errors log warning but don't raise."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        exc = Exception("Throttled")
        exc.response = {"Error": {"Code": "ProvisionedThroughputExceededException"}}
        mock_table.update_item.side_effect = exc

        result = {"status": "success", "datasets_processed": 1}

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            # Should not raise
            _write_job_callback("job-123", "mcp-jobs-table", result)

    def test_no_dataset_results_no_report_key(self):
        """When no dataset_results, callback_result has no report_s3_key."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {
            "status": "success",
            "datasets_processed": 0,
            "total_columns_removed": 0,
            "total_spice_freed_bytes": 0,
            "dataset_results": [],
        }

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-123", "mcp-jobs-table", result)

        call_kwargs = mock_table.update_item.call_args[1]
        assert "report_s3_key" not in call_kwargs["ExpressionAttributeValues"][":result"]

    def test_updated_at_is_iso_timestamp(self):
        """The updated_at value is a valid ISO-8601 timestamp."""
        mock_table = MagicMock()
        mock_dynamodb = MagicMock()
        mock_dynamodb.Table.return_value = mock_table

        result = {"status": "success", "datasets_processed": 1}

        with patch("lambda_source.handler.boto3.resource", return_value=mock_dynamodb):
            _write_job_callback("job-123", "mcp-jobs-table", result)

        call_kwargs = mock_table.update_item.call_args[1]
        updated_at = call_kwargs["ExpressionAttributeValues"][":updated_at"]
        # Should parse without error
        parsed = datetime.fromisoformat(updated_at)
        assert parsed.tzinfo is not None  # Must be timezone-aware


class TestHandlerCallbackIntegration:
    """Tests that the handler correctly invokes _write_job_callback."""

    @patch("lambda_source.handler._write_job_callback")
    @patch("lambda_source.handler.Orchestrator")
    @patch("lambda_source.handler.load_config")
    @patch("lambda_source.handler.validate_config", return_value=[])
    def test_handler_calls_callback_on_success(
        self, mock_validate, mock_load_config, mock_orch_cls, mock_callback
    ):
        """Handler calls _write_job_callback when callback fields are present."""
        from lambda_source.models import ExecutionResult

        mock_config = MagicMock()
        mock_config.execution_mode = "dry_run"
        mock_config.target_datasets = ["ds-1"]
        mock_config.aws_account_id = "123"
        mock_config.aws_region = "us-east-1"
        mock_load_config.return_value = mock_config

        mock_result = ExecutionResult(
            execution_id="exec-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        mock_result.datasets_processed = 1
        mock_result.datasets_failed = 0
        mock_result.total_columns_removed = 2
        mock_result.total_spice_freed_bytes = 500
        mock_result.dataset_results = []

        mock_orch = MagicMock()
        mock_orch.run.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        event = {
            "callback_job_id": "job-abc",
            "callback_table": "my-table",
            "aws_account_id": "123",
            "aws_region": "us-east-1",
            "execution_mode": "dry_run",
            "target_datasets": ["ds-1"],
        }

        handler(event, None)

        mock_callback.assert_called_once_with("job-abc", "my-table", ANY)
        # Verify the result passed has success status
        call_result = mock_callback.call_args[0][2]
        assert call_result["status"] == "success"

    @patch("lambda_source.handler._write_job_callback")
    @patch("lambda_source.handler.Orchestrator")
    @patch("lambda_source.handler.load_config")
    @patch("lambda_source.handler.validate_config", return_value=[])
    def test_handler_does_not_call_callback_without_fields(
        self, mock_validate, mock_load_config, mock_orch_cls, mock_callback
    ):
        """Handler does NOT call _write_job_callback when callback fields absent."""
        from lambda_source.models import ExecutionResult

        mock_config = MagicMock()
        mock_config.execution_mode = "dry_run"
        mock_config.target_datasets = ["ds-1"]
        mock_config.aws_account_id = "123"
        mock_config.aws_region = "us-east-1"
        mock_load_config.return_value = mock_config

        mock_result = ExecutionResult(
            execution_id="exec-1",
            timestamp="2024-01-01T00:00:00Z",
        )
        mock_result.datasets_processed = 1
        mock_result.datasets_failed = 0
        mock_result.total_columns_removed = 0
        mock_result.total_spice_freed_bytes = 0
        mock_result.dataset_results = []

        mock_orch = MagicMock()
        mock_orch.run.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        event = {
            "aws_account_id": "123",
            "aws_region": "us-east-1",
            "execution_mode": "dry_run",
            "target_datasets": ["ds-1"],
        }

        handler(event, None)

        mock_callback.assert_not_called()

    @patch("lambda_source.handler._write_job_callback")
    @patch("lambda_source.handler.load_config", side_effect=RuntimeError("boom"))
    def test_handler_calls_callback_on_exception(self, mock_load_config, mock_callback):
        """Handler calls _write_job_callback with error on unhandled exception."""
        event = {
            "callback_job_id": "job-err",
            "callback_table": "my-table",
        }

        result = handler(event, None)

        assert result["status"] == "error"
        mock_callback.assert_called_once_with("job-err", "my-table", ANY)
        call_result = mock_callback.call_args[0][2]
        assert call_result["status"] == "error"
        assert "boom" in call_result["error"]

    @patch("lambda_source.handler._write_job_callback")
    @patch("lambda_source.handler.validate_config")
    @patch("lambda_source.handler.load_config")
    def test_handler_calls_callback_on_validation_error(
        self, mock_load_config, mock_validate, mock_callback
    ):
        """Handler calls _write_job_callback when config validation fails."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config
        mock_validate.return_value = ["missing required field: target_datasets"]

        event = {
            "callback_job_id": "job-val",
            "callback_table": "my-table",
        }

        result = handler(event, None)

        assert result["status"] == "error"
        mock_callback.assert_called_once_with("job-val", "my-table", ANY)
        call_result = mock_callback.call_args[0][2]
        assert call_result["status"] == "error"
