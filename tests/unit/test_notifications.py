# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the notifications module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from lambda_source.notifications import NotificationPublisher


class TestNotificationPublisherInit:
    """Tests for NotificationPublisher initialization."""

    def test_init_with_topic_arn(self):
        """Publisher with topic ARN is marked as configured."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )
        assert publisher.is_configured is True

    def test_init_without_topic_arn(self):
        """Publisher without topic ARN is marked as not configured."""
        publisher = NotificationPublisher(sns_topic_arn=None)
        assert publisher.is_configured is False

    def test_init_with_none_does_not_create_client(self):
        """When topic ARN is None, no boto3 client is created."""
        publisher = NotificationPublisher(sns_topic_arn=None)
        assert publisher._sns_client is None


class TestPublishSummary:
    """Tests for the publish_summary method."""

    def test_publish_summary_sends_correct_message(self):
        """Summary notification contains dataset_id, columns_removed, and savings."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        publisher.publish_summary(
            dataset_id="dataset-123",
            columns_removed=5,
            spice_savings_bytes=1073741824,  # 1 GB
            execution_mode="apply",
        )

        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["TopicArn"] == "arn:aws:sns:us-east-1:123456789012:my-topic"
        # Subject states the mode, so a mailbox of these is skimmable without opening them.
        assert call_kwargs["Subject"] == "SPICE Optimiser (optimised): dataset-123"

        body = json.loads(call_kwargs["Message"])
        assert body["notification_type"] == "summary"
        assert body["dataset_id"] == "dataset-123"
        assert body["columns_removed"] == 5
        assert body["estimated_spice_savings_bytes"] == 1073741824
        assert body["execution_mode"] == "apply"
        # Past tense is correct here, and only here: this run did remove them.
        assert "optimisation complete" in body["message"]
        assert "Columns removed: 5" in body["message"]

    def test_publish_summary_dry_run_mode(self):
        """Summary notification includes execution_mode for dry_run."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        publisher.publish_summary(
            dataset_id="ds-456",
            columns_removed=10,
            spice_savings_bytes=0,
            execution_mode="dry_run",
        )

        call_kwargs = mock_client.publish.call_args[1]
        body = json.loads(call_kwargs["Message"])
        assert body["execution_mode"] == "dry_run"

        # columns_identified, not columns_removed. A scan modifies nothing, and this
        # notification previously said "optimisation complete (dry_run). Columns removed:
        # 10." The mode was in there, but every other clause contradicted it - and an
        # SNS summary is read at a glance, not parsed.
        assert body["columns_identified"] == 10
        assert "columns_removed" not in body
        assert "nothing was modified" in body["message"]
        assert "would be freed" in body["message"]
        assert call_kwargs["Subject"] == "SPICE Optimiser (scanned): ds-456"

    def test_publish_summary_skips_when_no_topic(self):
        """Summary notification is a no-op when topic ARN is None."""
        publisher = NotificationPublisher(sns_topic_arn=None)

        # Should not raise
        publisher.publish_summary(
            dataset_id="dataset-123",
            columns_removed=5,
            spice_savings_bytes=1000,
            execution_mode="apply",
        )

    def test_publish_summary_truncates_long_subject(self):
        """Subject is truncated to 100 characters when dataset_id is long."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        long_dataset_id = "a" * 200
        publisher.publish_summary(
            dataset_id=long_dataset_id,
            columns_removed=1,
            spice_savings_bytes=100,
            execution_mode="dry_run",
        )

        call_kwargs = mock_client.publish.call_args[1]
        assert len(call_kwargs["Subject"]) <= 100

    def test_publish_summary_handles_sns_failure_gracefully(self):
        """SNS publish failure is logged but does not raise."""
        mock_client = MagicMock()
        mock_client.publish.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Service unavailable"}},
            "Publish",
        )
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        # Should not raise
        publisher.publish_summary(
            dataset_id="dataset-123",
            columns_removed=5,
            spice_savings_bytes=1000,
            execution_mode="apply",
        )


class TestPublishError:
    """Tests for the publish_error method."""

    def test_publish_error_sends_correct_message(self):
        """Error notification contains dataset_id and error_message."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        publisher.publish_error(
            dataset_id="dataset-789",
            error_message="UpdateDataSet failed: Access denied",
        )

        mock_client.publish.assert_called_once()
        call_kwargs = mock_client.publish.call_args[1]
        assert call_kwargs["TopicArn"] == "arn:aws:sns:us-east-1:123456789012:my-topic"
        assert call_kwargs["Subject"] == "SPICE Optimiser Error: dataset-789"

        body = json.loads(call_kwargs["Message"])
        assert body["notification_type"] == "error"
        assert body["dataset_id"] == "dataset-789"
        assert body["error_message"] == "UpdateDataSet failed: Access denied"

    def test_publish_error_skips_when_no_topic(self):
        """Error notification is a no-op when topic ARN is None."""
        publisher = NotificationPublisher(sns_topic_arn=None)

        # Should not raise
        publisher.publish_error(
            dataset_id="dataset-789",
            error_message="Some error",
        )

    def test_publish_error_handles_sns_failure_gracefully(self):
        """SNS publish failure for error alert does not raise."""
        mock_client = MagicMock()
        mock_client.publish.side_effect = ClientError(
            {"Error": {"Code": "InvalidParameter", "Message": "Invalid ARN"}},
            "Publish",
        )
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        # Should not raise
        publisher.publish_error(
            dataset_id="dataset-789",
            error_message="Something failed",
        )

    def test_publish_error_truncates_long_subject(self):
        """Subject is truncated when dataset_id is long."""
        mock_client = MagicMock()
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        long_dataset_id = "b" * 200
        publisher.publish_error(
            dataset_id=long_dataset_id,
            error_message="Error occurred",
        )

        call_kwargs = mock_client.publish.call_args[1]
        assert len(call_kwargs["Subject"]) <= 100


class TestPublishHandlesUnexpectedExceptions:
    """Tests for handling unexpected exceptions from SNS."""

    def test_publish_summary_handles_generic_exception(self):
        """Generic exceptions during publish are caught gracefully."""
        mock_client = MagicMock()
        mock_client.publish.side_effect = RuntimeError("Unexpected network issue")
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        # Should not raise
        publisher.publish_summary(
            dataset_id="dataset-123",
            columns_removed=1,
            spice_savings_bytes=100,
            execution_mode="dry_run",
        )

    def test_publish_error_handles_generic_exception(self):
        """Generic exceptions during error publish are caught gracefully."""
        mock_client = MagicMock()
        mock_client.publish.side_effect = RuntimeError("Connection reset")
        publisher = NotificationPublisher(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sns_client=mock_client,
        )

        # Should not raise
        publisher.publish_error(
            dataset_id="dataset-123",
            error_message="Something went wrong",
        )


class TestFormatBytes:
    """Tests for the _format_bytes static method."""

    def test_format_zero_bytes(self):
        assert NotificationPublisher._format_bytes(0) == "0 B"

    def test_format_bytes_value(self):
        assert NotificationPublisher._format_bytes(512) == "512.00 B"

    def test_format_kilobytes(self):
        assert NotificationPublisher._format_bytes(1024) == "1.00 KB"

    def test_format_megabytes(self):
        assert NotificationPublisher._format_bytes(1048576) == "1.00 MB"

    def test_format_gigabytes(self):
        assert NotificationPublisher._format_bytes(1073741824) == "1.00 GB"

    def test_format_terabytes(self):
        assert NotificationPublisher._format_bytes(1099511627776) == "1.00 TB"
