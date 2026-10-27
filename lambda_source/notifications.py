# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""SNS notification publishing for the SPICE Dataset Optimiser.

Publishes summary notifications after successful dataset optimisation and error
alerts when dataset processing fails. Handles missing SNS topic ARN gracefully
by skipping notifications (no-op). All SNS publish failures are logged but do
not raise exceptions to avoid disrupting the main optimisation pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambda_source.models import ExecutionMode

logger = logging.getLogger(__name__)


class NotificationPublisher:
    """Publishes optimisation notifications via Amazon SNS.

    If no SNS topic ARN is configured, all publish methods are no-ops that
    log an informational message and return immediately.
    """

    def __init__(self, sns_topic_arn: str | None, sns_client: Any | None = None):
        """Initialize NotificationPublisher.

        Args:
            sns_topic_arn: ARN of the SNS topic to publish to.
                If None, all publish operations will be skipped.
            sns_client: Optional pre-configured boto3 SNS client (for testing).
        """
        self._sns_topic_arn = sns_topic_arn
        self._sns_client = sns_client or (boto3.client("sns") if sns_topic_arn else None)

    @property
    def is_configured(self) -> bool:
        """Whether notifications are configured (topic ARN is set)."""
        return self._sns_topic_arn is not None

    def publish_summary(
        self,
        dataset_id: str,
        columns_removed: int,
        spice_savings_bytes: int,
        execution_mode: str,
    ) -> None:
        """Publish a summary notification after dataset optimisation.

        Sends an SNS message with the dataset_id, columns_removed count,
        estimated SPICE savings, and execution mode. If no topic ARN is
        configured, logs an info message and returns without publishing.

        Args:
            dataset_id: QuickSight dataset identifier that was optimised.
            columns_removed: Number of columns removed (or identified for removal).
            spice_savings_bytes: Estimated SPICE savings in bytes.
            execution_mode: 'dry_run' or 'apply'.
        """
        if not self._sns_topic_arn:
            logger.info(
                "SNS topic not configured, skipping summary notification",
                extra={"dataset_id": dataset_id},
            )
            return

        verb = "scanned" if execution_mode == ExecutionMode.DRY_RUN.value else "optimised"
        subject = f"SPICE Optimiser ({verb}): {dataset_id}"
        # SNS subject has a 100 character limit
        if len(subject) > 100:
            subject = subject[:97] + "..."

        # Format savings for readability
        savings_display = self._format_bytes(spice_savings_bytes)

        # Worded per mode. This message read "optimisation complete (dry_run). Columns
        # removed: 8." for a scan that changed nothing — the mode was in there, but every
        # other clause contradicted it, and a notification is read at a glance rather than
        # parsed.
        is_dry_run = execution_mode == ExecutionMode.DRY_RUN.value
        if is_dry_run:
            message = (
                f"Dataset '{dataset_id}' scanned — nothing was modified. "
                f"Unused columns identified: {columns_removed}. "
                f"SPICE that would be freed if removed: {savings_display}."
            )
        else:
            message = (
                f"Dataset '{dataset_id}' optimisation complete. "
                f"Columns removed: {columns_removed}. "
                f"SPICE freed: {savings_display}."
            )

        body = {
            "notification_type": "summary",
            "dataset_id": dataset_id,
            "execution_mode": execution_mode,
            (
                "columns_identified" if is_dry_run else "columns_removed"
            ): columns_removed,
            "estimated_spice_savings_bytes": spice_savings_bytes,
            "estimated_spice_savings_display": savings_display,
            "message": message,
        }

        self._publish(subject=subject, message=json.dumps(body, indent=2))

        logger.info(
            "Published summary notification",
            extra={
                "dataset_id": dataset_id,
                "columns_removed": columns_removed,
                "spice_savings_bytes": spice_savings_bytes,
                "execution_mode": execution_mode,
            },
        )

    def publish_error(self, dataset_id: str, error_message: str) -> None:
        """Publish an error alert when dataset optimisation fails.

        Sends an SNS message with the dataset_id and error details. If no
        topic ARN is configured, logs an info message and returns without
        publishing.

        Args:
            dataset_id: QuickSight dataset identifier that failed.
            error_message: Description of the error that occurred.
        """
        if not self._sns_topic_arn:
            logger.info(
                "SNS topic not configured, skipping error notification",
                extra={"dataset_id": dataset_id},
            )
            return

        subject = f"SPICE Optimiser Error: {dataset_id}"
        # SNS subject has a 100 character limit
        if len(subject) > 100:
            subject = subject[:97] + "..."

        body = {
            "notification_type": "error",
            "dataset_id": dataset_id,
            "error_message": error_message,
            "message": (
                f"Dataset '{dataset_id}' optimisation failed. "
                f"Error: {error_message}"
            ),
        }

        self._publish(subject=subject, message=json.dumps(body, indent=2))

        logger.info(
            "Published error notification",
            extra={
                "dataset_id": dataset_id,
                "error_message": error_message,
            },
        )

    def _publish(self, subject: str, message: str) -> None:
        """Publish a message to the configured SNS topic.

        Handles SNS publish failures gracefully by logging a warning
        without raising the exception.

        Args:
            subject: SNS message subject.
            message: SNS message body.
        """
        try:
            self._sns_client.publish(
                TopicArn=self._sns_topic_arn,
                Subject=subject,
                Message=message,
            )
        except ClientError as exc:
            logger.warning(
                "Failed to publish SNS notification",
                extra={
                    "topic_arn": self._sns_topic_arn,
                    "subject": subject,
                    "error_code": exc.response.get("Error", {}).get("Code", "Unknown"),
                    "error_message": str(exc),
                },
            )
        except Exception as exc:
            logger.warning(
                "Unexpected error publishing SNS notification",
                extra={
                    "topic_arn": self._sns_topic_arn,
                    "subject": subject,
                    "error": str(exc),
                },
            )

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Format byte count into a human-readable string.

        Args:
            size_bytes: Size in bytes.

        Returns:
            Human-readable size string (e.g., '12.00 GB', '500.00 MB').
        """
        if size_bytes == 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        size = float(size_bytes)
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.2f} {units[unit_index]}"
