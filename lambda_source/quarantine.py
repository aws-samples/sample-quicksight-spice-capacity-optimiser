# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Quarantine period management for unused columns.

Tracks when unused columns were first detected and filters them based on a
configurable quarantine period. Only columns that have been continuously unused
for the full quarantine duration are eligible for removal.

Uses DynamoDB for persistent state between executions.
DynamoDB schema:
  - pk (Partition Key): dataset_id
  - sk (Sort Key): COL#{column_name}
  - first_detected: ISO-8601 timestamp
  - last_confirmed: ISO-8601 timestamp
  - ttl: Epoch seconds for auto-cleanup
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from lambda_source.models import QuarantineEntry

logger = logging.getLogger(__name__)

# Buffer days added to quarantine period for TTL auto-cleanup
_TTL_BUFFER_DAYS = 30


class QuarantineManager:
    """Manages quarantine period tracking for unused columns.

    When quarantine is disabled (dynamodb_table is None), all unused columns
    are immediately eligible for removal — filter_by_quarantine returns the
    full input list unchanged.
    """

    def __init__(
        self,
        dynamodb_table: str | None,
        s3_bucket: str | None = None,
        dynamodb_resource: Any | None = None,
    ):
        """Initialize QuarantineManager.

        Args:
            dynamodb_table: DynamoDB table name for quarantine state.
                None means quarantine is disabled.
            s3_bucket: S3 bucket (reserved for future state fallback).
            dynamodb_resource: Optional pre-configured boto3 DynamoDB resource
                (for testing).
        """
        self._table_name = dynamodb_table
        self._s3_bucket = s3_bucket
        self._table = None

        if dynamodb_table:
            resource = dynamodb_resource or boto3.resource("dynamodb")
            self._table = resource.Table(dynamodb_table)

    @property
    def enabled(self) -> bool:
        """Whether quarantine tracking is enabled."""
        return self._table_name is not None and self._table is not None

    def filter_by_quarantine(
        self,
        unused_columns: list[str],
        dataset_id: str,
        quarantine_days: int,
    ) -> list[str]:
        """Return only columns that have been unused for the full quarantine period.

        Queries DynamoDB for existing quarantine entries for the dataset and
        filters to only those columns whose first_detected timestamp is at
        least quarantine_days ago from now.

        If quarantine is disabled (no DynamoDB table configured), returns all
        unused columns immediately.

        Args:
            unused_columns: List of column names currently identified as unused.
            dataset_id: The dataset these columns belong to.
            quarantine_days: Number of days a column must be unused before removal.

        Returns:
            Subset of unused_columns that have passed the quarantine period.
        """
        if not self.enabled:
            logger.info(
                "Quarantine disabled, all unused columns eligible",
                extra={"dataset_id": dataset_id, "column_count": len(unused_columns)},
            )
            return list(unused_columns)

        if not unused_columns:
            return []

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=quarantine_days)

        # Query existing quarantine entries for this dataset
        entries = self._query_entries(dataset_id)

        # Build lookup of column_name -> QuarantineEntry
        entry_map: dict[str, QuarantineEntry] = {
            entry.column_name: entry for entry in entries
        }

        eligible: list[str] = []
        for col in unused_columns:
            entry = entry_map.get(col)
            if entry is None:
                # Column not yet tracked — not eligible
                logger.debug(
                    "Column not yet in quarantine",
                    extra={"dataset_id": dataset_id, "column_name": col},
                )
                continue

            first_detected_dt = datetime.fromisoformat(entry.first_detected)
            # Ensure timezone-aware comparison
            if first_detected_dt.tzinfo is None:
                first_detected_dt = first_detected_dt.replace(tzinfo=timezone.utc)

            if first_detected_dt <= cutoff:
                eligible.append(col)
                logger.debug(
                    "Column passed quarantine period",
                    extra={
                        "dataset_id": dataset_id,
                        "column_name": col,
                        "first_detected": entry.first_detected,
                        "quarantine_days": quarantine_days,
                    },
                )
            else:
                logger.debug(
                    "Column still in quarantine",
                    extra={
                        "dataset_id": dataset_id,
                        "column_name": col,
                        "first_detected": entry.first_detected,
                        "days_remaining": (first_detected_dt - cutoff).days,
                    },
                )

        logger.info(
            "Quarantine filtering complete",
            extra={
                "dataset_id": dataset_id,
                "total_unused": len(unused_columns),
                "eligible_for_removal": len(eligible),
                "still_in_quarantine": len(unused_columns) - len(eligible),
            },
        )
        return eligible

    def update_tracking(self, dataset_id: str, unused_columns: list[str]) -> None:
        """Record or update detection timestamps for unused columns.

        For each unused column:
        - If no existing entry: create with first_detected = now, last_confirmed = now
        - If existing entry: update last_confirmed = now (preserve first_detected)

        TTL is calculated as: first_detected + quarantine_days + 30 days buffer.
        Since we don't know quarantine_days here, we use a generous default of
        first_detected + 365 days for new entries and update TTL on confirmation.

        DynamoDB write failures are logged as warnings but do not raise exceptions.

        Args:
            dataset_id: The dataset these columns belong to.
            unused_columns: List of column names currently identified as unused.
        """
        if not self.enabled:
            return

        if not unused_columns:
            return

        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        for col in unused_columns:
            try:
                self._put_or_update_entry(dataset_id, col, now_iso, now)
            except (ClientError, Exception) as exc:
                logger.warning(
                    "Failed to update quarantine tracking",
                    extra={
                        "dataset_id": dataset_id,
                        "column_name": col,
                        "error": str(exc),
                    },
                )

    def remove_from_tracking(self, dataset_id: str, columns: list[str]) -> None:
        """Remove columns that are now used from quarantine tracking.

        Deletes DynamoDB entries for columns that have become referenced again.
        Write failures are logged as warnings but do not raise exceptions.

        Args:
            dataset_id: The dataset these columns belong to.
            columns: List of column names that are now used/referenced.
        """
        if not self.enabled:
            return

        if not columns:
            return

        for col in columns:
            try:
                self._delete_entry(dataset_id, col)
            except (ClientError, Exception) as exc:
                logger.warning(
                    "Failed to remove column from quarantine tracking",
                    extra={
                        "dataset_id": dataset_id,
                        "column_name": col,
                        "error": str(exc),
                    },
                )

        logger.info(
            "Removed columns from quarantine tracking",
            extra={
                "dataset_id": dataset_id,
                "columns_removed": len(columns),
            },
        )

    def _query_entries(self, dataset_id: str) -> list[QuarantineEntry]:
        """Query all quarantine entries for a dataset from DynamoDB.

        Args:
            dataset_id: The dataset to query entries for.

        Returns:
            List of QuarantineEntry objects for the dataset.
        """
        entries: list[QuarantineEntry] = []

        try:
            response = self._table.query(
                KeyConditionExpression=Key("pk").eq(dataset_id)
                & Key("sk").begins_with("COL#"),
            )

            for item in response.get("Items", []):
                sk = item.get("sk", "")
                column_name = sk.removeprefix("COL#")
                entries.append(
                    QuarantineEntry(
                        dataset_id=dataset_id,
                        column_name=column_name,
                        first_detected=item.get("first_detected", ""),
                        last_confirmed=item.get("last_confirmed", ""),
                    )
                )

            # Handle pagination
            while "LastEvaluatedKey" in response:
                response = self._table.query(
                    KeyConditionExpression=Key("pk").eq(dataset_id)
                    & Key("sk").begins_with("COL#"),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for item in response.get("Items", []):
                    sk = item.get("sk", "")
                    column_name = sk.removeprefix("COL#")
                    entries.append(
                        QuarantineEntry(
                            dataset_id=dataset_id,
                            column_name=column_name,
                            first_detected=item.get("first_detected", ""),
                            last_confirmed=item.get("last_confirmed", ""),
                        )
                    )

        except (ClientError, Exception) as exc:
            logger.warning(
                "Failed to query quarantine entries",
                extra={
                    "dataset_id": dataset_id,
                    "error": str(exc),
                },
            )

        return entries

    def _put_or_update_entry(
        self,
        dataset_id: str,
        column_name: str,
        now_iso: str,
        now: datetime,
    ) -> None:
        """Create or update a quarantine entry in DynamoDB.

        Uses a conditional update:
        - If the item doesn't exist, creates it with first_detected = now
        - If the item exists, only updates last_confirmed (preserves first_detected)

        TTL is calculated as: now + quarantine buffer (365 days for new entries,
        updated on each confirmation to extend cleanup window).

        Args:
            dataset_id: Dataset identifier (partition key).
            column_name: Column name (used in sort key as COL#{column_name}).
            now_iso: Current time as ISO-8601 string.
            now: Current time as datetime object.
        """
        sk = f"COL#{column_name}"
        # TTL: generous default of 365 days from now for auto-cleanup
        ttl_epoch = int((now + timedelta(days=365)).timestamp())

        # Use update_item with conditional expression to preserve first_detected
        self._table.update_item(
            Key={"pk": dataset_id, "sk": sk},
            UpdateExpression=(
                "SET last_confirmed = :now, "
                "first_detected = if_not_exists(first_detected, :now), "
                "ttl = :ttl"
            ),
            ExpressionAttributeValues={
                ":now": now_iso,
                ":ttl": ttl_epoch,
            },
        )

    def _delete_entry(self, dataset_id: str, column_name: str) -> None:
        """Delete a quarantine entry from DynamoDB.

        Args:
            dataset_id: Dataset identifier (partition key).
            column_name: Column name (used in sort key as COL#{column_name}).
        """
        sk = f"COL#{column_name}"
        self._table.delete_item(Key={"pk": dataset_id, "sk": sk})
