# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Dataset snapshot capture and rollback for the SPICE Dataset Optimiser.

Provides functionality to snapshot dataset definitions before modification
and restore them via UpdateDataSet if rollback is needed.

Snapshots are stored in S3 with the key pattern:
    {prefix}snapshots/{dataset_id}/{ISO-8601-timestamp}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambda_source.qs_client import QSClient
from lambda_source.update_params import build_update_params

logger = logging.getLogger(__name__)


class SnapshotError(Exception):
    """Raised when snapshot storage fails, signalling the caller to abort optimization."""

    def __init__(self, message: str, dataset_id: str | None = None):
        super().__init__(message)
        self.dataset_id = dataset_id


class SnapshotManager:
    """Manages dataset snapshot capture and rollback via S3 storage."""

    def __init__(self, s3_bucket: str, s3_prefix: str, qs_client: QSClient, s3_client: Any | None = None):
        """Initialize SnapshotManager.

        Args:
            s3_bucket: S3 bucket name for storing snapshots.
            s3_prefix: S3 key prefix for all stored objects.
            qs_client: QSClient instance for QuickSight API calls.
            s3_client: Optional pre-configured boto3 S3 client (for testing).
        """
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._qs_client = qs_client
        self._s3_client = s3_client or boto3.client("s3")

    def capture_snapshot(self, dataset_id: str) -> str:
        """Capture a complete dataset snapshot to S3.

        Calls DescribeDataSet to get the full dataset definition and stores
        the complete response body as JSON in S3.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The S3 key where the snapshot was stored.

        Raises:
            SnapshotError: If snapshot storage fails (signals caller to abort).
        """
        logger.info(
            "Capturing dataset snapshot",
            extra={"dataset_id": dataset_id},
        )

        # Get the full dataset definition
        dataset_definition = self._qs_client.describe_dataset(dataset_id)

        # Generate the S3 key
        s3_key = self._generate_s3_key(dataset_id)

        # Serialize and store to S3
        try:
            snapshot_body = json.dumps(dataset_definition, default=str)
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=s3_key,
                Body=snapshot_body,
                ContentType="application/json",
            )
        except (ClientError, Exception) as exc:
            logger.error(
                "Failed to store dataset snapshot to S3",
                extra={
                    "dataset_id": dataset_id,
                    "s3_bucket": self._s3_bucket,
                    "s3_key": s3_key,
                    "error": str(exc),
                },
            )
            raise SnapshotError(
                f"Failed to store snapshot for dataset {dataset_id} to "
                f"s3://{self._s3_bucket}/{s3_key}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        logger.info(
            "Dataset snapshot stored successfully",
            extra={
                "dataset_id": dataset_id,
                "s3_bucket": self._s3_bucket,
                "s3_key": s3_key,
            },
        )

        return s3_key

    def rollback_dataset(self, dataset_id: str, snapshot_key: str | None = None) -> dict:
        """Rollback a dataset to a previous snapshot.

        Retrieves the most recent snapshot (or specified snapshot key) from S3
        and calls UpdateDataSet to restore the original definition.

        Args:
            dataset_id: QuickSight dataset identifier.
            snapshot_key: Optional specific snapshot S3 key to use.
                If None, the most recent snapshot for the dataset is used.

        Returns:
            The UpdateDataSet API response dict.

        Raises:
            SnapshotError: If no snapshot is found, retrieval fails, or the snapshot
                does not belong to ``dataset_id``.
        """
        if snapshot_key is None:
            snapshot_key = self._find_most_recent_snapshot(dataset_id)
        else:
            self._require_snapshot_belongs_to(dataset_id, snapshot_key)

        logger.info(
            "Rolling back dataset from snapshot",
            extra={
                "dataset_id": dataset_id,
                "snapshot_key": snapshot_key,
            },
        )

        # Download the snapshot from S3
        try:
            response = self._s3_client.get_object(
                Bucket=self._s3_bucket,
                Key=snapshot_key,
            )
            snapshot_body = response["Body"].read().decode("utf-8")
            dataset_definition = json.loads(snapshot_body)
        except (ClientError, json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "Failed to retrieve snapshot from S3",
                extra={
                    "dataset_id": dataset_id,
                    "snapshot_key": snapshot_key,
                    "error": str(exc),
                },
            )
            raise SnapshotError(
                f"Failed to retrieve snapshot for dataset {dataset_id} from "
                f"s3://{self._s3_bucket}/{snapshot_key}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        # The key check above is the access control; this is the belt to its braces. A
        # snapshot is the DescribeDataSet response verbatim, so it names the dataset it
        # was taken from. If an object under this dataset's prefix describes a different
        # dataset, something has gone wrong with the bucket and the restore must not
        # proceed.
        recorded_id = dataset_definition.get("DataSetId") if isinstance(dataset_definition, dict) else None
        if recorded_id is not None and recorded_id != dataset_id:
            raise SnapshotError(
                f"Snapshot s3://{self._s3_bucket}/{snapshot_key} describes dataset "
                f"{recorded_id}, not {dataset_id}; refusing to restore it.",
                dataset_id=dataset_id,
            )

        # Build the UpdateDataSet parameters from the snapshot
        update_params = self._build_update_params(dataset_definition)

        # Call UpdateDataSet to restore the definition
        result = self._qs_client.update_dataset(dataset_id, update_params)

        logger.info(
            "Dataset rollback completed successfully",
            extra={
                "dataset_id": dataset_id,
                "snapshot_key": snapshot_key,
            },
        )

        return result

    def list_snapshots(self, dataset_id: str, limit: int = 10) -> list[dict]:
        """List available snapshots for a dataset, newest first.

        Exists so a caller can see what is restorable *before* restoring anything.
        ``rollback_dataset`` silently picks the most recent snapshot, which is the right
        default but a poor thing to do unseen: an operator asking to undo needs to know
        whether the newest snapshot is from five minutes ago or five weeks ago, because
        restoring the wrong one reverts work nobody intended to revert.

        Args:
            dataset_id: QuickSight dataset identifier.
            limit: Maximum number of snapshots to return.

        Returns:
            List of dicts with ``snapshot_key``, ``captured_at`` and ``size_bytes``,
            most recent first. Empty when the dataset has never been snapshotted.

        Raises:
            SnapshotError: If listing fails.
        """
        search_prefix = self._snapshot_prefix(dataset_id)

        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self._s3_bucket,
                Prefix=search_prefix,
            )
        except ClientError as exc:
            raise SnapshotError(
                f"Failed to list snapshots for dataset {dataset_id}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        contents = response.get("Contents") or []
        ordered = sorted(
            contents, key=lambda obj: obj["LastModified"], reverse=True
        )

        return [
            {
                "snapshot_key": obj["Key"],
                "captured_at": obj["LastModified"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "size_bytes": obj.get("Size", 0),
            }
            for obj in ordered[:limit]
        ]

    def _snapshot_prefix(self, dataset_id: str) -> str:
        """Return the S3 key prefix under which this dataset's snapshots live.

        Single source of truth for ``{prefix}snapshots/{dataset_id}/``: the writer,
        the listers and the rollback check all derive from it, so they cannot drift.
        """
        prefix = self._s3_prefix
        # Ensure prefix ends with '/' if non-empty
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        return f"{prefix}snapshots/{dataset_id}/"

    def _require_snapshot_belongs_to(self, dataset_id: str, snapshot_key: str) -> None:
        """Refuse a caller-supplied snapshot key that is not one of this dataset's own.

        ``rollback_dataset`` takes the key from the tool payload, and it used to pass
        that key straight to ``get_object`` and then ``UpdateDataSet``. Anyone allowed
        to call the tool could therefore overwrite dataset A with the definition of any
        object in the bucket, including dataset B's snapshot, by naming its key. The
        tool's ``action="rollback"`` confirmation does not help: it confirms intent, not
        the target. Checking the key here, in the layer that performs the write, means
        every caller is covered regardless of how the request reached us.

        Raises:
            SnapshotError: If the key is outside ``{prefix}snapshots/{dataset_id}/``.
        """
        expected_prefix = self._snapshot_prefix(dataset_id)
        if not snapshot_key.startswith(expected_prefix) or snapshot_key == expected_prefix:
            logger.warning(
                "Refused rollback from a snapshot key outside the dataset's prefix",
                extra={
                    "dataset_id": dataset_id,
                    "snapshot_key": snapshot_key,
                    "expected_prefix": expected_prefix,
                    "operation": "rollback",
                },
            )
            raise SnapshotError(
                f"snapshot_key {snapshot_key!r} does not belong to dataset {dataset_id}: "
                f"its snapshots live under s3://{self._s3_bucket}/{expected_prefix}. "
                "Use list_snapshots to see the keys that can be restored.",
                dataset_id=dataset_id,
            )

    def _generate_s3_key(self, dataset_id: str) -> str:
        """Generate S3 key with pattern: {prefix}snapshots/{dataset_id}/{ISO-8601-timestamp}.json.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The generated S3 key string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"{self._snapshot_prefix(dataset_id)}{timestamp}.json"

    def _find_most_recent_snapshot(self, dataset_id: str) -> str:
        """Find the most recent snapshot S3 key for a dataset.

        Lists objects in the snapshots prefix for the dataset and returns
        the key with the most recent LastModified timestamp.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The S3 key of the most recent snapshot.

        Raises:
            SnapshotError: If no snapshots are found for the dataset.
        """
        search_prefix = self._snapshot_prefix(dataset_id)

        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self._s3_bucket,
                Prefix=search_prefix,
            )
        except ClientError as exc:
            raise SnapshotError(
                f"Failed to list snapshots for dataset {dataset_id}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        contents = response.get("Contents", [])
        if not contents:
            raise SnapshotError(
                f"No snapshots found for dataset {dataset_id} at "
                f"s3://{self._s3_bucket}/{search_prefix}",
                dataset_id=dataset_id,
            )

        # Sort by LastModified descending and return the most recent
        most_recent = max(contents, key=lambda obj: obj["LastModified"])
        return most_recent["Key"]

    @staticmethod
    def _build_update_params(dataset_definition: dict) -> dict:
        """Build UpdateDataSet API parameters from a snapshot definition.

        Delegates to ``lambda_source.update_params`` so this and
        ``DatasetModifier`` cannot drift again. This copy had already fallen behind:
        it omitted ``DataPrepConfiguration``, ``SemanticModelConfiguration`` and
        ``PerformanceConfiguration``. Since ``UpdateDataSet`` replaces rather than
        merges, rolling back a new-experience dataset would have restored its columns
        and destroyed its entire transform chain.

        Args:
            dataset_definition: The full dataset definition from DescribeDataSet.

        Returns:
            Dict of parameters suitable for passing to QSClient.update_dataset.
        """
        return build_update_params(dataset_definition)
