# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the snapshot module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from lambda_source.qs_client import QSClient
from lambda_source.snapshot import SnapshotError, SnapshotManager


@pytest.fixture
def mock_qs_client():
    """Create a mock QSClient."""
    client = MagicMock(spec=QSClient)
    client.account_id = "123456789012"
    return client


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    return MagicMock()


@pytest.fixture
def snapshot_manager(mock_qs_client, mock_s3_client):
    """Create a SnapshotManager with mocked dependencies."""
    return SnapshotManager(
        s3_bucket="test-bucket",
        s3_prefix="spice-optimiser/",
        qs_client=mock_qs_client,
        s3_client=mock_s3_client,
    )


class TestCaptureSnapshot:
    """Tests for SnapshotManager.capture_snapshot."""

    def test_capture_snapshot_calls_describe_dataset(self, snapshot_manager, mock_qs_client):
        """Capture snapshot should call DescribeDataSet for the given dataset ID."""
        mock_qs_client.describe_dataset.return_value = {
            "Name": "TestDataset",
            "DataSetId": "ds-123",
            "PhysicalTableMap": {},
            "ImportMode": "SPICE",
        }

        snapshot_manager.capture_snapshot("ds-123")

        mock_qs_client.describe_dataset.assert_called_once_with("ds-123")

    def test_capture_snapshot_stores_to_s3_with_correct_key_pattern(
        self, snapshot_manager, mock_qs_client, mock_s3_client
    ):
        """Capture snapshot should store the definition to S3 with the correct key pattern."""
        dataset_def = {
            "Name": "TestDataset",
            "DataSetId": "ds-123",
            "PhysicalTableMap": {"table1": {}},
            "ImportMode": "SPICE",
        }
        mock_qs_client.describe_dataset.return_value = dataset_def

        with patch("lambda_source.snapshot.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            s3_key = snapshot_manager.capture_snapshot("ds-123")

        assert s3_key == "spice-optimiser/snapshots/ds-123/2024-01-15T10:30:00Z.json"
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == s3_key
        assert call_kwargs["ContentType"] == "application/json"

        # Verify the body is valid JSON containing the dataset definition
        stored_body = json.loads(call_kwargs["Body"])
        assert stored_body["Name"] == "TestDataset"
        assert stored_body["DataSetId"] == "ds-123"

    def test_capture_snapshot_raises_snapshot_error_on_s3_failure(
        self, snapshot_manager, mock_qs_client, mock_s3_client
    ):
        """If S3 put_object fails, SnapshotError should be raised."""
        mock_qs_client.describe_dataset.return_value = {"Name": "Test", "DataSetId": "ds-fail"}
        mock_s3_client.put_object.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "S3 failure"}},
            "PutObject",
        )

        with pytest.raises(SnapshotError) as exc_info:
            snapshot_manager.capture_snapshot("ds-fail")

        assert exc_info.value.dataset_id == "ds-fail"
        assert "Failed to store snapshot" in str(exc_info.value)

    def test_capture_snapshot_returns_s3_key(self, snapshot_manager, mock_qs_client):
        """Capture snapshot should return the S3 key where the snapshot was stored."""
        mock_qs_client.describe_dataset.return_value = {"Name": "Test", "DataSetId": "ds-001"}

        result = snapshot_manager.capture_snapshot("ds-001")

        assert "snapshots/ds-001/" in result
        assert result.endswith(".json")

    def test_capture_snapshot_prefix_without_trailing_slash(self, mock_qs_client, mock_s3_client):
        """S3 key should be correctly generated even if prefix lacks trailing slash."""
        manager = SnapshotManager(
            s3_bucket="bucket",
            s3_prefix="no-slash",
            qs_client=mock_qs_client,
            s3_client=mock_s3_client,
        )
        mock_qs_client.describe_dataset.return_value = {"Name": "Test", "DataSetId": "ds-x"}

        result = manager.capture_snapshot("ds-x")

        assert result.startswith("no-slash/snapshots/ds-x/")

    def test_capture_snapshot_empty_prefix(self, mock_qs_client, mock_s3_client):
        """S3 key should work with an empty prefix."""
        manager = SnapshotManager(
            s3_bucket="bucket",
            s3_prefix="",
            qs_client=mock_qs_client,
            s3_client=mock_s3_client,
        )
        mock_qs_client.describe_dataset.return_value = {"Name": "Test", "DataSetId": "ds-x"}

        result = manager.capture_snapshot("ds-x")

        assert result.startswith("snapshots/ds-x/")


class TestRollbackDataset:
    """Tests for SnapshotManager.rollback_dataset."""

    def test_rollback_with_specific_key(self, snapshot_manager, mock_qs_client, mock_s3_client):
        """Rollback should retrieve the specified snapshot and call UpdateDataSet."""
        dataset_def = {
            "Name": "OriginalDataset",
            "PhysicalTableMap": {"t1": {"RelationalTable": {}}},
            "LogicalTableMap": {"l1": {}},
            "ImportMode": "SPICE",
        }
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(dataset_def).encode("utf-8")))
        }
        mock_qs_client.update_dataset.return_value = {"Status": 200}

        result = snapshot_manager.rollback_dataset(
            "ds-123", snapshot_key="spice-optimiser/snapshots/ds-123/2024-01-15T10:30:00Z.json"
        )

        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="spice-optimiser/snapshots/ds-123/2024-01-15T10:30:00Z.json",
        )
        mock_qs_client.update_dataset.assert_called_once()
        call_args = mock_qs_client.update_dataset.call_args
        assert call_args[0][0] == "ds-123"
        update_params = call_args[0][1]
        assert update_params["Name"] == "OriginalDataset"
        assert update_params["PhysicalTableMap"] == {"t1": {"RelationalTable": {}}}
        assert update_params["ImportMode"] == "SPICE"
        assert result == {"Status": 200}

    def test_rollback_finds_most_recent_snapshot_when_no_key_specified(
        self, snapshot_manager, mock_qs_client, mock_s3_client
    ):
        """When no snapshot_key is specified, rollback should use the most recent."""
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "spice-optimiser/snapshots/ds-123/2024-01-14T10:00:00Z.json",
                    "LastModified": datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
                },
                {
                    "Key": "spice-optimiser/snapshots/ds-123/2024-01-15T10:00:00Z.json",
                    "LastModified": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                },
                {
                    "Key": "spice-optimiser/snapshots/ds-123/2024-01-13T10:00:00Z.json",
                    "LastModified": datetime(2024, 1, 13, 10, 0, 0, tzinfo=timezone.utc),
                },
            ]
        }
        dataset_def = {"Name": "Latest", "PhysicalTableMap": {}, "ImportMode": "SPICE"}
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(dataset_def).encode("utf-8")))
        }
        mock_qs_client.update_dataset.return_value = {"Status": 200}

        snapshot_manager.rollback_dataset("ds-123")

        # Should retrieve the most recent by LastModified
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="spice-optimiser/snapshots/ds-123/2024-01-15T10:00:00Z.json",
        )

    def test_rollback_raises_error_when_no_snapshots_found(
        self, snapshot_manager, mock_s3_client
    ):
        """Rollback should raise SnapshotError when no snapshots exist for the dataset."""
        mock_s3_client.list_objects_v2.return_value = {"Contents": []}

        with pytest.raises(SnapshotError) as exc_info:
            snapshot_manager.rollback_dataset("ds-missing")

        assert exc_info.value.dataset_id == "ds-missing"
        assert "No snapshots found" in str(exc_info.value)

    def test_rollback_raises_error_on_s3_get_failure(
        self, snapshot_manager, mock_s3_client
    ):
        """Rollback should raise SnapshotError if S3 get_object fails."""
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject",
        )

        with pytest.raises(SnapshotError) as exc_info:
            snapshot_manager.rollback_dataset(
                "ds-123", snapshot_key="spice-optimiser/snapshots/ds-123/2024-01-15T10:30:00Z.json"
            )

        assert exc_info.value.dataset_id == "ds-123"
        assert "Failed to retrieve snapshot" in str(exc_info.value)

    def test_rollback_raises_error_on_invalid_json(
        self, snapshot_manager, mock_s3_client
    ):
        """Rollback should raise SnapshotError if the stored snapshot is not valid JSON."""
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"not valid json"))
        }

        with pytest.raises(SnapshotError) as exc_info:
            snapshot_manager.rollback_dataset(
                "ds-123", snapshot_key="spice-optimiser/snapshots/ds-123/2024-01-15T10:30:00Z.json"
            )

        assert exc_info.value.dataset_id == "ds-123"


class TestBuildUpdateParams:
    """Tests for SnapshotManager._build_update_params."""

    def test_extracts_required_fields(self):
        """Build params should include required fields from the definition."""
        definition = {
            "Name": "MyDataset",
            "DataSetId": "ds-123",
            "Arn": "arn:aws:...",
            "PhysicalTableMap": {"t1": {}},
            "LogicalTableMap": {"l1": {}},
            "ImportMode": "SPICE",
        }

        params = SnapshotManager._build_update_params(definition)

        assert params["Name"] == "MyDataset"
        assert params["PhysicalTableMap"] == {"t1": {}}
        assert params["LogicalTableMap"] == {"l1": {}}
        assert params["ImportMode"] == "SPICE"
        # Should NOT include read-only fields
        assert "DataSetId" not in params
        assert "Arn" not in params

    def test_includes_optional_fields_when_present(self):
        """Build params should include optional fields if they exist in the definition."""
        definition = {
            "Name": "MyDataset",
            "PhysicalTableMap": {},
            "ImportMode": "SPICE",
            "ColumnGroups": [{"GeoSpatialColumnGroup": {}}],
            "FieldFolders": {"folder1": {}},
            "RowLevelPermissionDataSet": {"Arn": "arn:..."},
            "ColumnLevelPermissionRules": [{"Principals": []}],
            "DataSetUsageConfiguration": {"DisableUseAsDirectQuerySource": False},
            "DatasetParameters": [{"StringDatasetParameter": {}}],
        }

        params = SnapshotManager._build_update_params(definition)

        assert "ColumnGroups" in params
        assert "FieldFolders" in params
        assert "RowLevelPermissionDataSet" in params
        assert "ColumnLevelPermissionRules" in params
        assert "DataSetUsageConfiguration" in params
        assert "DatasetParameters" in params

    def test_omits_optional_fields_when_absent(self):
        """Build params should not include optional fields if they are absent."""
        definition = {
            "Name": "Minimal",
            "PhysicalTableMap": {},
            "ImportMode": "SPICE",
        }

        params = SnapshotManager._build_update_params(definition)

        assert "ColumnGroups" not in params
        assert "FieldFolders" not in params
        assert "RowLevelPermissionDataSet" not in params


class TestGenerateS3Key:
    """Tests for SnapshotManager._generate_s3_key."""

    def test_key_matches_pattern(self, snapshot_manager):
        """Generated key should match {prefix}snapshots/{dataset_id}/{timestamp}.json."""
        key = snapshot_manager._generate_s3_key("ds-abc-123")

        assert key.startswith("spice-optimiser/snapshots/ds-abc-123/")
        assert key.endswith(".json")
        # Timestamp portion should be ISO-8601 format
        timestamp_part = key.split("/")[-1].replace(".json", "")
        # Should be parseable as a datetime
        datetime.strptime(timestamp_part, "%Y-%m-%dT%H:%M:%SZ")

    def test_unique_keys_for_different_datasets(self, snapshot_manager):
        """Different dataset IDs should produce different key prefixes."""
        key1 = snapshot_manager._generate_s3_key("ds-001")
        key2 = snapshot_manager._generate_s3_key("ds-002")

        assert "ds-001" in key1
        assert "ds-002" in key2
        assert key1 != key2
