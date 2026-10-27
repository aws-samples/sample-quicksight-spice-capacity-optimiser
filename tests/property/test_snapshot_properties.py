# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for snapshot module.

Feature: spice-dataset-optimiser
Property 9: S3 key pattern correctness
Property 10: Snapshot and rollback round-trip

Validates: Requirements 6.2, 6.4, 10.4
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from unittest.mock import MagicMock, create_autospec

from lambda_source.qs_client import QSClient
from lambda_source.snapshot import SnapshotManager
from tests.property.strategies import st_dataset_id, st_s3_bucket_name, st_dataset_definition


# Strategy for generating valid S3 prefixes (including empty prefix)
st_s3_prefix = st.sampled_from([
    "spice-optimiser/",
    "reports/",
    "data/",
    "",
    "my-prefix",
    "nested/path/prefix/",
])


# ---------------------------------------------------------------------------
# Property 9: S3 key pattern correctness
# ---------------------------------------------------------------------------


@given(
    dataset_id=st_dataset_id(),
    s3_prefix=st_s3_prefix,
    s3_bucket=st_s3_bucket_name(),
)
@settings(max_examples=100)
def test_s3_key_matches_snapshot_pattern(
    dataset_id: str,
    s3_prefix: str,
    s3_bucket: str,
) -> None:
    """Property 9: S3 key pattern correctness.

    **Validates: Requirements 6.2, 10.4**

    For any valid dataset identifier and valid ISO-8601 timestamp, the
    generated S3 keys for snapshots SHALL match the configured pattern
    {prefix}snapshots/{dataset_id}/{ISO-timestamp}.json.
    """
    # Create a SnapshotManager with the generated inputs
    qs_client = create_autospec(QSClient, instance=True)
    s3_client = MagicMock()
    manager = SnapshotManager(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        qs_client=qs_client,
        s3_client=s3_client,
    )

    # Generate the S3 key
    key = manager._generate_s3_key(dataset_id)

    # Normalize prefix: should end with '/' if non-empty
    expected_prefix = s3_prefix
    if expected_prefix and not expected_prefix.endswith("/"):
        expected_prefix = expected_prefix + "/"

    # Assert the key matches the pattern: {prefix}snapshots/{dataset_id}/{timestamp}.json
    pattern = re.escape(expected_prefix) + r"snapshots/" + re.escape(dataset_id) + r"/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\.json"
    assert re.fullmatch(pattern, key), (
        f"Key '{key}' does not match expected pattern: "
        f"'{expected_prefix}snapshots/{dataset_id}/{{ISO-timestamp}}.json'"
    )


@given(
    dataset_id=st_dataset_id(),
    s3_prefix=st_s3_prefix,
    s3_bucket=st_s3_bucket_name(),
)
@settings(max_examples=100)
def test_s3_key_contains_dataset_id(
    dataset_id: str,
    s3_prefix: str,
    s3_bucket: str,
) -> None:
    """Property 9: S3 key pattern correctness - key contains dataset_id.

    **Validates: Requirements 6.2, 10.4**

    The generated S3 key SHALL contain the dataset_id.
    """
    qs_client = create_autospec(QSClient, instance=True)
    s3_client = MagicMock()
    manager = SnapshotManager(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        qs_client=qs_client,
        s3_client=s3_client,
    )

    key = manager._generate_s3_key(dataset_id)

    assert dataset_id in key, f"Key '{key}' does not contain dataset_id '{dataset_id}'"


@given(
    dataset_id=st_dataset_id(),
    s3_prefix=st_s3_prefix,
    s3_bucket=st_s3_bucket_name(),
)
@settings(max_examples=100)
def test_s3_key_ends_with_json_extension(
    dataset_id: str,
    s3_prefix: str,
    s3_bucket: str,
) -> None:
    """Property 9: S3 key pattern correctness - key ends with .json.

    **Validates: Requirements 6.2, 10.4**

    The generated S3 key SHALL end with .json.
    """
    qs_client = create_autospec(QSClient, instance=True)
    s3_client = MagicMock()
    manager = SnapshotManager(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        qs_client=qs_client,
        s3_client=s3_client,
    )

    key = manager._generate_s3_key(dataset_id)

    assert key.endswith(".json"), f"Key '{key}' does not end with '.json'"


@given(
    dataset_id_1=st_dataset_id(),
    dataset_id_2=st_dataset_id(),
    s3_prefix=st_s3_prefix,
    s3_bucket=st_s3_bucket_name(),
)
@settings(max_examples=100)
def test_different_dataset_ids_produce_different_keys(
    dataset_id_1: str,
    dataset_id_2: str,
    s3_prefix: str,
    s3_bucket: str,
) -> None:
    """Property 9: S3 key pattern correctness - uniqueness for distinct dataset_ids.

    **Validates: Requirements 6.2, 10.4**

    Two different dataset_ids SHALL produce different S3 keys.
    """
    # Only test when dataset IDs are actually different
    if dataset_id_1 == dataset_id_2:
        return

    qs_client = create_autospec(QSClient, instance=True)
    s3_client = MagicMock()
    manager = SnapshotManager(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        qs_client=qs_client,
        s3_client=s3_client,
    )

    key_1 = manager._generate_s3_key(dataset_id_1)
    key_2 = manager._generate_s3_key(dataset_id_2)

    assert key_1 != key_2, (
        f"Different dataset_ids produced the same key: "
        f"'{dataset_id_1}' and '{dataset_id_2}' both produced '{key_1}'"
    )


@given(
    dataset_id=st_dataset_id(),
    s3_prefix=st_s3_prefix,
    s3_bucket=st_s3_bucket_name(),
)
@settings(max_examples=100)
def test_s3_key_timestamp_is_valid_iso8601(
    dataset_id: str,
    s3_prefix: str,
    s3_bucket: str,
) -> None:
    """Property 9: S3 key pattern correctness - valid ISO-8601 timestamp.

    **Validates: Requirements 6.2, 10.4**

    The timestamp portion of the generated S3 key SHALL be valid ISO-8601.
    """
    qs_client = create_autospec(QSClient, instance=True)
    s3_client = MagicMock()
    manager = SnapshotManager(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        qs_client=qs_client,
        s3_client=s3_client,
    )

    key = manager._generate_s3_key(dataset_id)

    # Extract timestamp from key: it's the filename without .json
    # Key pattern: {prefix}snapshots/{dataset_id}/{timestamp}.json
    filename = key.rsplit("/", 1)[-1]  # Get last path segment
    timestamp_str = filename.replace(".json", "")

    # Validate it's a parseable ISO-8601 timestamp
    try:
        parsed = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise AssertionError(
            f"Timestamp '{timestamp_str}' extracted from key '{key}' "
            f"is not a valid ISO-8601 timestamp (expected format: YYYY-MM-DDTHH:MM:SSZ)"
        )

    # Verify parsed timestamp components are reasonable
    assert 2000 <= parsed.year <= 2100, f"Year {parsed.year} is out of reasonable range"
    assert 1 <= parsed.month <= 12, f"Month {parsed.month} is invalid"
    assert 1 <= parsed.day <= 31, f"Day {parsed.day} is invalid"
    assert 0 <= parsed.hour <= 23, f"Hour {parsed.hour} is invalid"
    assert 0 <= parsed.minute <= 59, f"Minute {parsed.minute} is invalid"
    assert 0 <= parsed.second <= 59, f"Second {parsed.second} is invalid"


# ---------------------------------------------------------------------------
# Property 10: Snapshot and rollback round-trip
# ---------------------------------------------------------------------------


@given(dataset_def=st_dataset_definition())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_snapshot_rollback_round_trip(dataset_def: dict) -> None:
    """Property 10: Snapshot and rollback round-trip.

    **Validates: Requirements 6.4**

    For any valid dataset definition, capturing a snapshot and then performing
    a rollback SHALL produce a definition passed to UpdateDataSet that is
    logically equivalent to the original definition captured.
    """
    dataset_id = dataset_def["DataSetId"]

    # --- Set up mocks ---
    mock_qs_client = MagicMock()
    mock_s3_client = MagicMock()

    # Mock describe_dataset to return our generated definition
    mock_qs_client.describe_dataset.return_value = dataset_def

    # Track what gets stored in S3
    stored_body: dict = {}

    def mock_put_object(**kwargs):
        stored_body["key"] = kwargs["Key"]
        stored_body["body"] = kwargs["Body"]
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    mock_s3_client.put_object.side_effect = mock_put_object

    # Mock get_object to return whatever was stored
    def mock_get_object(**kwargs):
        body_bytes = stored_body["body"].encode("utf-8")
        body_stream = MagicMock()
        body_stream.read.return_value = body_bytes
        return {"Body": body_stream}

    mock_s3_client.get_object.side_effect = mock_get_object

    # Track what gets passed to update_dataset
    update_call_args: dict = {}

    def mock_update_dataset(ds_id, params):
        update_call_args["dataset_id"] = ds_id
        update_call_args["params"] = params
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    mock_qs_client.update_dataset.side_effect = mock_update_dataset

    # --- Execute capture + rollback ---
    manager = SnapshotManager(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix",
        qs_client=mock_qs_client,
        s3_client=mock_s3_client,
    )

    # Capture snapshot
    snapshot_key = manager.capture_snapshot(dataset_id)

    # Rollback using the captured snapshot
    manager.rollback_dataset(dataset_id, snapshot_key=snapshot_key)

    # --- Assert round-trip equivalence ---
    # The update_dataset call should have been made
    assert "params" in update_call_args, "update_dataset was not called during rollback"
    assert update_call_args["dataset_id"] == dataset_id

    restored_params = update_call_args["params"]

    # Fields that _build_update_params transfers from the original definition
    # must be logically equivalent to the original
    transferable_fields = [
        "Name",
        "PhysicalTableMap",
        "LogicalTableMap",
        "ImportMode",
        "ColumnGroups",
        "FieldFolders",
        "RowLevelPermissionDataSet",
        "RowLevelPermissionTagConfiguration",
        "ColumnLevelPermissionRules",
        "DataSetUsageConfiguration",
        "DatasetParameters",
    ]

    for field in transferable_fields:
        if field in dataset_def:
            assert field in restored_params, (
                f"Field '{field}' present in original definition but missing from "
                f"restored params after rollback"
            )
            # Compare by serializing to JSON to handle any type coercion
            # from the json.dumps(default=str) / json.loads round-trip
            original_json = json.loads(json.dumps(dataset_def[field], default=str))
            restored_json = json.loads(json.dumps(restored_params[field], default=str))
            assert original_json == restored_json, (
                f"Field '{field}' differs after snapshot round-trip.\n"
                f"Original: {original_json}\n"
                f"Restored: {restored_json}"
            )
        else:
            # If field wasn't in original, it shouldn't be in restored params
            assert field not in restored_params, (
                f"Field '{field}' not in original definition but appeared "
                f"in restored params after rollback"
            )
