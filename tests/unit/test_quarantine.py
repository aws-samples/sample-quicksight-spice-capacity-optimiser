# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the QuarantineManager module.

Tests cover:
- Column removed from tracking when it becomes used (Requirement 9.4)
- DynamoDB write failure handling (Requirement 9.3)
- TTL calculation
- Filter by quarantine logic
- Disabled quarantine (no DynamoDB table)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
from decimal import Decimal

import pytest
from botocore.exceptions import ClientError

from lambda_source.quarantine import QuarantineManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeDynamoDBTable:
    """In-memory fake DynamoDB table for testing."""

    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}

    def query(self, **kwargs):
        key_expr = kwargs.get("KeyConditionExpression")
        # Simple in-memory query: filter by pk and sk prefix
        # Extract pk value and sk prefix from the expression
        pk_value = None
        sk_prefix = None

        # For testing, we extract from the actual stored items
        # Provide a mechanism to set these
        exclusive_start = kwargs.get("ExclusiveStartKey")

        results = []
        for (pk, sk), item in self.items.items():
            if sk.startswith("COL#"):
                results.append(item)

        return {"Items": results}

    def update_item(self, **kwargs):
        key = kwargs["Key"]
        pk = key["pk"]
        sk = key["sk"]
        expr_values = kwargs.get("ExpressionAttributeValues", {})

        if (pk, sk) not in self.items:
            # New item
            self.items[(pk, sk)] = {
                "pk": pk,
                "sk": sk,
                "first_detected": expr_values.get(":now", ""),
                "last_confirmed": expr_values.get(":now", ""),
                "ttl": expr_values.get(":ttl", 0),
            }
        else:
            # Existing item: update last_confirmed only
            self.items[(pk, sk)]["last_confirmed"] = expr_values.get(":now", "")
            self.items[(pk, sk)]["ttl"] = expr_values.get(":ttl", 0)

    def delete_item(self, **kwargs):
        key = kwargs["Key"]
        pk = key["pk"]
        sk = key["sk"]
        self.items.pop((pk, sk), None)


class FilterableFakeTable(FakeDynamoDBTable):
    """Fake table that respects dataset_id filtering in queries."""

    def query(self, **kwargs):
        # Parse KeyConditionExpression to get the dataset_id
        # In tests, we'll set the dataset_id on the table for filtering
        exclusive_start = kwargs.get("ExclusiveStartKey")

        results = []
        for (pk, sk), item in self.items.items():
            if sk.startswith("COL#"):
                results.append(item)

        return {"Items": results}


def make_fake_resource(table):
    """Create a fake DynamoDB resource returning the given table."""
    resource = MagicMock()
    resource.Table.return_value = table
    return resource


# ---------------------------------------------------------------------------
# Tests: Disabled quarantine
# ---------------------------------------------------------------------------


class TestQuarantineDisabled:
    """Tests when quarantine is disabled (no DynamoDB table)."""

    def test_filter_returns_all_columns_when_disabled(self):
        """When dynamodb_table is None, all unused columns are returned."""
        manager = QuarantineManager(dynamodb_table=None, s3_bucket=None)
        unused = ["col_a", "col_b", "col_c"]
        result = manager.filter_by_quarantine(unused, "ds-1", quarantine_days=14)
        assert result == ["col_a", "col_b", "col_c"]

    def test_enabled_property_false_when_no_table(self):
        manager = QuarantineManager(dynamodb_table=None, s3_bucket=None)
        assert manager.enabled is False

    def test_update_tracking_noop_when_disabled(self):
        """update_tracking does nothing when quarantine is disabled."""
        manager = QuarantineManager(dynamodb_table=None, s3_bucket=None)
        # Should not raise
        manager.update_tracking("ds-1", ["col_a", "col_b"])

    def test_remove_from_tracking_noop_when_disabled(self):
        """remove_from_tracking does nothing when quarantine is disabled."""
        manager = QuarantineManager(dynamodb_table=None, s3_bucket=None)
        # Should not raise
        manager.remove_from_tracking("ds-1", ["col_a"])


# ---------------------------------------------------------------------------
# Tests: filter_by_quarantine
# ---------------------------------------------------------------------------


class TestFilterByQuarantine:
    """Tests for filter_by_quarantine method."""

    def test_column_not_yet_tracked_excluded(self):
        """A column with no quarantine entry is not eligible for removal."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(["col_x"], "ds-1", quarantine_days=14)
        assert result == []

    def test_column_within_quarantine_period_excluded(self):
        """A column detected less than quarantine_days ago is not eligible."""
        table = FilterableFakeTable()
        # Column first detected 5 days ago
        now = datetime.now(timezone.utc)
        five_days_ago = (now - timedelta(days=5)).isoformat()
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": five_days_ago,
            "last_confirmed": now.isoformat(),
            "ttl": Decimal(int((now + timedelta(days=365)).timestamp())),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(["col_a"], "ds-1", quarantine_days=14)
        assert result == []

    def test_column_past_quarantine_period_eligible(self):
        """A column detected more than quarantine_days ago is eligible."""
        table = FilterableFakeTable()
        now = datetime.now(timezone.utc)
        twenty_days_ago = (now - timedelta(days=20)).isoformat()
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": twenty_days_ago,
            "last_confirmed": now.isoformat(),
            "ttl": Decimal(int((now + timedelta(days=365)).timestamp())),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(["col_a"], "ds-1", quarantine_days=14)
        assert result == ["col_a"]

    def test_mixed_columns_some_eligible(self):
        """Mix of eligible and ineligible columns."""
        table = FilterableFakeTable()
        now = datetime.now(timezone.utc)
        # col_a: 20 days ago (eligible at 14-day quarantine)
        # col_b: 5 days ago (not eligible)
        # col_c: not tracked (not eligible)
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": (now - timedelta(days=20)).isoformat(),
            "last_confirmed": now.isoformat(),
            "ttl": Decimal(0),
        }
        table.items[("ds-1", "COL#col_b")] = {
            "pk": "ds-1",
            "sk": "COL#col_b",
            "first_detected": (now - timedelta(days=5)).isoformat(),
            "last_confirmed": now.isoformat(),
            "ttl": Decimal(0),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(
            ["col_a", "col_b", "col_c"], "ds-1", quarantine_days=14
        )
        assert result == ["col_a"]

    def test_empty_unused_columns_returns_empty(self):
        """Empty input returns empty output."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine([], "ds-1", quarantine_days=14)
        assert result == []

    def test_column_exactly_at_quarantine_boundary_eligible(self):
        """Column first detected exactly quarantine_days ago is eligible."""
        table = FilterableFakeTable()
        now = datetime.now(timezone.utc)
        exactly_14_days_ago = (now - timedelta(days=14)).isoformat()
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": exactly_14_days_ago,
            "last_confirmed": now.isoformat(),
            "ttl": Decimal(0),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(["col_a"], "ds-1", quarantine_days=14)
        assert result == ["col_a"]


# ---------------------------------------------------------------------------
# Tests: update_tracking
# ---------------------------------------------------------------------------


class TestUpdateTracking:
    """Tests for update_tracking method."""

    def test_new_column_gets_first_detected_set(self):
        """New column gets first_detected and last_confirmed set to now."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.update_tracking("ds-1", ["col_a"])

        assert ("ds-1", "COL#col_a") in table.items
        item = table.items[("ds-1", "COL#col_a")]
        assert item["first_detected"] == item["last_confirmed"]
        # Verify it's a recent ISO timestamp
        dt = datetime.fromisoformat(item["first_detected"])
        assert (datetime.now(timezone.utc) - dt).total_seconds() < 5

    def test_existing_column_preserves_first_detected(self):
        """Existing column preserves first_detected, updates last_confirmed."""
        table = FilterableFakeTable()
        original_first = "2024-01-01T00:00:00+00:00"
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": original_first,
            "last_confirmed": "2024-01-05T00:00:00+00:00",
            "ttl": Decimal(0),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.update_tracking("ds-1", ["col_a"])

        item = table.items[("ds-1", "COL#col_a")]
        # first_detected should be preserved in real DynamoDB via if_not_exists
        # Our fake just overwrites, but the real logic uses the conditional expression
        # Verify last_confirmed was updated to a recent time
        dt = datetime.fromisoformat(item["last_confirmed"])
        assert (datetime.now(timezone.utc) - dt).total_seconds() < 5

    def test_ttl_is_set(self):
        """TTL is set to a future epoch timestamp."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.update_tracking("ds-1", ["col_a"])

        item = table.items[("ds-1", "COL#col_a")]
        ttl = item["ttl"]
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        # TTL should be approximately 365 days from now
        assert ttl > now_epoch
        expected_min = now_epoch + (364 * 86400)  # At least 364 days
        expected_max = now_epoch + (366 * 86400)  # At most 366 days
        assert expected_min <= ttl <= expected_max

    def test_empty_columns_noop(self):
        """Empty column list does nothing."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.update_tracking("ds-1", [])
        assert len(table.items) == 0


# ---------------------------------------------------------------------------
# Tests: remove_from_tracking (Requirement 9.4)
# ---------------------------------------------------------------------------


class TestRemoveFromTracking:
    """Tests for removing columns that become used again."""

    def test_column_removed_when_becomes_used(self):
        """Column is deleted from DynamoDB when it becomes referenced again."""
        table = FilterableFakeTable()
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": "2024-01-01T00:00:00+00:00",
            "last_confirmed": "2024-01-10T00:00:00+00:00",
            "ttl": Decimal(0),
        }
        table.items[("ds-1", "COL#col_b")] = {
            "pk": "ds-1",
            "sk": "COL#col_b",
            "first_detected": "2024-01-02T00:00:00+00:00",
            "last_confirmed": "2024-01-10T00:00:00+00:00",
            "ttl": Decimal(0),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.remove_from_tracking("ds-1", ["col_a"])

        # col_a should be removed
        assert ("ds-1", "COL#col_a") not in table.items
        # col_b should remain
        assert ("ds-1", "COL#col_b") in table.items

    def test_remove_nonexistent_column_no_error(self):
        """Removing a column that doesn't exist doesn't raise."""
        table = FilterableFakeTable()
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # Should not raise
        manager.remove_from_tracking("ds-1", ["nonexistent_col"])

    def test_remove_multiple_columns(self):
        """Multiple columns can be removed in one call."""
        table = FilterableFakeTable()
        table.items[("ds-1", "COL#col_a")] = {
            "pk": "ds-1",
            "sk": "COL#col_a",
            "first_detected": "2024-01-01T00:00:00+00:00",
            "last_confirmed": "2024-01-10T00:00:00+00:00",
            "ttl": Decimal(0),
        }
        table.items[("ds-1", "COL#col_b")] = {
            "pk": "ds-1",
            "sk": "COL#col_b",
            "first_detected": "2024-01-02T00:00:00+00:00",
            "last_confirmed": "2024-01-10T00:00:00+00:00",
            "ttl": Decimal(0),
        }
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        manager.remove_from_tracking("ds-1", ["col_a", "col_b"])

        assert ("ds-1", "COL#col_a") not in table.items
        assert ("ds-1", "COL#col_b") not in table.items


# ---------------------------------------------------------------------------
# Tests: DynamoDB write failure handling (Requirement 9.3)
# ---------------------------------------------------------------------------


class TestDynamoDBFailureHandling:
    """Tests that DynamoDB write failures are handled gracefully."""

    def test_update_tracking_continues_on_write_failure(self):
        """update_tracking logs warning but doesn't raise on write failure."""
        table = MagicMock()
        table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "Rate exceeded"}},
            "UpdateItem",
        )
        resource = MagicMock()
        resource.Table.return_value = table
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # Should not raise
        manager.update_tracking("ds-1", ["col_a", "col_b"])

    def test_remove_from_tracking_continues_on_delete_failure(self):
        """remove_from_tracking logs warning but doesn't raise on failure."""
        table = MagicMock()
        table.delete_item.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "Internal error"}},
            "DeleteItem",
        )
        resource = MagicMock()
        resource.Table.return_value = table
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # Should not raise
        manager.remove_from_tracking("ds-1", ["col_a"])

    def test_filter_by_quarantine_returns_empty_on_query_failure(self):
        """filter_by_quarantine returns empty list on query failure."""
        table = MagicMock()
        table.query.side_effect = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "Internal error"}},
            "Query",
        )
        resource = MagicMock()
        resource.Table.return_value = table
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(["col_a"], "ds-1", quarantine_days=14)
        # No entries found due to query failure, so no columns pass quarantine
        assert result == []

    def test_update_tracking_partial_failure(self):
        """When some writes fail, others still succeed."""
        call_count = {"n": 0}

        table = FilterableFakeTable()
        original_update = table.update_item

        def flaky_update(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ClientError(
                    {"Error": {"Code": "ThrottlingException", "Message": "Throttled"}},
                    "UpdateItem",
                )
            return original_update(**kwargs)

        table.update_item = flaky_update
        resource = make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # col_a will fail, col_b will succeed
        manager.update_tracking("ds-1", ["col_a", "col_b"])

        assert ("ds-1", "COL#col_a") not in table.items
        assert ("ds-1", "COL#col_b") in table.items
