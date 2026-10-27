# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the Quarantine Manager module.

Tests correctness properties related to quarantine period filtering: only columns
whose first_detected timestamp is at least D days before current execution time
are eligible for removal, and columns that become referenced are removed from
quarantine tracking immediately.

Feature: spice-dataset-optimiser
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from lambda_source.quarantine import QuarantineManager
from tests.property.strategies import st_column_name


# ---------------------------------------------------------------------------
# Helpers: in-memory DynamoDB fake table
# ---------------------------------------------------------------------------


class FakeDynamoDBTable:
    """In-memory fake DynamoDB table for property testing."""

    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}

    def query(self, **kwargs):
        """Return all COL# items (simple in-memory query)."""
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
            self.items[(pk, sk)] = {
                "pk": pk,
                "sk": sk,
                "first_detected": expr_values.get(":now", ""),
                "last_confirmed": expr_values.get(":now", ""),
                "ttl": expr_values.get(":ttl", 0),
            }
        else:
            self.items[(pk, sk)]["last_confirmed"] = expr_values.get(":now", "")
            self.items[(pk, sk)]["ttl"] = expr_values.get(":ttl", 0)

    def delete_item(self, **kwargs):
        key = kwargs["Key"]
        pk = key["pk"]
        sk = key["sk"]
        self.items.pop((pk, sk), None)


def _make_fake_resource(table: FakeDynamoDBTable):
    """Create a fake DynamoDB resource returning the given table."""
    resource = MagicMock()
    resource.Table.return_value = table
    return resource


# ---------------------------------------------------------------------------
# Custom strategies for quarantine testing
# ---------------------------------------------------------------------------


@st.composite
def st_quarantine_days(draw: st.DrawFn) -> int:
    """Generate a valid quarantine period in days (1-90)."""
    return draw(st.integers(min_value=1, max_value=90))


@st.composite
def st_columns_with_timestamps(
    draw: st.DrawFn,
    quarantine_days: int,
    min_columns: int = 1,
    max_columns: int = 15,
) -> tuple[list[str], dict[str, datetime], list[str], list[str]]:
    """Generate columns with first_detected timestamps, some before and some after cutoff.

    Returns:
        Tuple of (all_column_names, timestamp_map, expected_eligible, expected_ineligible)
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=quarantine_days)

    num_columns = draw(st.integers(min_value=min_columns, max_value=max_columns))

    names: set[str] = set()
    while len(names) < num_columns:
        name = draw(st_column_name())
        names.add(name)

    column_names = sorted(names)
    timestamp_map: dict[str, datetime] = {}
    expected_eligible: list[str] = []
    expected_ineligible: list[str] = []

    for col in column_names:
        # Decide whether this column is before or after cutoff
        is_before_cutoff = draw(st.booleans())

        if is_before_cutoff:
            # Generate timestamp between quarantine_days and quarantine_days+180 days ago
            extra_days = draw(st.integers(min_value=0, max_value=180))
            ts = now - timedelta(days=quarantine_days + extra_days)
            timestamp_map[col] = ts
            expected_eligible.append(col)
        else:
            # Generate timestamp between 0 and quarantine_days-1 days ago (strictly after cutoff)
            days_ago = draw(st.integers(min_value=0, max_value=max(0, quarantine_days - 1)))
            # Ensure it's strictly after cutoff by adding some hours
            ts = now - timedelta(days=days_ago, hours=draw(st.integers(min_value=0, max_value=23)))
            # Double-check it's after cutoff
            if ts > cutoff:
                timestamp_map[col] = ts
                expected_ineligible.append(col)
            else:
                # Edge case: ended up at/before cutoff, treat as eligible
                timestamp_map[col] = ts
                expected_eligible.append(col)

    return column_names, timestamp_map, expected_eligible, expected_ineligible


# ---------------------------------------------------------------------------
# Property 14: Quarantine period filtering
# ---------------------------------------------------------------------------


class TestQuarantinePeriodFiltering:
    """Property 14: Quarantine period filtering.

    For any set of unused columns with first-detection timestamps, and a configured
    quarantine period of D days, only columns whose first_detected timestamp is at
    least D days before the current execution time SHALL be eligible for removal.
    Columns that become referenced between executions SHALL be removed from
    quarantine tracking immediately.

    Feature: spice-dataset-optimiser, Property 14: Quarantine period filtering
    **Validates: Requirements 9.1, 9.2, 9.4**
    """

    @given(
        quarantine_days=st.integers(min_value=1, max_value=90),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_only_columns_past_quarantine_period_are_eligible(
        self,
        quarantine_days: int,
        data: st.DataObject,
    ) -> None:
        """Only columns with first_detected <= (now - quarantine_days) are returned
        by filter_by_quarantine. Columns with first_detected > (now - quarantine_days)
        are NOT returned.

        **Validates: Requirements 9.1, 9.2**
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=quarantine_days)

        # Generate a set of unique column names
        num_columns = data.draw(st.integers(min_value=1, max_value=15))
        names: set[str] = set()
        while len(names) < num_columns:
            names.add(data.draw(st_column_name()))
        column_names = sorted(names)

        # Set up DynamoDB table with entries for each column
        table = FakeDynamoDBTable()
        dataset_id = "test-dataset-001"

        expected_eligible: list[str] = []
        expected_ineligible: list[str] = []

        for col in column_names:
            is_before_cutoff = data.draw(st.booleans())

            if is_before_cutoff:
                # first_detected is at least quarantine_days ago
                extra_days = data.draw(st.integers(min_value=0, max_value=180))
                extra_seconds = data.draw(st.integers(min_value=0, max_value=86399))
                ts = cutoff - timedelta(days=extra_days, seconds=extra_seconds)
            else:
                # first_detected is strictly after cutoff (within quarantine period)
                seconds_after_cutoff = data.draw(
                    st.integers(min_value=1, max_value=quarantine_days * 86400)
                )
                ts = cutoff + timedelta(seconds=seconds_after_cutoff)

            # Store in fake table
            table.items[(dataset_id, f"COL#{col}")] = {
                "pk": dataset_id,
                "sk": f"COL#{col}",
                "first_detected": ts.isoformat(),
                "last_confirmed": now.isoformat(),
                "ttl": Decimal(0),
            }

            # Classify based on actual cutoff comparison
            if ts <= cutoff:
                expected_eligible.append(col)
            else:
                expected_ineligible.append(col)

        # Create manager with fake table
        resource = _make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # Call filter_by_quarantine
        result = manager.filter_by_quarantine(
            column_names, dataset_id, quarantine_days
        )

        # Assert: only eligible columns are returned
        assert set(result) == set(expected_eligible), (
            f"Expected eligible: {sorted(expected_eligible)}, "
            f"Got: {sorted(result)}, "
            f"Quarantine days: {quarantine_days}"
        )

        # Assert: no ineligible columns are returned
        for col in expected_ineligible:
            assert col not in result, (
                f"Column '{col}' should NOT be eligible but was returned. "
                f"Quarantine days: {quarantine_days}"
            )

    @given(
        quarantine_days=st.integers(min_value=1, max_value=90),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_columns_become_referenced_removed_from_tracking(
        self,
        quarantine_days: int,
        data: st.DataObject,
    ) -> None:
        """Columns that become referenced between executions SHALL be removed
        from quarantine tracking immediately via remove_from_tracking.

        **Validates: Requirements 9.4**
        """
        now = datetime.now(timezone.utc)

        # Generate columns currently in quarantine tracking
        num_tracked = data.draw(st.integers(min_value=2, max_value=15))
        names: set[str] = set()
        while len(names) < num_tracked:
            names.add(data.draw(st_column_name()))
        tracked_columns = sorted(names)

        # Set up DynamoDB table with quarantine entries
        table = FakeDynamoDBTable()
        dataset_id = "test-dataset-002"

        for col in tracked_columns:
            days_ago = data.draw(st.integers(min_value=1, max_value=180))
            ts = now - timedelta(days=days_ago)
            table.items[(dataset_id, f"COL#{col}")] = {
                "pk": dataset_id,
                "sk": f"COL#{col}",
                "first_detected": ts.isoformat(),
                "last_confirmed": now.isoformat(),
                "ttl": Decimal(0),
            }

        # Choose a non-empty subset of columns that "become referenced"
        num_now_referenced = data.draw(
            st.integers(min_value=1, max_value=len(tracked_columns))
        )
        now_referenced = data.draw(
            st.sampled_from(
                sorted(
                    [
                        combo
                        for combo in _combinations(tracked_columns, num_now_referenced)
                    ]
                )
            )
            if num_now_referenced <= len(tracked_columns)
            else st.just(tracked_columns)
        )

        # Create manager
        resource = _make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        # Call remove_from_tracking for the columns that became referenced
        manager.remove_from_tracking(dataset_id, list(now_referenced))

        # Assert: removed columns no longer exist in the table
        for col in now_referenced:
            assert (dataset_id, f"COL#{col}") not in table.items, (
                f"Column '{col}' should have been removed from tracking "
                f"after becoming referenced"
            )

        # Assert: columns NOT removed still exist in the table
        still_tracked = set(tracked_columns) - set(now_referenced)
        for col in still_tracked:
            assert (dataset_id, f"COL#{col}") in table.items, (
                f"Column '{col}' should still be in tracking "
                f"(it was not referenced)"
            )

    @given(
        quarantine_days=st.integers(min_value=1, max_value=90),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_untracked_columns_never_eligible(
        self,
        quarantine_days: int,
        data: st.DataObject,
    ) -> None:
        """Columns without a quarantine entry (not yet tracked) SHALL NOT be
        returned as eligible for removal.

        **Validates: Requirements 9.1, 9.2**
        """
        # Generate columns that have NO entry in DynamoDB
        num_columns = data.draw(st.integers(min_value=1, max_value=10))
        names: set[str] = set()
        while len(names) < num_columns:
            names.add(data.draw(st_column_name()))
        untracked_columns = sorted(names)

        # Empty table — no entries exist
        table = FakeDynamoDBTable()
        dataset_id = "test-dataset-003"

        resource = _make_fake_resource(table)
        manager = QuarantineManager(
            dynamodb_table="test-table", dynamodb_resource=resource
        )

        result = manager.filter_by_quarantine(
            untracked_columns, dataset_id, quarantine_days
        )

        # Assert: no untracked columns are eligible
        assert result == [], (
            f"Untracked columns should never be eligible, but got: {result}"
        )


# ---------------------------------------------------------------------------
# Helper: generate combinations for subset selection
# ---------------------------------------------------------------------------


def _combinations(items: list[str], k: int) -> list[tuple[str, ...]]:
    """Generate all k-combinations of items (limited for performance)."""
    from itertools import combinations

    result = list(combinations(items, k))
    # Limit to first 50 to avoid blow-up in hypothesis
    return result[:50] if len(result) > 50 else result
