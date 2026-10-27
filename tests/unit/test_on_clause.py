# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared OnClause parsing.

The clauses in TestRealClausesFromLiveAccount are copied verbatim from a real
QuickSight account on 2026-08-25. Under the previous parser, `inventory_id` — the
right-hand side of `inventories.csv`'s own join clause and a real output column —
was classified unused and reached the removal list of a generated plan.
"""

from __future__ import annotations

import pytest

from lambda_source.on_clause import (
    candidate_names,
    join_keys_from_data_prep,
    join_keys_from_dataset,
    join_keys_from_logical_table_map,
    parse_on_clause,
    strip_braces,
)


class TestStripBraces:
    """Single-name reduction, as ChildDatasetDiscovery needs."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("id", "id"),
            ("{inventory_id}", "inventory_id"),
            ("{part_num[parts]}", "part_num"),
            ("part_num[parts]", "part_num"),
            ("  {spaced}  ", "spaced"),
            ("", ""),
        ],
    )
    def test_reduces_to_the_bare_column(self, raw: str, expected: str):
        assert strip_braces(raw) == expected


class TestCandidateNames:
    def test_bare_column(self):
        assert candidate_names("id") == {"id"}

    def test_braced_column(self):
        assert candidate_names("{inventory_id}") == {"inventory_id"}

    def test_table_qualified_yields_both_readings(self):
        """`part_num[parts]` is a table-qualified reference on some datasets and a
        real materialised column on others. Both exist on inventories.csv."""
        assert candidate_names("{part_num[parts]}") == {"part_num[parts]", "part_num"}

    @pytest.mark.parametrize("empty", ["", "   ", "{}", "{ }"])
    def test_empty_yields_nothing(self, empty: str):
        assert candidate_names(empty) == set()


class TestParseOnClause:
    def test_simple_equality(self):
        assert parse_on_clause("col_a = col_b") == {"col_a", "col_b"}

    def test_no_spaces(self):
        assert parse_on_clause("col_a=col_b") == {"col_a", "col_b"}

    def test_multi_condition(self):
        assert parse_on_clause("region = region AND year = year") == {"region", "year"}

    def test_and_is_case_insensitive(self):
        assert parse_on_clause("a = b and c = d") == {"a", "b", "c", "d"}

    def test_empty(self):
        assert parse_on_clause("") == set()


class TestRealClausesFromLiveAccount:
    def test_inventories_csv(self):
        """The clause that put a join key into a removal list."""
        assert parse_on_clause("id = {inventory_id}") == {"id", "inventory_id"}

    def test_inventory_parts_csv(self):
        """Previously neither side was usable."""
        assert parse_on_clause("{part_num} = {part_num[parts]}") == {
            "part_num",
            "part_num[parts]",
        }

    def test_customers_join_sales_orders(self):
        assert parse_on_clause("{customer_id} = {customer_id}") == {"customer_id"}


class TestLegacyExtraction:
    def test_join_instruction(self):
        ltm = {"lt": {"Source": {"JoinInstruction": {"OnClause": "id = {inv_id}"}}}}
        assert join_keys_from_logical_table_map(ltm) == {"id", "inv_id"}

    def test_declared_key_properties(self):
        ltm = {
            "lt": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "a = b",
                        "LeftJoinKeyProperties": {"Columns": ["lk"]},
                        "RightJoinKeyProperties": {"Columns": ["rk"]},
                    }
                }
            }
        }
        assert join_keys_from_logical_table_map(ltm) == {"a", "b", "lk", "rk"}

    def test_multiple_logical_tables(self):
        ltm = {
            "a": {"Source": {"JoinInstruction": {"OnClause": "{p} = {q}"}}},
            "b": {"Source": {"JoinInstruction": {"OnClause": "{r} = {s}"}}},
        }
        assert join_keys_from_logical_table_map(ltm) == {"p", "q", "r", "s"}

    @pytest.mark.parametrize(
        "ltm",
        [None, {}, {"lt": None}, {"lt": {"Source": None}}, {"lt": {"Source": {"JoinInstruction": None}}}],
    )
    def test_malformed_never_raises(self, ltm):
        assert join_keys_from_logical_table_map(ltm) == set()


class TestNewExperienceExtraction:
    """Nothing read this for the target dataset before, so new-experience datasets
    had no join key protection at all."""

    def test_join_step(self):
        dp = {"TransformStepMap": {"s": {"JoinStep": {"OnClause": "{cid} = {cid}"}}}}
        assert join_keys_from_data_prep(dp) == {"cid"}

    def test_ignores_other_step_types(self):
        dp = {
            "TransformStepMap": {
                "s1": {"ProjectStep": {"ProjectedColumns": ["a"]}},
                "s2": {"JoinStep": {"OnClause": "{k} = {k}"}},
            }
        }
        assert join_keys_from_data_prep(dp) == {"k"}

    @pytest.mark.parametrize(
        "dp", [None, {}, {"TransformStepMap": None}, {"TransformStepMap": {"s": None}}]
    )
    def test_malformed_never_raises(self, dp):
        assert join_keys_from_data_prep(dp) == set()


class TestDatasetWideExtraction:
    def test_covers_both_shapes_at_once(self):
        dataset = {
            "LogicalTableMap": {
                "lt": {"Source": {"JoinInstruction": {"OnClause": "{legacy} = x"}}}
            },
            "DataPrepConfiguration": {
                "TransformStepMap": {"s": {"JoinStep": {"OnClause": "{modern} = y"}}}
            },
        }
        assert join_keys_from_dataset(dataset) == {"legacy", "x", "modern", "y"}

    def test_intersects_with_real_columns(self):
        dataset = {
            "LogicalTableMap": {
                "lt": {"Source": {"JoinInstruction": {"OnClause": "id = {ghost}"}}}
            }
        }
        assert join_keys_from_dataset(dataset, {"id", "other"}) == {"id"}

    def test_unprojected_join_key_is_dropped(self):
        """A join key that is not an output column cannot be removed anyway.

        This is the live situation on customers_SO_join_sales_orders_SO.json, which
        joins on customer_id without projecting it.
        """
        dataset = {
            "DataPrepConfiguration": {
                "TransformStepMap": {"s": {"JoinStep": {"OnClause": "{customer_id} = {customer_id}"}}}
            }
        }
        assert join_keys_from_dataset(dataset, {"region", "sales"}) == set()

    def test_empty_dataset(self):
        assert join_keys_from_dataset({}) == set()
