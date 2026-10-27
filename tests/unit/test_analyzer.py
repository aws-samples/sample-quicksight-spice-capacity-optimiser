# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the UsageAnalyzer module.

Tests extract_join_keys, classify_columns, check_safety_threshold,
get_unused_columns, and extract_dataset_calc_field_inputs with various
inputs including edge cases.

Requirements: 3.4, 4.4, 4.5
"""

from __future__ import annotations

import pytest

from lambda_source.analyzer import UsageAnalyzer
from lambda_source.models import ColumnClassification, ColumnStatus


@pytest.fixture
def analyzer() -> UsageAnalyzer:
    """Create a UsageAnalyzer instance for testing."""
    return UsageAnalyzer()


# ---------------------------------------------------------------------------
# extract_join_keys — OnClause formats
# ---------------------------------------------------------------------------


class TestExtractJoinKeysOnClause:
    """Tests for extract_join_keys with various OnClause formats."""

    def test_simple_on_clause(self, analyzer: UsageAnalyzer):
        """Simple 'col_a = col_b' OnClause."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "LeftOperand": "lt_left",
                        "RightOperand": "lt_right",
                        "OnClause": "col_a = col_b",
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"col_a", "col_b"}

    def test_on_clause_no_spaces_around_equals(self, analyzer: UsageAnalyzer):
        """OnClause with 'col_a=col_b' format (no spaces)."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "LeftOperand": "lt_left",
                        "RightOperand": "lt_right",
                        "OnClause": "col_a=col_b",
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"col_a", "col_b"}

    def test_multi_condition_on_clause(self, analyzer: UsageAnalyzer):
        """Multi-condition OnClause: 'col_a = col_b AND col_c = col_d'."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "LeftOperand": "lt_left",
                        "RightOperand": "lt_right",
                        "OnClause": "col_a = col_b AND col_c = col_d",
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"col_a", "col_b", "col_c", "col_d"}

    def test_empty_logical_table_map(self, analyzer: UsageAnalyzer):
        """Empty logical table map returns empty set."""
        result = analyzer.extract_join_keys({})
        assert result == set()

    def test_none_logical_table_map(self, analyzer: UsageAnalyzer):
        """None logical table map returns empty set."""
        result = analyzer.extract_join_keys(None)
        assert result == set()

    def test_logical_table_without_join_instruction(self, analyzer: UsageAnalyzer):
        """Logical table with PhysicalTableSource (no join) returns empty."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "PhysicalTableId": "phys1"
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == set()

    def test_multiple_join_instructions(self, analyzer: UsageAnalyzer):
        """Multiple logical tables each with JoinInstructions."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "id = customer_id",
                        "Type": "LEFT",
                    }
                }
            },
            "lt2": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "order_id = fk_order_id",
                        "Type": "INNER",
                    }
                }
            },
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"id", "customer_id", "order_id", "fk_order_id"}


# ---------------------------------------------------------------------------
# extract_join_keys — LeftJoinKeyProperties / RightJoinKeyProperties
# ---------------------------------------------------------------------------


class TestExtractJoinKeysKeyProperties:
    """Tests for extract_join_keys with LeftJoinKeyProperties/RightJoinKeyProperties."""

    def test_left_join_key_properties(self, analyzer: UsageAnalyzer):
        """LeftJoinKeyProperties with Columns list."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "",
                        "LeftJoinKeyProperties": {
                            "Columns": ["left_col_a", "left_col_b"]
                        },
                        "Type": "LEFT",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert "left_col_a" in result
        assert "left_col_b" in result

    def test_right_join_key_properties(self, analyzer: UsageAnalyzer):
        """RightJoinKeyProperties with Columns list."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "",
                        "RightJoinKeyProperties": {
                            "Columns": ["right_col_x", "right_col_y"]
                        },
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert "right_col_x" in result
        assert "right_col_y" in result

    def test_both_key_properties_combined_with_on_clause(self, analyzer: UsageAnalyzer):
        """Both LeftJoinKeyProperties and RightJoinKeyProperties plus OnClause."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "id = cust_id",
                        "LeftJoinKeyProperties": {
                            "Columns": ["left_unique"]
                        },
                        "RightJoinKeyProperties": {
                            "Columns": ["right_unique"]
                        },
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"id", "cust_id", "left_unique", "right_unique"}

    def test_empty_key_properties_columns(self, analyzer: UsageAnalyzer):
        """Empty Columns list in key properties."""
        logical_table_map = {
            "lt1": {
                "Source": {
                    "JoinInstruction": {
                        "OnClause": "a = b",
                        "LeftJoinKeyProperties": {"Columns": []},
                        "RightJoinKeyProperties": {"Columns": []},
                        "Type": "INNER",
                    }
                }
            }
        }
        result = analyzer.extract_join_keys(logical_table_map)
        assert result == {"a", "b"}


# ---------------------------------------------------------------------------
# classify_columns — zero consumers (all non-join-keys marked unused)
# ---------------------------------------------------------------------------


class TestClassifyColumnsZeroConsumers:
    """Tests for classify_columns when dataset has zero consumers."""

    def test_zero_consumers_all_columns_unused(self, analyzer: UsageAnalyzer):
        """Req 4.5: zero consumers → all non-join-keys classified as unused."""
        all_columns = [
            {"Name": "col_a", "Type": "STRING"},
            {"Name": "col_b", "Type": "INTEGER"},
            {"Name": "col_c", "Type": "DECIMAL"},
        ]
        # No consumer references, no join keys, no calc field inputs
        result = analyzer.classify_columns(all_columns, set(), set(), set())

        assert len(result) == 3
        for c in result:
            assert c.status == ColumnStatus.UNUSED.value
            assert "no consumer references" in c.reasons

    def test_zero_consumers_with_join_keys_preserved(self, analyzer: UsageAnalyzer):
        """Req 4.5: zero consumers but join keys still classified as join_key."""
        all_columns = [
            {"Name": "id", "Type": "STRING"},
            {"Name": "data_col", "Type": "INTEGER"},
            {"Name": "fk_id", "Type": "STRING"},
        ]
        join_keys = {"id", "fk_id"}
        result = analyzer.classify_columns(all_columns, set(), join_keys, set())

        statuses = {c.column_name: c.status for c in result}
        assert statuses["id"] == ColumnStatus.JOIN_KEY.value
        assert statuses["fk_id"] == ColumnStatus.JOIN_KEY.value
        assert statuses["data_col"] == ColumnStatus.UNUSED.value


# ---------------------------------------------------------------------------
# classify_columns — all columns used (empty unused list)
# ---------------------------------------------------------------------------


class TestClassifyColumnsAllUsed:
    """Tests for classify_columns when all columns are used."""

    def test_all_columns_used_via_consumer_references(self, analyzer: UsageAnalyzer):
        """Req 4.4: all columns used → no unused columns."""
        all_columns = [
            {"Name": "col_a", "Type": "STRING"},
            {"Name": "col_b", "Type": "INTEGER"},
            {"Name": "col_c", "Type": "DECIMAL"},
        ]
        used_columns = {"col_a", "col_b", "col_c"}
        result = analyzer.classify_columns(all_columns, used_columns, set(), set())

        for c in result:
            assert c.status == ColumnStatus.USED.value

        unused = analyzer.get_unused_columns(result)
        assert unused == []

    def test_all_columns_used_mix_of_sources(self, analyzer: UsageAnalyzer):
        """All columns used through mix of consumer refs, join keys, calc fields."""
        all_columns = [
            {"Name": "col_a", "Type": "STRING"},
            {"Name": "col_b", "Type": "INTEGER"},
            {"Name": "col_c", "Type": "DECIMAL"},
        ]
        used_columns = {"col_a"}
        join_keys = {"col_b"}
        calc_fields = {"col_c"}
        result = analyzer.classify_columns(all_columns, used_columns, join_keys, calc_fields)

        statuses = {c.column_name: c.status for c in result}
        assert statuses["col_a"] == ColumnStatus.USED.value
        assert statuses["col_b"] == ColumnStatus.JOIN_KEY.value
        assert statuses["col_c"] == ColumnStatus.USED.value

        unused = analyzer.get_unused_columns(result)
        assert unused == []

    def test_empty_column_list(self, analyzer: UsageAnalyzer):
        """Empty dataset produces empty classifications."""
        result = analyzer.classify_columns([], set(), set(), set())
        assert result == []


# ---------------------------------------------------------------------------
# check_safety_threshold — boundary cases
# ---------------------------------------------------------------------------


class TestCheckSafetyThreshold:
    """Tests for check_safety_threshold boundary cases."""

    def test_exactly_at_threshold_is_safe(self, analyzer: UsageAnalyzer):
        """Removal ratio exactly equal to threshold is within threshold (safe)."""
        # 7 out of 10 unused = 0.7 ratio, threshold = 0.7
        # ratio is NOT > threshold, so safe
        classifications = [
            ColumnClassification(column_name=f"used_{i}", data_type="STRING", status=ColumnStatus.USED.value)
            for i in range(3)
        ] + [
            ColumnClassification(column_name=f"unused_{i}", data_type="STRING", status=ColumnStatus.UNUSED.value)
            for i in range(7)
        ]
        assert analyzer.check_safety_threshold(classifications, 0.7) is True

    def test_above_threshold_is_unsafe(self, analyzer: UsageAnalyzer):
        """Removal ratio above threshold triggers breach (unsafe)."""
        # 8 out of 10 unused = 0.8 ratio, threshold = 0.7
        classifications = [
            ColumnClassification(column_name=f"used_{i}", data_type="STRING", status=ColumnStatus.USED.value)
            for i in range(2)
        ] + [
            ColumnClassification(column_name=f"unused_{i}", data_type="STRING", status=ColumnStatus.UNUSED.value)
            for i in range(8)
        ]
        assert analyzer.check_safety_threshold(classifications, 0.7) is False

    def test_below_threshold_is_safe(self, analyzer: UsageAnalyzer):
        """Removal ratio below threshold is within threshold (safe)."""
        # 3 out of 10 unused = 0.3 ratio, threshold = 0.7
        classifications = [
            ColumnClassification(column_name=f"used_{i}", data_type="STRING", status=ColumnStatus.USED.value)
            for i in range(7)
        ] + [
            ColumnClassification(column_name=f"unused_{i}", data_type="STRING", status=ColumnStatus.UNUSED.value)
            for i in range(3)
        ]
        assert analyzer.check_safety_threshold(classifications, 0.7) is True

    def test_zero_columns_returns_true(self, analyzer: UsageAnalyzer):
        """Zero columns edge case — returns True (safe)."""
        assert analyzer.check_safety_threshold([], 0.7) is True

    def test_threshold_zero_with_any_unused_breaches(self, analyzer: UsageAnalyzer):
        """Threshold 0.0 — any unused column triggers breach."""
        classifications = [
            ColumnClassification(column_name="used", data_type="STRING", status=ColumnStatus.USED.value),
            ColumnClassification(column_name="unused", data_type="STRING", status=ColumnStatus.UNUSED.value),
        ]
        assert analyzer.check_safety_threshold(classifications, 0.0) is False

    def test_threshold_one_never_breaches(self, analyzer: UsageAnalyzer):
        """Threshold 1.0 — never breached even when all columns unused."""
        classifications = [
            ColumnClassification(column_name=f"unused_{i}", data_type="STRING", status=ColumnStatus.UNUSED.value)
            for i in range(10)
        ]
        # 10/10 = 1.0, which is NOT > 1.0
        assert analyzer.check_safety_threshold(classifications, 1.0) is True

    def test_all_columns_used_never_breaches(self, analyzer: UsageAnalyzer):
        """No unused columns — never breaches any threshold."""
        classifications = [
            ColumnClassification(column_name=f"used_{i}", data_type="STRING", status=ColumnStatus.USED.value)
            for i in range(5)
        ]
        assert analyzer.check_safety_threshold(classifications, 0.0) is True

    def test_single_unused_column_at_boundary(self, analyzer: UsageAnalyzer):
        """Single column that is unused (1/1 = 1.0) at threshold 0.5 → breached."""
        classifications = [
            ColumnClassification(column_name="unused", data_type="STRING", status=ColumnStatus.UNUSED.value),
        ]
        assert analyzer.check_safety_threshold(classifications, 0.5) is False


# ---------------------------------------------------------------------------
# get_unused_columns — filter
# ---------------------------------------------------------------------------


class TestGetUnusedColumns:
    """Tests for get_unused_columns filter."""

    def test_filters_only_unused(self, analyzer: UsageAnalyzer):
        """Returns only columns with UNUSED status."""
        classifications = [
            ColumnClassification(column_name="used_col", data_type="STRING", status=ColumnStatus.USED.value),
            ColumnClassification(column_name="unused_col", data_type="INTEGER", status=ColumnStatus.UNUSED.value),
            ColumnClassification(column_name="join_col", data_type="STRING", status=ColumnStatus.JOIN_KEY.value),
            ColumnClassification(column_name="unused_col_2", data_type="DECIMAL", status=ColumnStatus.UNUSED.value),
        ]
        result = analyzer.get_unused_columns(classifications)
        assert len(result) == 2
        names = {c.column_name for c in result}
        assert names == {"unused_col", "unused_col_2"}

    def test_no_unused_columns_returns_empty(self, analyzer: UsageAnalyzer):
        """All used or join_key columns returns empty list."""
        classifications = [
            ColumnClassification(column_name="used", data_type="STRING", status=ColumnStatus.USED.value),
            ColumnClassification(column_name="join_key", data_type="STRING", status=ColumnStatus.JOIN_KEY.value),
        ]
        result = analyzer.get_unused_columns(classifications)
        assert result == []

    def test_empty_list_returns_empty(self, analyzer: UsageAnalyzer):
        """Empty input returns empty output."""
        result = analyzer.get_unused_columns([])
        assert result == []

    def test_all_unused_returns_all(self, analyzer: UsageAnalyzer):
        """All columns unused returns the full list."""
        classifications = [
            ColumnClassification(column_name=f"col_{i}", data_type="STRING", status=ColumnStatus.UNUSED.value)
            for i in range(5)
        ]
        result = analyzer.get_unused_columns(classifications)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# extract_dataset_calc_field_inputs
# ---------------------------------------------------------------------------


class TestExtractDatasetCalcFieldInputs:
    """Tests for extract_dataset_calc_field_inputs with both patterns."""

    def test_dataset_identifier_dot_column_pattern(self, analyzer: UsageAnalyzer):
        """Extracts columns from {DataSetId.Col} pattern."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "profit",
                    "Expression": "{ds_123.revenue} - {ds_123.cost}",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert result == {"revenue", "cost"}

    def test_bracket_column_pattern(self, analyzer: UsageAnalyzer):
        """Extracts columns from {[Col]} pattern."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "derived",
                    "Expression": "UPPER({[first_name]})",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert "first_name" in result

    def test_mixed_patterns(self, analyzer: UsageAnalyzer):
        """Both patterns in a single expression."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "mixed_calc",
                    "Expression": "CONCAT({ds_123.col_a}, {[col_b]})",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert "col_a" in result
        assert "col_b" in result

    def test_no_calculated_fields(self, analyzer: UsageAnalyzer):
        """No CalculatedFields key returns empty set."""
        dataset_def = {"DataSetId": "ds_123"}
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert result == set()

    def test_empty_calculated_fields_list(self, analyzer: UsageAnalyzer):
        """Empty CalculatedFields list returns empty set."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert result == set()

    def test_expression_without_column_refs(self, analyzer: UsageAnalyzer):
        """Expression with no column references returns empty set."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "constant",
                    "Expression": "42 + 10",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert result == set()

    def test_multiple_calc_fields(self, analyzer: UsageAnalyzer):
        """Multiple calculated fields combine all references."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "calc_a",
                    "Expression": "SUM({ds_123.amount})",
                },
                {
                    "Name": "calc_b",
                    "Expression": "AVG({[rating]})",
                },
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert result == {"amount", "rating"}

    def test_cross_dataset_reference_still_included(self, analyzer: UsageAnalyzer):
        """References to another dataset identifier are still included conservatively."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "cross_ref",
                    "Expression": "{other_dataset.foreign_col}",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        # Conservative approach includes all column refs
        assert "foreign_col" in result

    def test_duplicate_columns_deduplicated(self, analyzer: UsageAnalyzer):
        """Same column referenced multiple times returns it once."""
        dataset_def = {
            "DataSetId": "ds_123",
            "CalculatedFields": [
                {
                    "Name": "calc",
                    "Expression": "{ds_123.amount} + {ds_123.amount} * {[amount]}",
                }
            ],
        }
        result = analyzer.extract_dataset_calc_field_inputs(dataset_def)
        assert "amount" in result
        # It's a set so no duplicates
        assert isinstance(result, set)
