# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the ColumnScanner module.

Tests each extraction method with realistic QuickSight definition structures,
validates scoping to correct DataSetIdentifier, and verifies the conservative
handling of unparseable calculated field expressions.
"""

from __future__ import annotations

import logging

import pytest

from lambda_source.scanner import ColumnScanner


@pytest.fixture
def scanner() -> ColumnScanner:
    """Create a ColumnScanner instance for testing."""
    return ColumnScanner()


# ---------------------------------------------------------------------------
# Visual field well extraction
# ---------------------------------------------------------------------------


class TestExtractVisualColumns:
    """Tests for _extract_visual_columns and visual-related extraction."""

    def test_extracts_dimensions_and_measures_from_table_visual(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {
                            "FieldWells": {
                                "TableAggregatedFieldWells": {
                                    "GroupBy": [{
                                        "CategoricalDimensionField": {
                                            "FieldId": "d1",
                                            "Column": {
                                                "DataSetIdentifier": "ds_1",
                                                "ColumnName": "region",
                                            },
                                        }
                                    }],
                                    "Values": [{
                                        "NumericalMeasureField": {
                                            "FieldId": "m1",
                                            "Column": {
                                                "DataSetIdentifier": "ds_1",
                                                "ColumnName": "revenue",
                                            },
                                        }
                                    }],
                                }
                            }
                        }
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"region", "revenue"}

    def test_extracts_from_bar_chart_visual(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "BarChartVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {
                            "FieldWells": {
                                "BarChartAggregatedFieldWells": {
                                    "Category": [{
                                        "CategoricalDimensionField": {
                                            "FieldId": "c1",
                                            "Column": {
                                                "DataSetIdentifier": "ds_1",
                                                "ColumnName": "product",
                                            },
                                        }
                                    }],
                                    "Values": [{
                                        "NumericalMeasureField": {
                                            "FieldId": "v1",
                                            "Column": {
                                                "DataSetIdentifier": "ds_1",
                                                "ColumnName": "sales",
                                            },
                                        }
                                    }],
                                }
                            }
                        }
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"product", "sales"}

    def test_extracts_tooltip_columns(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "LineChartVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {
                            "FieldWells": {},
                            "Tooltip": {
                                "TooltipVisibility": "VISIBLE",
                                "FieldBasedTooltip": {
                                    "TooltipFields": [{
                                        "FieldTooltipItem": {
                                            "FieldId": "tt1",
                                            "Column": {
                                                "DataSetIdentifier": "ds_1",
                                                "ColumnName": "description",
                                            },
                                        }
                                    }]
                                }
                            },
                        },
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert "description" in result

    def test_extracts_from_multiple_visuals_across_sheets(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [
                {
                    "SheetId": "s1",
                    "Visuals": [{
                        "PieChartVisual": {
                            "VisualId": "v1",
                            "ChartConfiguration": {
                                "FieldWells": {
                                    "PieChartAggregatedFieldWells": {
                                        "Category": [{
                                            "CategoricalDimensionField": {
                                                "FieldId": "c1",
                                                "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "category"},
                                            }
                                        }],
                                    }
                                }
                            }
                        }
                    }],
                },
                {
                    "SheetId": "s2",
                    "Visuals": [{
                        "TableVisual": {
                            "VisualId": "v2",
                            "ChartConfiguration": {
                                "FieldWells": {
                                    "TableAggregatedFieldWells": {
                                        "GroupBy": [{
                                            "CategoricalDimensionField": {
                                                "FieldId": "d1",
                                                "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "name"},
                                            }
                                        }],
                                    }
                                }
                            }
                        }
                    }],
                },
            ],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"category", "name"}

    def test_ignores_columns_from_other_datasets(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {
                            "FieldWells": {
                                "TableAggregatedFieldWells": {
                                    "GroupBy": [
                                        {"CategoricalDimensionField": {"FieldId": "d1", "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "mine"}}},
                                        {"CategoricalDimensionField": {"FieldId": "d2", "Column": {"DataSetIdentifier": "other_ds", "ColumnName": "theirs"}}},
                                    ],
                                }
                            }
                        }
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"mine"}
        assert "theirs" not in result


# ---------------------------------------------------------------------------
# Calculated field extraction
# ---------------------------------------------------------------------------


class TestExtractCalculatedFieldColumns:
    """Tests for _extract_calculated_field_columns."""

    def test_extracts_single_column_reference(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "calc1",
                "Expression": "sum({ds_1.amount})",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"amount"}

    def test_extracts_multiple_column_references(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "profit_margin",
                "Expression": "ifelse({ds_1.revenue} > 0, {ds_1.profit} / {ds_1.revenue}, 0)",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"revenue", "profit"}

    def test_ignores_columns_from_other_datasets_in_expression(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "cross_ds",
                "Expression": "ifelse({ds_1.col_a} > {other.col_b}, 1, 0)",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"col_a"}
        assert "col_b" not in result

    def test_ignores_calc_field_scoped_to_different_dataset(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "other_ds",
                "Name": "other_calc",
                "Expression": "sum({other_ds.amount})",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()

    def test_unparseable_expression_returns_sentinel(self, scanner: ColumnScanner, caplog):
        """Req 2.9: unparseable expression logs warning and returns sentinel."""
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "broken_calc",
                "Expression": "{this is not a valid pattern but has braces}",
            }],
        }
        with caplog.at_level(logging.WARNING):
            result = scanner.scan_definition(definition, "ds_1", "arn:test")

        assert "__UNPARSEABLE__" in result
        assert "Cannot parse calculated field expression" in caplog.text

    def test_expression_without_column_refs_returns_empty(self, scanner: ColumnScanner):
        """A literal expression with no braces at all."""
        definition = {
            "Sheets": [],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "constant",
                "Expression": "42",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()

    def test_multiple_calc_fields_combined(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "CalculatedFields": [
                {
                    "DataSetIdentifier": "ds_1",
                    "Name": "calc_a",
                    "Expression": "strlen({ds_1.name})",
                },
                {
                    "DataSetIdentifier": "ds_1",
                    "Name": "calc_b",
                    "Expression": "avg({ds_1.score})",
                },
            ],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"name", "score"}


# ---------------------------------------------------------------------------
# Filter extraction
# ---------------------------------------------------------------------------


class TestExtractFilterColumns:
    """Tests for _extract_filter_columns."""

    def test_extracts_category_filter_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "CategoryFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "status"},
                        "Configuration": {},
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"status"}

    def test_extracts_numeric_range_filter_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "NumericRangeFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "price"},
                        "RangeMinimum": {"StaticValue": 0},
                        "RangeMaximum": {"StaticValue": 100},
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"price"}

    def test_extracts_time_range_filter_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "TimeRangeFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "order_date"},
                        "NullOption": "ALL_VALUES",
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"order_date"}

    def test_extracts_relative_datetime_filter_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "RelativeDateTimeFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "created_at"},
                        "RelativeDateType": "LAST",
                        "TimeGranularity": "DAY",
                        "RelativeDateValue": 7,
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"created_at"}

    def test_extracts_top_bottom_filter_with_aggregation_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "TopBottomFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "product"},
                        "Limit": 10,
                        "AggregationSortConfigurations": [{
                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "total_sales"},
                            "SortDirection": "DESC",
                        }],
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"product", "total_sales"}

    def test_extracts_numeric_equality_filter(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "NumericEqualityFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "quantity"},
                        "Value": 5,
                        "MatchOperator": "EQUALS",
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"quantity"}

    def test_multiple_filters_across_groups(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [
                {
                    "FilterGroupId": "fg1",
                    "Filters": [{
                        "CategoryFilter": {
                            "FilterId": "f1",
                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "region"},
                        }
                    }],
                },
                {
                    "FilterGroupId": "fg2",
                    "Filters": [{
                        "NumericRangeFilter": {
                            "FilterId": "f2",
                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "amount"},
                        }
                    }],
                },
            ],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"region", "amount"}


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------


class TestExtractParameterColumns:
    """Tests for _extract_parameter_columns."""

    def test_extracts_dynamic_default_value_column(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ParameterDeclarations": [{
                "StringParameterDeclaration": {
                    "Name": "param1",
                    "DefaultValues": {
                        "DynamicValue": {
                            "DefaultValueColumn": {
                                "DataSetIdentifier": "ds_1",
                                "ColumnName": "default_region",
                            },
                        }
                    },
                }
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert "default_region" in result

    def test_extracts_username_and_groupname_columns(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ParameterDeclarations": [{
                "StringParameterDeclaration": {
                    "Name": "param1",
                    "DefaultValues": {
                        "DynamicValue": {
                            "DefaultValueColumn": {
                                "DataSetIdentifier": "ds_1",
                                "ColumnName": "value_col",
                            },
                            "UserNameColumn": {
                                "DataSetIdentifier": "ds_1",
                                "ColumnName": "user_col",
                            },
                            "GroupNameColumn": {
                                "DataSetIdentifier": "ds_1",
                                "ColumnName": "group_col",
                            },
                        }
                    },
                }
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"value_col", "user_col", "group_col"}

    def test_ignores_static_defaults(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ParameterDeclarations": [{
                "IntegerParameterDeclaration": {
                    "Name": "static_param",
                    "DefaultValues": {"StaticValues": [0]},
                }
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()


# ---------------------------------------------------------------------------
# Drill-down hierarchy extraction
# ---------------------------------------------------------------------------


class TestExtractDrillDownColumns:
    """Tests for _extract_drill_down_columns."""

    def test_extracts_visual_level_hierarchy(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "BarChartVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {"FieldWells": {}},
                        "ColumnHierarchies": [{
                            "HierarchyId": "h1",
                            "Columns": [
                                {"DataSetIdentifier": "ds_1", "ColumnName": "year"},
                                {"DataSetIdentifier": "ds_1", "ColumnName": "quarter"},
                                {"DataSetIdentifier": "ds_1", "ColumnName": "month"},
                            ],
                        }],
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"year", "quarter", "month"}

    def test_extracts_definition_level_hierarchy(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ColumnHierarchies": [{
                "HierarchyId": "h1",
                "Columns": [
                    {"DataSetIdentifier": "ds_1", "ColumnName": "continent"},
                    {"DataSetIdentifier": "ds_1", "ColumnName": "country"},
                ],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"continent", "country"}

    def test_ignores_hierarchy_for_other_dataset(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "BarChartVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {"FieldWells": {}},
                        "ColumnHierarchies": [{
                            "HierarchyId": "h1",
                            "Columns": [
                                {"DataSetIdentifier": "other_ds", "ColumnName": "col_x"},
                            ],
                        }],
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()


# ---------------------------------------------------------------------------
# Conditional formatting extraction
# ---------------------------------------------------------------------------


class TestExtractConditionalFormatColumns:
    """Tests for _extract_conditional_format_columns."""

    def test_extracts_column_from_expression(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {"FieldWells": {}},
                        "ConditionalFormatting": {
                            "ConditionalFormattingOptions": [{
                                "Cell": {
                                    "FieldId": "f1",
                                    "TextFormat": {
                                        "BackgroundColor": {
                                            "Solid": {
                                                "Expression": "IF({ds_1.score} > 90, 'green', 'red')",
                                            }
                                        }
                                    },
                                }
                            }]
                        },
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert "score" in result

    def test_extracts_column_from_icon_set_expression(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {"FieldWells": {}},
                        "ConditionalFormatting": {
                            "ConditionalFormattingOptions": [{
                                "Cell": {
                                    "FieldId": "f1",
                                    "TextFormat": {
                                        "Icon": {
                                            "IconSet": {
                                                "Expression": "IF({ds_1.growth} > 0, 'UP', 'DOWN')",
                                            }
                                        }
                                    },
                                }
                            }]
                        },
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert "growth" in result

    def test_extracts_column_ref_in_conditional_format_column_object(self, scanner: ColumnScanner):
        """Conditional formatting may also have Column objects directly."""
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {"FieldWells": {}},
                        "ConditionalFormatting": {
                            "ConditionalFormattingOptions": [{
                                "Row": {
                                    "Column": {
                                        "DataSetIdentifier": "ds_1",
                                        "ColumnName": "status",
                                    },
                                    "Expression": "TRUE",
                                }
                            }]
                        },
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert "status" in result


# ---------------------------------------------------------------------------
# RLS/CLS extraction
# ---------------------------------------------------------------------------


class TestExtractRlsClsColumns:
    """Tests for _extract_rls_cls_columns."""

    def test_extracts_column_level_permission_columns(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ColumnLevelPermissionRules": [{
                "Principals": ["arn:aws:quicksight:us-east-1:123:user/default/admin"],
                "ColumnNames": ["salary", "ssn", "bonus"],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"salary", "ssn", "bonus"}

    def test_extracts_row_level_permission_tag_columns(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "RowLevelPermissionTagConfiguration": {
                "TagRules": [
                    {"ColumnName": "department", "TagKey": "dept"},
                    {"ColumnName": "office_location", "TagKey": "office"},
                ],
            },
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"department", "office_location"}

    def test_multiple_cls_rules_combined(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ColumnLevelPermissionRules": [
                {"Principals": ["user1"], "ColumnNames": ["col_a", "col_b"]},
                {"Principals": ["user2"], "ColumnNames": ["col_b", "col_c"]},
            ],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"col_a", "col_b", "col_c"}


# ---------------------------------------------------------------------------
# Deduplication and combined scenarios
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for output deduplication (Req 2.8)."""

    def test_same_column_in_multiple_locations_deduped(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "TableVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {
                            "FieldWells": {
                                "TableAggregatedFieldWells": {
                                    "GroupBy": [{
                                        "CategoricalDimensionField": {
                                            "FieldId": "d1",
                                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "sales"},
                                        }
                                    }],
                                    "Values": [{
                                        "NumericalMeasureField": {
                                            "FieldId": "m1",
                                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "sales"},
                                        }
                                    }],
                                }
                            },
                            "Tooltip": {
                                "FieldBasedTooltip": {
                                    "TooltipFields": [{
                                        "FieldTooltipItem": {
                                            "FieldId": "tt1",
                                            "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "sales"},
                                        }
                                    }]
                                }
                            },
                        }
                    }
                }],
            }],
            "FilterGroups": [{
                "FilterGroupId": "fg1",
                "Filters": [{
                    "NumericRangeFilter": {
                        "FilterId": "f1",
                        "Column": {"DataSetIdentifier": "ds_1", "ColumnName": "sales"},
                    }
                }],
            }],
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "calc",
                "Expression": "sum({ds_1.sales})",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"sales"}

    def test_empty_definition_returns_empty_set(self, scanner: ColumnScanner):
        result = scanner.scan_definition({}, "ds_1", "arn:test")
        assert result == set()

    def test_sheets_with_no_visuals_returns_empty(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{"SheetId": "s1", "Visuals": []}],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error resilience."""

    def test_missing_sheets_key(self, scanner: ColumnScanner):
        definition = {
            "CalculatedFields": [{
                "DataSetIdentifier": "ds_1",
                "Name": "calc",
                "Expression": "sum({ds_1.amount})",
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == {"amount"}

    def test_visual_with_empty_chart_configuration(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [{
                "SheetId": "s1",
                "Visuals": [{
                    "CustomContentVisual": {
                        "VisualId": "v1",
                        "ChartConfiguration": {},
                    }
                }],
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()

    def test_filter_group_with_empty_filters(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "FilterGroups": [{"FilterGroupId": "fg1", "Filters": []}],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()

    def test_parameter_with_no_dynamic_value(self, scanner: ColumnScanner):
        definition = {
            "Sheets": [],
            "ParameterDeclarations": [{
                "DecimalParameterDeclaration": {
                    "Name": "threshold",
                    "DefaultValues": {"StaticValues": [0.5]},
                }
            }],
        }
        result = scanner.scan_definition(definition, "ds_1", "arn:test")
        assert result == set()
