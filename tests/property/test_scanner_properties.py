# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the ColumnScanner module.

Validates that the scanner correctly extracts all column references from
supported locations in QuickSight analysis/dashboard definitions.
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from lambda_source.scanner import ColumnScanner
from tests.property.strategies import (
    st_analysis_definition,
    st_column_name,
    st_dataset_identifier,
)


# Helper regex matching the scanner's own pattern for calculated field expressions
_CALC_FIELD_PATTERN = re.compile(r"\{([^}]+)\.([^}]+)\}")


def _extract_columns_from_structure(obj, ds_id: str) -> set[str]:
    """Independent column extraction by walking the definition tree.

    This provides a ground-truth reference for what columns exist in a
    generated definition. It looks for Column dicts with matching
    DataSetIdentifier and also parses {ds_id.col} tokens in expressions.
    """
    columns: set[str] = set()

    if isinstance(obj, dict):
        # Check if this dict is a Column reference
        if (
            "DataSetIdentifier" in obj
            and "ColumnName" in obj
            and obj.get("DataSetIdentifier") == ds_id
        ):
            col_name = obj.get("ColumnName", "")
            if col_name:
                columns.add(col_name)

        # Check for Expression strings containing {ds_id.col} references
        if "Expression" in obj and isinstance(obj["Expression"], str):
            expression = obj["Expression"]
            for match_ds_id, col_name in _CALC_FIELD_PATTERN.findall(expression):
                if match_ds_id == ds_id:
                    columns.add(col_name)

        # Recurse into all values
        for value in obj.values():
            columns.update(_extract_columns_from_structure(value, ds_id))

    elif isinstance(obj, list):
        for item in obj:
            columns.update(_extract_columns_from_structure(item, ds_id))

    return columns


def _extract_rls_cls_columns(definition: dict) -> set[str]:
    """Extract columns from RLS/CLS configurations independently.

    Mirrors the scanner's logic for ColumnLevelPermissionRules and
    RowLevelPermissionTagConfiguration.
    """
    columns: set[str] = set()

    cls_rules = definition.get("ColumnLevelPermissionRules", [])
    for rule in cls_rules:
        if isinstance(rule, dict):
            for name in rule.get("ColumnNames", []):
                if isinstance(name, str) and name:
                    columns.add(name)

    rls_config = definition.get("RowLevelPermissionTagConfiguration", {})
    if rls_config:
        for rule in rls_config.get("TagRules", []):
            if isinstance(rule, dict):
                col_name = rule.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

    return columns


class TestScannerExtractionCompleteness:
    """Property 3: Scanner extracts all column references from supported locations.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

    For any consumer definition containing column references (in visual field
    wells, calculated fields, filters, parameters, drill-down hierarchies,
    conditional formatting, or RLS/CLS configurations) scoped to a target
    dataset identifier, the scanner SHALL return a set containing every column
    name from those locations that belongs to the target dataset, and no columns
    from other datasets.
    """

    @given(
        dataset_identifier=st_dataset_identifier(),
        column_names=st.lists(
            st_column_name(),
            min_size=3,
            max_size=10,
            unique=True,
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_scanner_finds_all_columns_from_target_dataset(
        self,
        dataset_identifier: str,
        column_names: list[str],
        data,
    ):
        """The scanner must find every column placed in the definition for the target dataset."""
        # Generate a definition with columns placed in various locations
        definition = data.draw(
            st_analysis_definition(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            )
        )

        # Run the scanner
        scanner = ColumnScanner()
        result = scanner.scan_definition(
            definition=definition,
            dataset_identifier=dataset_identifier,
            consumer_arn="arn:aws:quicksight:us-east-1:123456789012:analysis/test",
        )

        # Remove the __UNPARSEABLE__ sentinel if present (it's an internal signal)
        result.discard("__UNPARSEABLE__")

        # Independently extract all columns that ARE in the definition
        expected_columns = _extract_columns_from_structure(
            definition, dataset_identifier
        )
        # Also include RLS/CLS columns (which are not scoped by DataSetIdentifier)
        expected_columns.update(_extract_rls_cls_columns(definition))

        # Property: scanner result must contain ALL columns placed in the definition
        missing = expected_columns - result
        assert missing == set(), (
            f"Scanner missed columns: {missing}. "
            f"Expected: {expected_columns}, Got: {result}"
        )

    @given(
        dataset_identifier=st_dataset_identifier(),
        column_names=st.lists(
            st_column_name(),
            min_size=3,
            max_size=10,
            unique=True,
        ),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_scanner_returns_no_columns_from_other_datasets(
        self,
        dataset_identifier: str,
        column_names: list[str],
        data,
    ):
        """The scanner must not return columns belonging to other datasets."""
        # Generate a definition - the strategy may include extra dataset declarations
        definition = data.draw(
            st_analysis_definition(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            )
        )

        # Run the scanner scoped to our target dataset
        scanner = ColumnScanner()
        result = scanner.scan_definition(
            definition=definition,
            dataset_identifier=dataset_identifier,
            consumer_arn="arn:aws:quicksight:us-east-1:123456789012:analysis/test",
        )

        # Remove the __UNPARSEABLE__ sentinel if present
        result.discard("__UNPARSEABLE__")

        # Property: all returned columns must be from the column_names list
        # (since st_analysis_definition only places columns from column_names
        # for the target dataset_identifier)
        # RLS/CLS columns are not scoped by DataSetIdentifier in the strategy,
        # but the scanner's _extract_rls_cls_columns also isn't scoped.
        # However, in the strategy, no RLS/CLS is generated at analysis level.
        # The scanner result should only contain columns that were placed.
        unexpected = result - set(column_names)
        assert unexpected == set(), (
            f"Scanner returned columns not from target dataset: {unexpected}. "
            f"Known columns: {column_names}, Got: {result}"
        )



class TestScannerOutputDeduplication:
    """Property 4: Scanner output deduplication.

    For any set of consumer definitions where the same column name appears
    N times (N >= 1) across different consumers or locations, the scanner
    output set SHALL contain that column name exactly once. The output set
    size SHALL be less than or equal to the total number of raw column
    references extracted.

    **Validates: Requirements 2.8**
    """

    @given(
        dataset_identifier=st_dataset_identifier(),
        shared_columns=st.lists(
            st_column_name(), min_size=1, max_size=5, unique=True
        ),
        num_visuals=st.integers(min_value=2, max_value=5),
        num_filters=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100)
    def test_duplicate_columns_across_locations_deduplicated(
        self,
        dataset_identifier: str,
        shared_columns: list[str],
        num_visuals: int,
        num_filters: int,
    ):
        """The scanner output set SHALL contain each column name exactly once,
        even when the same column appears in multiple visuals, filters, calc
        fields, parameters, and drill-down hierarchies.

        **Validates: Requirements 2.8**
        """
        # Build a definition that deliberately places the same columns in
        # multiple locations: visuals, filters, calculated fields, parameters,
        # and drill-down hierarchies.
        visuals = []
        for i in range(num_visuals):
            # Each visual references all shared_columns as dimensions
            dimensions = [
                {
                    "CategoricalDimensionField": {
                        "FieldId": f"dim_{i}_{j}",
                        "Column": {
                            "DataSetIdentifier": dataset_identifier,
                            "ColumnName": col,
                        },
                    }
                }
                for j, col in enumerate(shared_columns)
            ]
            visuals.append({
                "TableVisual": {
                    "VisualId": f"visual_{i}",
                    "ChartConfiguration": {
                        "FieldWells": {
                            "TableAggregatedFieldWells": {
                                "GroupBy": dimensions,
                                "Values": [],
                            }
                        }
                    },
                }
            })

        sheets = [{
            "SheetId": "sheet_1",
            "Name": "Sheet 1",
            "Visuals": visuals,
        }]

        # Add filters referencing the same columns
        filters = []
        for i in range(num_filters):
            for col in shared_columns:
                filters.append({
                    "CategoryFilter": {
                        "FilterId": f"filter_{i}_{col}",
                        "Column": {
                            "DataSetIdentifier": dataset_identifier,
                            "ColumnName": col,
                        },
                        "Configuration": {
                            "FilterListConfiguration": {
                                "MatchOperator": "CONTAINS",
                                "CategoryValues": ["test"],
                            }
                        },
                    }
                })

        filter_groups = [{
            "FilterGroupId": "fg_1",
            "Filters": filters,
            "ScopeConfiguration": {},
            "CrossDataset": "ALL_DATASETS",
        }]

        # Add calculated fields referencing the same columns
        calc_fields = [
            {
                "DataSetIdentifier": dataset_identifier,
                "Name": f"calc_{i}",
                "Expression": f"ifelse({{{dataset_identifier}.{col}}} > 0, 1, 0)",
            }
            for i, col in enumerate(shared_columns)
        ]

        # Add parameter declarations referencing the same columns
        params = [
            {
                "StringParameterDeclaration": {
                    "Name": f"param_{i}",
                    "ParameterValueType": "SINGLE_VALUED",
                    "DefaultValues": {
                        "DynamicValue": {
                            "DefaultValueColumn": {
                                "DataSetIdentifier": dataset_identifier,
                                "ColumnName": col,
                            },
                            "UserNameColumn": {
                                "DataSetIdentifier": dataset_identifier,
                                "ColumnName": col,
                            },
                            "GroupNameColumn": {
                                "DataSetIdentifier": dataset_identifier,
                                "ColumnName": col,
                            },
                        }
                    },
                }
            }
            for i, col in enumerate(shared_columns)
        ]

        # Build definition with all duplicated references
        definition = {
            "Sheets": sheets,
            "CalculatedFields": calc_fields,
            "FilterGroups": filter_groups,
            "ParameterDeclarations": params,
            "ColumnHierarchies": [
                {
                    "HierarchyId": "hier_1",
                    "Columns": [
                        {
                            "DataSetIdentifier": dataset_identifier,
                            "ColumnName": col,
                        }
                        for col in shared_columns
                    ],
                }
            ],
        }

        # Count total raw references (each column placed in multiple locations)
        total_raw_refs = (
            num_visuals * len(shared_columns)       # visuals
            + num_filters * len(shared_columns)     # filters
            + len(shared_columns)                   # calc fields
            + len(shared_columns) * 3               # params (3 column refs each)
            + len(shared_columns)                   # hierarchies
        )

        scanner = ColumnScanner()
        result = scanner.scan_definition(
            definition=definition,
            dataset_identifier=dataset_identifier,
            consumer_arn="arn:aws:quicksight:us-east-1:123456789012:analysis/test",
        )

        # Assert the output is a proper set (each column appears exactly once)
        assert isinstance(result, set), "Scanner output should be a set"
        for col in shared_columns:
            assert col in result, (
                f"Column '{col}' should be in the result set"
            )

        # Assert deduplication: result size <= total raw references
        assert len(result) <= total_raw_refs, (
            f"Result set size ({len(result)}) should be <= "
            f"total raw refs ({total_raw_refs})"
        )

        # Each shared column should appear exactly once in the result
        result_list_check = [col for col in shared_columns if col in result]
        assert len(result_list_check) == len(shared_columns), (
            "Each shared column should be present exactly once"
        )

    @given(
        dataset_identifier=st_dataset_identifier(),
        column_name=st_column_name(),
        repetitions=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_single_column_repeated_n_times_appears_once(
        self,
        dataset_identifier: str,
        column_name: str,
        repetitions: int,
    ):
        """A single column name appearing N times (N >= 2) across different
        visuals SHALL result in exactly one entry in the output set.

        **Validates: Requirements 2.8**
        """
        # Create N visuals each referencing the same single column
        visuals = []
        for i in range(repetitions):
            visuals.append({
                "BarChartVisual": {
                    "VisualId": f"visual_{i}",
                    "ChartConfiguration": {
                        "FieldWells": {
                            "BarChartAggregatedFieldWells": {
                                "Category": [
                                    {
                                        "CategoricalDimensionField": {
                                            "FieldId": f"cat_{i}",
                                            "Column": {
                                                "DataSetIdentifier": dataset_identifier,
                                                "ColumnName": column_name,
                                            },
                                        }
                                    }
                                ],
                                "Values": [],
                            }
                        }
                    },
                }
            })

        definition = {
            "Sheets": [{
                "SheetId": "sheet_1",
                "Name": "Test Sheet",
                "Visuals": visuals,
            }],
        }

        scanner = ColumnScanner()
        result = scanner.scan_definition(
            definition=definition,
            dataset_identifier=dataset_identifier,
            consumer_arn="arn:aws:quicksight:us-east-1:123456789012:analysis/test",
        )

        # The column should appear exactly once
        assert column_name in result
        # Output size should be 1 (only one unique column was referenced)
        assert len(result) == 1, (
            f"Expected 1 unique column, got {len(result)}: {result}"
        )
        # Size is definitely <= the number of raw references
        assert len(result) <= repetitions

    @given(
        definition=st_analysis_definition(min_columns=3, max_columns=10),
    )
    @settings(max_examples=100)
    def test_output_is_always_a_set_with_size_leq_raw_refs(
        self,
        definition: dict,
    ):
        """For any generated analysis definition, the scanner output SHALL
        be a Python set (guaranteeing uniqueness) with size <= total raw
        column references in the definition.

        **Validates: Requirements 2.8**
        """
        # Extract the dataset identifier from the definition
        ds_declarations = definition.get("DataSetIdentifierDeclarations", [])
        if not ds_declarations:
            return  # Skip if no dataset declarations

        dataset_identifier = ds_declarations[0]["Identifier"]

        scanner = ColumnScanner()
        result = scanner.scan_definition(
            definition=definition,
            dataset_identifier=dataset_identifier,
            consumer_arn="arn:aws:quicksight:us-east-1:123456789012:analysis/test",
        )

        # Remove internal sentinel
        result.discard("__UNPARSEABLE__")

        # Result must be a set (deduplication property)
        assert isinstance(result, set), "Scanner output must be a set"

        # Count all raw column references in the definition
        raw_refs = self._count_raw_column_refs(definition, dataset_identifier)

        # The output size must be <= total raw references
        if raw_refs > 0:
            assert len(result) <= raw_refs, (
                f"Result set size ({len(result)}) should be <= "
                f"raw refs ({raw_refs})"
            )

    def _count_raw_column_refs(self, obj, ds_id: str) -> int:
        """Count total raw column references in a nested structure."""
        count = 0
        if isinstance(obj, dict):
            if (
                "DataSetIdentifier" in obj
                and "ColumnName" in obj
                and obj.get("DataSetIdentifier") == ds_id
            ):
                count += 1
            # Also count expression-based column refs
            if "Expression" in obj and isinstance(obj["Expression"], str):
                for match_ds_id, _ in _CALC_FIELD_PATTERN.findall(obj["Expression"]):
                    if match_ds_id == ds_id:
                        count += 1
            for value in obj.values():
                count += self._count_raw_column_refs(value, ds_id)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_raw_column_refs(item, ds_id)
        return count
