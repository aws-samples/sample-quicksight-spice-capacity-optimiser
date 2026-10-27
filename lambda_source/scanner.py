# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Column reference scanner for QuickSight consumer definitions.

Extracts all column references from analysis and dashboard definitions,
scoped to a specific DataSetIdentifier. Handles visual field wells,
calculated fields, filters, parameters, drill-down hierarchies,
conditional formatting, and row/column-level security configurations.

The scanner deduplicates references into a single set per dataset and
conservatively includes all columns when a calculated field expression
cannot be parsed.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex to extract column references from calculated field expressions.
# Matches tokens like {DataSetIdentifier.ColumnName}
_CALC_FIELD_PATTERN = re.compile(r"\{([^}]+)\.([^}]+)\}")


class ColumnScanner:
    """Extracts column references from QuickSight consumer definitions.

    The scanner walks the definition tree looking for Column objects with
    a matching DataSetIdentifier. It collects references from:
    - Visual field wells (dimensions, measures, groupings, tooltip fields)
    - Calculated field expressions ({DataSetIdentifier.ColumnName} tokens)
    - Filter configurations (all filter types)
    - Parameter dynamic default column bindings
    - Drill-down hierarchies (visual/sheet-level)
    - Conditional formatting rules
    - Row-level security and column-level security configurations
    """

    def __init__(self) -> None:
        """Initialize the ColumnScanner."""
        pass

    def scan_definition(
        self,
        definition: dict,
        dataset_identifier: str,
        consumer_arn: str,
    ) -> set[str]:
        """Extract all column names referenced for the given dataset_identifier.

        This is the main entry point. It delegates to specialized extraction
        methods for each location type, then deduplicates results into a
        single set.

        Args:
            definition: The analysis or dashboard definition dict from
                DescribeAnalysisDefinition or DescribeDashboardDefinition.
            dataset_identifier: The DataSetIdentifier to scope extraction to.
            consumer_arn: ARN of the consumer (for logging context).

        Returns:
            A set of unique column names referenced for the given dataset.
        """
        columns: set[str] = set()

        sheets = definition.get("Sheets", [])
        calc_fields = definition.get("CalculatedFields", [])
        filter_groups = definition.get("FilterGroups", [])
        params = definition.get("ParameterDeclarations", [])

        # Extract from visual field wells, tooltips, etc.
        columns.update(self._extract_visual_columns(sheets, dataset_identifier))

        # Extract from calculated field expressions
        columns.update(
            self._extract_calculated_field_columns(calc_fields, dataset_identifier)
        )

        # Extract from filters
        filters = self._flatten_filters(filter_groups)
        columns.update(self._extract_filter_columns(filters, dataset_identifier))

        # Extract from parameter declarations
        columns.update(self._extract_parameter_columns(params, dataset_identifier))

        # Extract from drill-down hierarchies (sheet-level and definition-level)
        columns.update(self._extract_drill_down_columns(sheets, dataset_identifier))

        # Also check definition-level ColumnHierarchies
        definition_hierarchies = definition.get("ColumnHierarchies", [])
        if definition_hierarchies:
            columns.update(
                self._extract_hierarchy_columns(definition_hierarchies, dataset_identifier)
            )

        # Extract from conditional formatting
        columns.update(
            self._extract_conditional_format_columns(sheets, dataset_identifier)
        )

        # Extract from RLS/CLS (these are at the dataset definition level,
        # but may appear in the consumer definition context as well)
        columns.update(self._extract_rls_cls_columns(definition))

        logger.info(
            "Column scan complete",
            extra={
                "consumer_arn": consumer_arn,
                "dataset_identifier": dataset_identifier,
                "columns_found": len(columns),
            },
        )

        return columns

    def _extract_visual_columns(
        self, sheets: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from visual field wells.

        Walks all visuals in all sheets, extracting columns from dimensions,
        measures, groupings, tooltip fields, and any other field wells.
        Also extracts columns from sheet-level filter controls.

        Args:
            sheets: List of sheet definitions.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names found in visual field wells and filter controls.
        """
        columns: set[str] = set()

        for sheet in sheets:
            visuals = sheet.get("Visuals", [])
            for visual_wrapper in visuals:
                # Each visual is wrapped in a type key like "TableVisual", "BarChartVisual", etc.
                for visual_type, visual_def in visual_wrapper.items():
                    if not isinstance(visual_def, dict):
                        continue
                    # Extract from ChartConfiguration.FieldWells
                    chart_config = visual_def.get("ChartConfiguration", {})
                    field_wells = chart_config.get("FieldWells", {})
                    columns.update(
                        self._extract_columns_from_field_wells(field_wells, ds_id)
                    )

                    # Extract from Tooltip fields
                    tooltip = chart_config.get("Tooltip", {})
                    columns.update(
                        self._extract_columns_from_tooltip(tooltip, ds_id)
                    )

                    # Extract from SortConfiguration if present
                    sort_config = chart_config.get("SortConfiguration", {})
                    columns.update(
                        self._walk_for_columns(sort_config, ds_id)
                    )

            # Extract from sheet-level FilterControls
            filter_controls = sheet.get("FilterControls", [])
            columns.update(self._walk_for_columns(filter_controls, ds_id))

        return columns

    def _extract_columns_from_field_wells(
        self, field_wells: dict, ds_id: str
    ) -> set[str]:
        """Recursively extract Column references from a field wells structure.

        Field wells can contain various aggregated field well types like
        TableAggregatedFieldWells, BarChartAggregatedFieldWells, etc.
        Each contains lists of dimension/measure field objects with Column refs.
        """
        return self._walk_for_columns(field_wells, ds_id)

    def _extract_columns_from_tooltip(
        self, tooltip: dict, ds_id: str
    ) -> set[str]:
        """Extract columns from tooltip configuration."""
        return self._walk_for_columns(tooltip, ds_id)

    def _extract_calculated_field_columns(
        self, calc_fields: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from calculated field expressions.

        Parses expressions containing {DataSetIdentifier.ColumnName} tokens.
        If an expression cannot be parsed (no matches found but expression
        is non-empty), logs a warning. The caller should handle the conservative
        fallback of including all columns.

        Args:
            calc_fields: List of calculated field definitions.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names found in calculated field expressions.
            Returns a special sentinel if unparseable expression encountered.
        """
        columns: set[str] = set()

        for calc_field in calc_fields:
            # Only process calculated fields scoped to our dataset
            field_ds_id = calc_field.get("DataSetIdentifier", "")
            if field_ds_id != ds_id:
                continue

            expression = calc_field.get("Expression", "")
            if not expression:
                continue

            extracted = self._parse_expression_columns(expression, ds_id)
            if extracted is None:
                # Unparseable expression - log warning
                field_name = calc_field.get("Name", "unknown")
                logger.warning(
                    "Cannot parse calculated field expression, "
                    "conservatively including all columns",
                    extra={
                        "field_name": field_name,
                        "dataset_identifier": ds_id,
                        "expression_preview": expression[:100],
                    },
                )
                # Signal that we should include all columns
                columns.add("__UNPARSEABLE__")
            else:
                columns.update(extracted)

        return columns

    def _parse_expression_columns(
        self, expression: str, ds_id: str
    ) -> set[str] | None:
        """Parse a calculated field expression for column references.

        Extracts column names from {DataSetIdentifier.ColumnName} tokens
        where the DataSetIdentifier matches ds_id.

        Args:
            expression: The calculated field expression string.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names found, or None if the expression appears
            to contain column references that cannot be parsed.
        """
        matches = _CALC_FIELD_PATTERN.findall(expression)

        if not matches:
            # If expression contains curly braces but we couldn't match,
            # it might be an unparseable format
            if "{" in expression and "}" in expression:
                return None
            # Simple expression with no column references (e.g., literal calc)
            return set()

        columns: set[str] = set()
        for match_ds_id, col_name in matches:
            if match_ds_id == ds_id:
                columns.add(col_name)

        return columns

    def _extract_filter_columns(
        self, filters: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from filter configurations.

        Handles all filter types: CategoryFilter, NumericRangeFilter,
        NumericEqualityFilter, TimeRangeFilter, RelativeDateTimeFilter,
        TopBottomFilter.

        Args:
            filters: List of filter wrapper dicts (each has a filter type key).
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names referenced in filters.
        """
        columns: set[str] = set()

        for filter_wrapper in filters:
            # Each filter is wrapped in a type key
            for filter_type, filter_def in filter_wrapper.items():
                if not isinstance(filter_def, dict):
                    continue
                # All filter types have a Column with DataSetIdentifier
                columns.update(
                    self._walk_for_columns(filter_def, ds_id)
                )

        return columns

    def _extract_parameter_columns(
        self, params: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from parameter declarations.

        Looks for DefaultValues.DynamicValue.DefaultValueColumn,
        UserNameColumn, and GroupNameColumn.

        Args:
            params: List of parameter declaration wrapper dicts.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names referenced in parameter defaults.
        """
        columns: set[str] = set()

        for param_wrapper in params:
            # Each parameter is wrapped in a type key
            for param_type, param_def in param_wrapper.items():
                if not isinstance(param_def, dict):
                    continue
                # Walk the parameter definition looking for Column references
                columns.update(self._walk_for_columns(param_def, ds_id))

        return columns

    def _extract_drill_down_columns(
        self, sheets: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from drill-down hierarchies.

        Looks for ColumnHierarchies at the sheet level and within visuals.

        Args:
            sheets: List of sheet definitions.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names in drill-down hierarchies.
        """
        columns: set[str] = set()

        for sheet in sheets:
            # Sheet-level ColumnHierarchies
            hierarchies = sheet.get("ColumnHierarchies", [])
            columns.update(self._extract_hierarchy_columns(hierarchies, ds_id))

            # Visual-level ColumnHierarchies
            visuals = sheet.get("Visuals", [])
            for visual_wrapper in visuals:
                for visual_type, visual_def in visual_wrapper.items():
                    if not isinstance(visual_def, dict):
                        continue
                    vis_hierarchies = visual_def.get("ColumnHierarchies", [])
                    columns.update(
                        self._extract_hierarchy_columns(vis_hierarchies, ds_id)
                    )

        return columns

    def _extract_hierarchy_columns(
        self, hierarchies: list[dict], ds_id: str
    ) -> set[str]:
        """Extract columns from a list of hierarchy definitions.

        Each hierarchy has a Columns list with DataSetIdentifier and ColumnName.
        """
        columns: set[str] = set()

        for hierarchy in hierarchies:
            hierarchy_columns = hierarchy.get("Columns", [])
            for col_ref in hierarchy_columns:
                if isinstance(col_ref, dict):
                    if col_ref.get("DataSetIdentifier") == ds_id:
                        col_name = col_ref.get("ColumnName", "")
                        if col_name:
                            columns.add(col_name)

        return columns

    def _extract_conditional_format_columns(
        self, sheets: list[dict], ds_id: str
    ) -> set[str]:
        """Extract column references from conditional formatting rules.

        Looks for ConditionalFormatting in visuals, which contains
        ConditionalFormattingOptions with expressions referencing columns.

        Args:
            sheets: List of sheet definitions.
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names in conditional formatting rules.
        """
        columns: set[str] = set()

        for sheet in sheets:
            visuals = sheet.get("Visuals", [])
            for visual_wrapper in visuals:
                for visual_type, visual_def in visual_wrapper.items():
                    if not isinstance(visual_def, dict):
                        continue
                    cond_formatting = visual_def.get("ConditionalFormatting", {})
                    if not cond_formatting:
                        continue

                    options = cond_formatting.get(
                        "ConditionalFormattingOptions", []
                    )
                    for option in options:
                        # Walk the option tree for Column refs and expressions
                        columns.update(
                            self._walk_for_columns(option, ds_id)
                        )
                        # Also parse any expression strings for column refs
                        columns.update(
                            self._extract_expression_columns_from_tree(
                                option, ds_id
                            )
                        )

        return columns

    def _extract_expression_columns_from_tree(
        self, obj: Any, ds_id: str
    ) -> set[str]:
        """Recursively find and parse expression strings in a nested structure.

        Looks for string values containing {DataSetIdentifier.ColumnName}
        patterns within any "Expression" key.
        """
        columns: set[str] = set()

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "Expression" and isinstance(value, str):
                    parsed = self._parse_expression_columns(value, ds_id)
                    if parsed is not None:
                        columns.update(parsed)
                else:
                    columns.update(
                        self._extract_expression_columns_from_tree(value, ds_id)
                    )
        elif isinstance(obj, list):
            for item in obj:
                columns.update(
                    self._extract_expression_columns_from_tree(item, ds_id)
                )

        return columns

    def _extract_rls_cls_columns(self, definition: dict) -> set[str]:
        """Extract column references from RLS and CLS configurations.

        Row-Level Security: RowLevelPermissionDataSet may reference columns.
        Column-Level Security: ColumnLevelPermissionRules list ColumnNames.

        Note: RLS/CLS config is typically at the dataset level, but we check
        in the consumer definition context as well for completeness.

        Args:
            definition: The definition dict which may contain RLS/CLS config.

        Returns:
            Set of column names from RLS/CLS configurations.
        """
        columns: set[str] = set()

        # Column-Level Security: ColumnLevelPermissionRules
        cls_rules = definition.get("ColumnLevelPermissionRules", [])
        for rule in cls_rules:
            if isinstance(rule, dict):
                col_names = rule.get("ColumnNames", [])
                for name in col_names:
                    if isinstance(name, str) and name:
                        columns.add(name)

        # Row-Level Security: RowLevelPermissionTagConfiguration
        rls_tag_config = definition.get("RowLevelPermissionTagConfiguration", {})
        if rls_tag_config:
            tag_rules = rls_tag_config.get("TagRules", [])
            for rule in tag_rules:
                if isinstance(rule, dict):
                    col_name = rule.get("ColumnName", "")
                    if col_name:
                        columns.add(col_name)

        return columns

    def _walk_for_columns(self, obj: Any, ds_id: str) -> set[str]:
        """Recursively walk a nested dict/list structure looking for Column references.

        A Column reference is a dict with keys "DataSetIdentifier" and "ColumnName".
        Only includes columns where DataSetIdentifier matches ds_id.

        Args:
            obj: The object to walk (dict, list, or primitive).
            ds_id: The DataSetIdentifier to match.

        Returns:
            Set of column names found.
        """
        columns: set[str] = set()

        if isinstance(obj, dict):
            # Check if this dict IS a column reference
            if (
                "DataSetIdentifier" in obj
                and "ColumnName" in obj
                and obj.get("DataSetIdentifier") == ds_id
            ):
                col_name = obj.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

            # Also recurse into children
            for value in obj.values():
                columns.update(self._walk_for_columns(value, ds_id))

        elif isinstance(obj, list):
            for item in obj:
                columns.update(self._walk_for_columns(item, ds_id))

        return columns

    def _flatten_filters(self, filter_groups: list[dict]) -> list[dict]:
        """Flatten FilterGroups into a flat list of filter wrapper dicts.

        FilterGroups contain a "Filters" key with the actual filter objects.
        """
        filters: list[dict] = []
        for group in filter_groups:
            group_filters = group.get("Filters", [])
            filters.extend(group_filters)
        return filters
