# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Usage analyzer for QuickSight SPICE dataset columns.

Classifies every column in a dataset as used, unused, or join_key based on:
- Consumer column references (from analyses/dashboards)
- Join key columns (from LogicalTableMap JoinInstructions)
- Dataset-level calculated field input columns

Also provides safety threshold evaluation to prevent over-aggressive pruning.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lambda_source.attribution import UsageAttribution
from lambda_source.models import ColumnClassification, ColumnStatus, ColumnUsageSource
from lambda_source.on_clause import (
    join_keys_from_data_prep,
    join_keys_from_dataset,
    join_keys_from_logical_table_map,
    parse_on_clause,
)

logger = logging.getLogger(__name__)

# Regex to extract column references from calculated field expressions.
# Matches tokens like {DataSetIdentifier.ColumnName} or {[ColumnName]}
_CALC_FIELD_PATTERN = re.compile(r"\{([^}]+)\.([^}]+)\}")

# Alternate pattern for dataset-level calc fields that use {[column]} syntax
_BRACKET_COL_PATTERN = re.compile(r"\{\[([^\]]+)\]\}")


class UsageAnalyzer:
    """Analyzes column usage and classifies columns as used, unused, or join_key.

    The analyzer combines information from multiple sources:
    - Consumer references (extracted by ColumnScanner)
    - Join key columns (extracted from LogicalTableMap JoinInstructions)
    - Dataset-level calculated field inputs

    It produces a classification for every column in the dataset and evaluates
    whether the removal percentage is within a configurable safety threshold.
    """

    def __init__(self) -> None:
        """Initialize the UsageAnalyzer."""
        pass

    def extract_join_keys(self, logical_table_map: dict) -> set[str]:
        """Extract all columns referenced in JoinInstruction OnClauses.

        Parses JoinInstruction structures to find columns used as join keys.
        Checks:
        - OnClause strings (e.g., "col_a = col_b") to extract both column names
        - LeftJoinKeyProperties for column lists
        - RightJoinKeyProperties for column lists

        Args:
            logical_table_map: The LogicalTableMap dict from a dataset definition.

        Returns:
            Set of column names that serve as join keys.
        """
        join_keys: set[str] = set()

        if not logical_table_map:
            return join_keys

        for table_id, table_def in logical_table_map.items():
            if not isinstance(table_def, dict):
                continue

            source = table_def.get("Source", {})
            if not isinstance(source, dict):
                continue

            join_instruction = source.get("JoinInstruction")
            if not join_instruction or not isinstance(join_instruction, dict):
                continue

            # Parse OnClause: "col_a = col_b"
            on_clause = join_instruction.get("OnClause", "")
            if on_clause and isinstance(on_clause, str):
                join_keys.update(self._parse_on_clause(on_clause))

            # Check LeftJoinKeyProperties for column lists
            left_props = join_instruction.get("LeftJoinKeyProperties", {})
            if isinstance(left_props, dict):
                left_columns = left_props.get("Columns", [])
                if isinstance(left_columns, list):
                    for col in left_columns:
                        if isinstance(col, str) and col:
                            join_keys.add(col)

            # Check RightJoinKeyProperties for column lists
            right_props = join_instruction.get("RightJoinKeyProperties", {})
            if isinstance(right_props, dict):
                right_columns = right_props.get("Columns", [])
                if isinstance(right_columns, list):
                    for col in right_columns:
                        if isinstance(col, str) and col:
                            join_keys.add(col)

        return join_keys

    def _parse_on_clause(self, on_clause: str) -> set[str]:
        """Parse an OnClause string to extract column names.

        Delegates to the shared parser in ``lambda_source.on_clause`` so this and
        ``ChildDatasetDiscovery`` cannot disagree about the format again.

        Handles:
        - "col_a = col_b" and "col_a=col_b"
        - "{col_a} = {col_b}" (brace syntax)
        - "{col[table]}" (table-qualified; yields both readings)
        - Multi-condition clauses separated by AND

        Previously this kept the raw text, so a braced clause produced
        "{inventory_id}" — matching no entry in OutputColumns. The join key was then
        classified unused and became a removal candidate.

        Args:
            on_clause: The OnClause string from a JoinInstruction.

        Returns:
            Set of column names found in the clause.
        """
        return parse_on_clause(on_clause)

    def extract_all_join_keys(
        self, dataset_def: dict, known_columns: set[str] | None = None
    ) -> set[str]:
        """Extract join keys from a dataset regardless of preparation experience.

        ``extract_join_keys`` reads only ``LogicalTableMap``, so a new-experience
        dataset — which keeps joins in ``DataPrepConfiguration.TransformStepMap`` —
        received no join key protection whatsoever. This covers both shapes.

        Args:
            dataset_def: Full ``DescribeDataSet`` response.
            known_columns: When supplied, results are intersected with it so only
                real columns are returned.

        Returns:
            Set of column names that must not be removed.
        """
        return join_keys_from_dataset(dataset_def, known_columns)

    def extract_join_keys_by_source(
        self, dataset_def: dict, known_columns: set[str] | None = None
    ) -> dict[str, set[str]]:
        """Same join keys as :meth:`extract_all_join_keys`, split by declaring shape.

        The report used to state "join key in LogicalTableMap JoinInstruction" for
        every protected join key. Once new-experience joins were covered, that
        string became actively wrong for any key declared in a
        ``DataPrepConfiguration.JoinStep`` — the report named a location the
        operator would not find it in.

        Args:
            dataset_def: Full ``DescribeDataSet`` response.
            known_columns: When supplied, each set is intersected with it.

        Returns:
            Mapping of :class:`ColumnUsageSource` value to column names. The union
            of the values equals :meth:`extract_all_join_keys` for the same inputs,
            because intersection distributes over union.
        """
        legacy = join_keys_from_logical_table_map(dataset_def.get("LogicalTableMap"))
        data_prep = join_keys_from_data_prep(dataset_def.get("DataPrepConfiguration"))

        if known_columns is not None:
            legacy = {name for name in legacy if name in known_columns}
            data_prep = {name for name in data_prep if name in known_columns}

        return {
            ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value: legacy,
            ColumnUsageSource.JOIN_KEY_DATA_PREP.value: data_prep,
        }

    def extract_dataset_calc_field_inputs(self, dataset_def: dict) -> set[str]:
        """Extract columns used as inputs to dataset-level calculated fields.

        Dataset-level calculated fields are defined within the dataset definition
        itself (not within consumer definitions). These reference columns using
        patterns like {DataSetIdentifier.ColumnName} or {[ColumnName]}.

        Columns that are inputs to dataset-level calculated fields are considered
        used, even if no consumer directly references the raw column.

        Args:
            dataset_def: The full dataset definition dict from DescribeDataSet.

        Returns:
            Set of column names used as inputs to dataset-level calculated fields.
        """
        columns: set[str] = set()

        calc_fields = dataset_def.get("CalculatedFields", [])
        if not calc_fields:
            return columns

        dataset_id = dataset_def.get("DataSetId", "")

        for calc_field in calc_fields:
            if not isinstance(calc_field, dict):
                continue

            expression = calc_field.get("Expression", "")
            if not expression:
                continue

            # Try {DataSetIdentifier.ColumnName} pattern
            matches = _CALC_FIELD_PATTERN.findall(expression)
            for match_ds_id, col_name in matches:
                # Include columns from this dataset
                # Dataset-level calc fields may use the DataSetId as identifier
                if match_ds_id == dataset_id or not dataset_id:
                    columns.add(col_name)
                else:
                    # Also include if the identifier isn't a known other dataset
                    # (conservative approach for dataset-level calc fields)
                    columns.add(col_name)

            # Try {[ColumnName]} pattern (common in dataset-level calc fields)
            bracket_matches = _BRACKET_COL_PATTERN.findall(expression)
            for col_name in bracket_matches:
                if col_name:
                    columns.add(col_name)

        return columns

    def classify_columns(
        self,
        all_columns: list[dict],
        used_columns: set[str],
        join_key_columns: set[str],
        calc_field_input_columns: set[str],
        attribution: UsageAttribution | None = None,
    ) -> list[ColumnClassification]:
        """Classify every column in the dataset.

        Classification logic:
        - A column is "join_key" if it's in join_key_columns (regardless of other refs)
        - A column is "used" if it's in used_columns OR calc_field_input_columns
        - Otherwise, the column is "unused"

        Join keys are treated as a special case of "used" but get their own
        status to clearly identify them in reports.

        Args:
            all_columns: List of column dicts with "Name" and "Type" keys
                (from OutputColumns in the dataset definition).
            used_columns: Set of column names referenced by consumers.
            join_key_columns: Set of column names that are join keys.
            calc_field_input_columns: Set of column names used as inputs
                to dataset-level calculated fields.
            attribution: Optional per-column provenance. When supplied, ``reasons``
                names the specific consumers and child datasets requiring each
                column and ``attributions`` carries the structured form. When
                omitted, the coarse legacy reason strings are used. Note that this
                argument never affects ``status`` — only how it is explained.

        Returns:
            List of ColumnClassification for every column in the dataset.
        """
        classifications: list[ColumnClassification] = []

        for col_def in all_columns:
            col_name = col_def.get("Name", "")
            data_type = col_def.get("Type", "STRING")

            reasons: list[str] = []
            attributions = (
                attribution.for_column(col_name) if attribution is not None else []
            )
            attributed_reasons = (
                attribution.reasons_for(col_name) if attribution is not None else []
            )

            # Determine classification status. Unchanged by attribution — the same
            # three flat sets decide it, exactly as before.
            if col_name in join_key_columns:
                status = ColumnStatus.JOIN_KEY.value
                if attributed_reasons:
                    reasons.extend(attributed_reasons)
                else:
                    reasons.append("join key in LogicalTableMap JoinInstruction")
                    # Also note if it has consumer references
                    if col_name in used_columns:
                        reasons.append("referenced by consumer")
                    if col_name in calc_field_input_columns:
                        reasons.append("input to dataset-level calculated field")
            elif col_name in used_columns:
                status = ColumnStatus.USED.value
                if attributed_reasons:
                    reasons.extend(attributed_reasons)
                else:
                    reasons.append("referenced by consumer")
                    if col_name in calc_field_input_columns:
                        reasons.append("input to dataset-level calculated field")
            elif col_name in calc_field_input_columns:
                status = ColumnStatus.USED.value
                reasons.extend(
                    attributed_reasons
                    or ["input to dataset-level calculated field"]
                )
            else:
                status = ColumnStatus.UNUSED.value
                # Deliberately the legacy string even under attribution: an unused
                # column has no requesters, so there is nothing richer to say.
                reasons.append("no consumer references")

            classifications.append(
                ColumnClassification(
                    column_name=col_name,
                    data_type=data_type,
                    status=status,
                    reasons=reasons,
                    attributions=attributions,
                )
            )

        return classifications

    def get_unused_columns(
        self, classifications: list[ColumnClassification]
    ) -> list[ColumnClassification]:
        """Return only columns classified as unused.

        Args:
            classifications: Full list of column classifications.

        Returns:
            Filtered list containing only unused columns.
        """
        return [
            c for c in classifications
            if c.status == ColumnStatus.UNUSED.value
        ]

    def check_safety_threshold(
        self, classifications: list[ColumnClassification], threshold: float
    ) -> bool:
        """Check if the removal percentage is within the safety threshold.

        Returns True if it is safe to proceed (within threshold).
        Returns False if the threshold is breached (too many unused columns).

        The threshold represents the maximum allowable fraction of columns
        that can be removed. If unused_count / total_count > threshold,
        the threshold is breached.

        Edge cases:
        - If total_count is 0, returns True (nothing to remove).
        - If threshold is 0.0, any unused columns breach the threshold.
        - If threshold is 1.0, the threshold is never breached.

        Args:
            classifications: Full list of column classifications.
            threshold: Maximum fraction (0.0–1.0) of columns removable.

        Returns:
            True if within threshold (safe to proceed).
            False if breached (unsafe, too many columns would be removed).
        """
        total_count = len(classifications)

        if total_count == 0:
            return True

        unused_count = sum(
            1 for c in classifications
            if c.status == ColumnStatus.UNUSED.value
        )

        removal_ratio = unused_count / total_count

        if removal_ratio > threshold:
            logger.warning(
                "Safety threshold breached",
                extra={
                    "unused_count": unused_count,
                    "total_count": total_count,
                    "removal_ratio": removal_ratio,
                    "threshold": threshold,
                },
            )
            return False

        return True
