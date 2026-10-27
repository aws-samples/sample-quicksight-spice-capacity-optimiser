# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Child dataset discovery for the SPICE Dataset Optimiser.

Discovers datasets that use a target dataset as a data source (parent-child
relationship) recursively at all depths. Extracts which columns from the parent
dataset are referenced by each child, and scans each child's consumers
(dashboards/analyses) to determine which parent columns are actually used
end-to-end through the dependency chain.

Handles both:
- Legacy datasets: LogicalTableMap[*].Source.DataSetArn
- New experience datasets: DataPrepConfiguration.SourceTableMap[*].DataSet.DataSetArn
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lambda_source.scanner import ColumnScanner

from lambda_source.on_clause import strip_braces
from lambda_source.qs_client import (
    QSClient,
    QSClientError,
    AccessDeniedError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)

# Regex to extract column references from calculated field expressions.
# Matches tokens like {DataSetIdentifier.ColumnName} or {[ColumnName]}
_CALC_FIELD_PATTERN = re.compile(r"\{([^}]+)\.([^}]+)\}")
_BRACKET_COL_PATTERN = re.compile(r"\{\[([^\]]+)\]\}")

# Default maximum recursion depth for child discovery
DEFAULT_MAX_DEPTH = 10


@dataclass
class ChildDataset:
    """A dataset that uses the target dataset as a source."""

    dataset_id: str
    """ID of the child dataset."""

    name: str
    """Display name of the child dataset."""

    referenced_columns: set[str] = field(default_factory=set)
    """Column names from the parent that this child references."""

    depth: int = 1
    """Depth in the dependency tree (1 = direct child, 2 = grandchild, etc.)."""

    consumer_used_columns: set[str] = field(default_factory=set)
    """Columns from the original target that are actually used by this child's
    consumers (dashboards/analyses). Populated by scan_child_consumers()."""

    # The three fields below decompose the protection this child provides. The
    # union of them is what the pipeline acts on, but the report needs them apart:
    # a parent column held only by a child's ProjectStep can be freed by editing
    # the child, whereas one held by the child's join key cannot. Collapsing them
    # made every child-protected column look equally immovable.

    consumer_traced_columns: set[str] = field(default_factory=set)
    """Parent columns proven to reach a visual in one of this child's consumers."""

    join_key_columns: set[str] = field(default_factory=set)
    """Parent columns this child joins on. Not removable without restructuring."""

    projected_columns: set[str] = field(default_factory=set)
    """Parent columns carried by this child's ProjectStep/RenameColumnsStep/etc."""

    structural_fallback: bool = False
    """True when consumer tracing could not run for this child and every imported
    column was retained instead. The columns are protected, but unproven."""


class ChildDiscoveryError(Exception):
    """Raised when child dataset discovery encounters a blocking error."""

    def __init__(self, message: str, unreachable_datasets: list[str] | None = None):
        super().__init__(message)
        self.unreachable_datasets = unreachable_datasets or []


class ChildDatasetDiscovery:
    """Discovers datasets that reference a target dataset as a source.

    Recursively scans all datasets in the account to find the full dependency
    tree rooted at the target dataset. For each child found (at any depth),
    extracts the set of parent columns it depends on. Optionally scans each
    child's consumers (dashboards/analyses) to trace which parent columns are
    actually used end-to-end.
    """

    def __init__(self, qs_client: QSClient, account_id: str, region: str):
        """Initialize ChildDatasetDiscovery.

        Args:
            qs_client: Pre-configured QSClient instance for API calls.
            account_id: AWS account ID for constructing dataset ARNs.
            region: AWS region for constructing dataset ARNs.
        """
        self._qs_client = qs_client
        self._account_id = account_id
        self._region = region
        # Cache of dataset_id -> dataset_def (populated during discovery)
        self._all_datasets_listed: list[dict] | None = None

    def discover_child_datasets(
        self,
        target_dataset_id: str,
        *,
        skip_unreachable: bool = False,
        max_depth: int = DEFAULT_MAX_DEPTH,
        unreachable_out: list[str] | None = None,
    ) -> list[ChildDataset]:
        """Find all datasets that use target_dataset_id as a source, recursively.

        Discovers the full dependency tree: direct children, grandchildren, and
        so on up to max_depth. Uses cycle detection to handle circular references.

        Carries the same defect ``ConsumerDiscovery`` had, for the same reason: on the
        ``skip_unreachable=True`` path the internal ``unreachable`` list was logged and
        then discarded, so a child dataset that could not be described was
        indistinguishable from one that does not exist. Its columns would then look
        unreferenced and become removal candidates. ``unreachable_out`` gives the caller
        the list either way.

        Args:
            target_dataset_id: The dataset ID to find children for.
            skip_unreachable: If True, skip datasets that can't be described
                instead of raising. Unreachable children are logged as warnings.
            max_depth: Maximum recursion depth (default 10). Prevents runaway
                traversal in pathological cases.
            unreachable_out: Optional accumulator receiving the IDs of every dataset
                that could not be described, whether or not the run halts.

        Returns:
            List of ChildDataset objects with referenced_columns populated,
            including children at all depths.

        Raises:
            ChildDiscoveryError: If any child dataset can't be described and
                skip_unreachable is False.
        """
        # List all datasets once (cached for the entire recursive traversal)
        if self._all_datasets_listed is None:
            try:
                self._all_datasets_listed = self._qs_client.list_datasets()
            except QSClientError as exc:
                logger.error(
                    "Failed to list datasets for child discovery",
                    extra={"error": str(exc), "error_code": exc.error_code},
                )
                raise ChildDiscoveryError(
                    f"Failed to list datasets: {exc}"
                ) from exc

        # Recursive traversal with cycle detection
        all_children: list[ChildDataset] = []
        visited: set[str] = {target_dataset_id}  # Track visited to detect cycles
        unreachable: list[str] = []

        self._discover_recursive(
            target_dataset_id=target_dataset_id,
            all_children=all_children,
            visited=visited,
            unreachable=unreachable,
            skip_unreachable=skip_unreachable,
            current_depth=0,
            max_depth=max_depth,
        )

        # Surface to the caller before deciding whether to halt, so the accumulator is
        # usable even when the caller catches ChildDiscoveryError.
        if unreachable_out is not None:
            unreachable_out.extend(unreachable)

        # Handle unreachable datasets
        if unreachable:
            if skip_unreachable:
                logger.warning(
                    "Skipping %d unreachable dataset(s) during child discovery",
                    len(unreachable),
                    extra={
                        "unreachable_count": len(unreachable),
                        "unreachable_ids": unreachable,
                    },
                )
            else:
                raise ChildDiscoveryError(
                    f"Cannot proceed: {len(unreachable)} dataset(s) unreachable "
                    f"during child discovery. IDs: {unreachable}",
                    unreachable_datasets=unreachable,
                )

        logger.info(
            "Recursive child dataset discovery complete",
            extra={
                "target_dataset_id": target_dataset_id,
                "children_found": len(all_children),
                "max_depth_reached": max(
                    (c.depth for c in all_children), default=0
                ),
                "total_datasets_scanned": len(self._all_datasets_listed),
            },
        )

        return all_children

    def _discover_recursive(
        self,
        target_dataset_id: str,
        all_children: list[ChildDataset],
        visited: set[str],
        unreachable: list[str],
        skip_unreachable: bool,
        current_depth: int,
        max_depth: int,
    ) -> None:
        """Recursively discover children of a dataset.

        Args:
            target_dataset_id: Dataset to find children of at this level.
            all_children: Accumulator list for all discovered children.
            visited: Set of already-visited dataset IDs (cycle detection).
            unreachable: Accumulator list for unreachable dataset IDs.
            skip_unreachable: Whether to skip or raise on unreachable datasets.
            current_depth: Current recursion depth.
            max_depth: Maximum allowed recursion depth.
        """
        if current_depth >= max_depth:
            logger.warning(
                "Max depth reached during recursive child discovery",
                extra={
                    "target_dataset_id": target_dataset_id,
                    "max_depth": max_depth,
                },
            )
            return

        target_arn_suffix = f"dataset/{target_dataset_id}"
        target_arn = (
            f"arn:aws:quicksight:{self._region}:{self._account_id}:"
            f"dataset/{target_dataset_id}"
        )

        for dataset_summary in self._all_datasets_listed:
            ds_id = dataset_summary.get("DataSetId", "")

            # Skip if no ID, is the target itself, or already visited (cycle)
            if not ds_id or ds_id == target_dataset_id or ds_id in visited:
                continue

            # Describe the dataset (uses cache from QSClient)
            try:
                dataset_def = self._qs_client.describe_dataset(ds_id)
            except ResourceNotFoundError:
                logger.debug(
                    "Dataset not found during child discovery, skipping",
                    extra={"dataset_id": ds_id},
                )
                continue
            except AccessDeniedError as exc:
                logger.warning(
                    "Access denied describing dataset during child discovery",
                    extra={"dataset_id": ds_id, "error": str(exc)},
                )
                unreachable.append(ds_id)
                continue
            except QSClientError as exc:
                logger.warning(
                    "Error describing dataset during child discovery",
                    extra={
                        "dataset_id": ds_id,
                        "error_code": exc.error_code,
                        "error": str(exc),
                    },
                )
                unreachable.append(ds_id)
                continue

            # Check if this dataset references the target
            is_child = self._references_target(dataset_def, target_arn, target_arn_suffix)

            if is_child:
                ds_name = dataset_def.get("Name", ds_id)
                referenced_cols = self._extract_referenced_columns(
                    dataset_def, target_arn, target_arn_suffix
                )

                child = ChildDataset(
                    dataset_id=ds_id,
                    name=ds_name,
                    referenced_columns=referenced_cols,
                    depth=current_depth + 1,
                )
                all_children.append(child)

                logger.info(
                    "Found child dataset",
                    extra={
                        "child_dataset_id": ds_id,
                        "child_name": ds_name,
                        "target_dataset_id": target_dataset_id,
                        "depth": current_depth + 1,
                        "referenced_columns_count": len(referenced_cols),
                    },
                )

                # Mark as visited and recurse into this child's children
                visited.add(ds_id)
                self._discover_recursive(
                    target_dataset_id=ds_id,
                    all_children=all_children,
                    visited=visited,
                    unreachable=unreachable,
                    skip_unreachable=skip_unreachable,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                )

    def scan_child_consumers(
        self,
        children: list[ChildDataset],
        target_dataset_id: str,
        scanner: "ColumnScanner",
    ) -> set[str]:
        """Scan consumers of each child dataset to find which parent columns are used.

        For each child, discovers its consumers (dashboards/analyses), scans
        their definitions for column references, then traces back which of those
        columns originate from the target parent dataset.

        This provides a more precise answer than just looking at the child's
        structural dependencies — it tells us which parent columns are actually
        displayed in a visual somewhere in the dependency chain.

        Args:
            children: List of discovered ChildDataset objects.
            target_dataset_id: The original target dataset being optimised.
            scanner: ColumnScanner instance for extracting column refs from consumers.

        Returns:
            Set of column names from the target dataset that are actually used
            by consumers of child datasets (at any depth).
        """
        from lambda_source.discovery import ConsumerDiscovery

        consumer_discovery = ConsumerDiscovery(
            qs_client=self._qs_client,
            account_id=self._account_id,
        )

        target_columns_used: set[str] = set()

        for child in children:
            try:
                # Discover consumers of this child dataset
                consumers = consumer_discovery.discover_consumers(
                    child.dataset_id, skip_unreachable=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed to discover consumers for child dataset, "
                    "falling back to structural dependencies",
                    extra={
                        "child_dataset_id": child.dataset_id,
                        "error": str(exc),
                    },
                )
                # Fallback: use structural dependencies (conservative)
                target_columns_used.update(child.referenced_columns)
                child.structural_fallback = True
                continue

            if not consumers:
                # No consumers — the child's structural dependencies are what matter
                target_columns_used.update(child.referenced_columns)
                child.structural_fallback = True
                continue

            # Scan each consumer to find columns used from this child
            child_used_columns: set[str] = set()
            for consumer in consumers:
                try:
                    if consumer.consumer_type == "analysis":
                        analysis_id = consumer.arn.split("/")[-1]
                        definition = self._qs_client.describe_analysis_definition(
                            analysis_id
                        )
                    else:
                        dashboard_id = consumer.arn.split("/")[-1]
                        definition = self._qs_client.describe_dashboard_definition(
                            dashboard_id
                        )

                    columns = scanner.scan_definition(
                        definition=definition,
                        dataset_identifier=consumer.dataset_identifier,
                        consumer_arn=consumer.arn,
                    )
                    columns.discard("__UNPARSEABLE__")
                    child_used_columns.update(columns)

                except Exception as exc:
                    logger.warning(
                        "Failed to scan consumer of child dataset",
                        extra={
                            "child_dataset_id": child.dataset_id,
                            "consumer_arn": consumer.arn,
                            "error": str(exc),
                        },
                    )
                    # Conservative: assume all child structural deps are needed
                    child_used_columns.update(child.referenced_columns)
                    child.structural_fallback = True
                    break

            # Now trace which of the child's used columns originate from the parent.
            # The child's referenced_columns tells us what comes from the parent.
            # Intersection = parent columns that flow through AND are used downstream.
            if child_used_columns:
                # Get the child's output columns to understand the column mapping
                child_def = self._qs_client.describe_dataset(child.dataset_id)
                parent_columns_in_child = child.referenced_columns

                # Columns used by child's consumers that also come from the parent
                traced_to_parent = child_used_columns & parent_columns_in_child
                target_columns_used.update(traced_to_parent)

                # Also protect columns that the child uses structurally even if
                # the consumer doesn't directly reference them (e.g., join keys
                # within the child that enable the consumer's columns)
                child_logical_map = child_def.get("LogicalTableMap") or {}
                child_join_keys = self._extract_all_join_keys(child_logical_map)

                # Also extract join keys from DataPrepConfiguration (new experience)
                child_data_prep = child_def.get("DataPrepConfiguration") or {}
                child_join_keys.update(
                    self._extract_data_prep_join_keys(child_data_prep)
                )

                structural_join_keys = child_join_keys & parent_columns_in_child
                target_columns_used.update(structural_join_keys)

                # Protect columns referenced in the child's ProjectStep/RenameColumnsStep
                # that originate from the parent. After a join, QuickSight suffixes
                # columns with .1/.2 — strip those to match parent names.
                child_projected_cols = self._extract_data_prep_projected_columns(
                    child_data_prep
                )
                matched_projected: set[str] = set()
                for col in child_projected_cols:
                    # Strip .1, .2 suffixes added by join disambiguation
                    base_col = col.rsplit(".", 1)[0] if re.match(r".*\.\d+$", col) else col
                    if base_col in parent_columns_in_child:
                        target_columns_used.add(base_col)
                        matched_projected.add(base_col)

                child.consumer_used_columns = traced_to_parent | structural_join_keys
                # Kept apart from the union above purely for reporting; the
                # pipeline still acts on target_columns_used.
                child.consumer_traced_columns = traced_to_parent
                child.join_key_columns = structural_join_keys
                child.projected_columns = matched_projected
            else:
                # No columns found in consumers — use structural deps
                target_columns_used.update(child.referenced_columns)
                child.consumer_used_columns = child.referenced_columns
                child.structural_fallback = True

            logger.info(
                "Child consumer scan complete",
                extra={
                    "child_dataset_id": child.dataset_id,
                    "child_consumers_count": len(consumers),
                    "child_used_columns_count": len(child_used_columns),
                    "traced_to_parent_count": len(child.consumer_used_columns),
                },
            )

        return target_columns_used

    def _extract_all_join_keys(self, logical_table_map: dict) -> set[str]:
        """Extract all join key column names from a LogicalTableMap.

        Args:
            logical_table_map: The LogicalTableMap from a dataset definition.

        Returns:
            Set of column names used as join keys.
        """
        join_keys: set[str] = set()
        for _table_id, table_def in logical_table_map.items():
            if not isinstance(table_def, dict):
                continue
            source = table_def.get("Source", {})
            if not isinstance(source, dict):
                continue
            join_instruction = source.get("JoinInstruction")
            if not join_instruction:
                continue
            on_clause = join_instruction.get("OnClause", "")
            if on_clause:
                conditions = re.split(r"\s+AND\s+", on_clause, flags=re.IGNORECASE)
                for condition in conditions:
                    parts = condition.split("=")
                    for part in parts:
                        col = self._strip_on_clause_braces(part.strip())
                        if col:
                            join_keys.add(col)
        return join_keys

    def _extract_data_prep_join_keys(self, data_prep_config: dict) -> set[str]:
        """Extract join key column names from DataPrepConfiguration TransformStepMap.

        New experience datasets store joins in TransformStepMap.JoinStep instead
        of LogicalTableMap.JoinInstruction. This method extracts column names
        from JoinStep.OnClause.

        Args:
            data_prep_config: The DataPrepConfiguration dict from a dataset.

        Returns:
            Set of column names used as join keys in JoinSteps.
        """
        join_keys: set[str] = set()
        transform_step_map = data_prep_config.get("TransformStepMap") or {}

        for _step_id, step in transform_step_map.items():
            if not isinstance(step, dict):
                continue
            join_step = step.get("JoinStep")
            if not join_step:
                continue
            on_clause = join_step.get("OnClause", "")
            if on_clause:
                conditions = re.split(r"\s+AND\s+", on_clause, flags=re.IGNORECASE)
                for condition in conditions:
                    parts = condition.split("=")
                    for part in parts:
                        col = self._strip_on_clause_braces(part.strip())
                        if col:
                            join_keys.add(col)

        return join_keys

    def _extract_data_prep_projected_columns(self, data_prep_config: dict) -> set[str]:
        """Extract all column names referenced by any step in DataPrepConfiguration.

        Covers all TransformStepMap step types:
        - ProjectStep: ProjectedColumns
        - RenameColumnsStep: RenameColumnOperations[].ColumnName
        - CastColumnTypesStep: CastColumnTypeOperations[].ColumnName
        - CreateColumnsStep: Columns[].Expression (parsed for column refs)
        - AggregateStep: GroupByColumnNames + Aggregations[].SimpleAggregation/ListAggregation.InputColumnName
        - FiltersStep: FilterOperations[].ConditionExpression + column conditions
        - JoinStep: OnClause (handled separately by _extract_data_prep_join_keys)
        - PivotStep: GroupByColumnNames + PivotConfiguration.LabelColumnName + ValueColumnConfiguration
        - AppendStep: AppendedColumns[].ColumnName
        - UnpivotStep: ColumnsToUnpivot[].ColumnName

        Args:
            data_prep_config: The DataPrepConfiguration dict from a dataset.

        Returns:
            Set of column names found across all transform steps.
        """
        columns: set[str] = set()
        transform_step_map = data_prep_config.get("TransformStepMap") or {}

        for _step_id, step in transform_step_map.items():
            if not isinstance(step, dict):
                continue

            # ProjectStep — columns that are selected/kept
            project_step = step.get("ProjectStep")
            if project_step:
                projected = project_step.get("ProjectedColumns", [])
                for col in projected:
                    if isinstance(col, str) and col:
                        columns.add(col)

            # RenameColumnsStep — source column names being renamed
            rename_step = step.get("RenameColumnsStep")
            if rename_step:
                renames = rename_step.get("RenameColumnOperations", [])
                for rename in renames:
                    col_name = rename.get("ColumnName", "")
                    if col_name:
                        columns.add(col_name)

            # CastColumnTypesStep — columns being type-cast
            cast_step = step.get("CastColumnTypesStep")
            if cast_step:
                cast_ops = cast_step.get("CastColumnTypeOperations", [])
                for cast_op in cast_ops:
                    col_name = cast_op.get("ColumnName", "")
                    if col_name:
                        columns.add(col_name)

            # CreateColumnsStep — parse expressions for column refs
            create_step = step.get("CreateColumnsStep")
            if create_step:
                calc_columns = create_step.get("Columns", [])
                for calc_col in calc_columns:
                    expression = calc_col.get("Expression", "")
                    if expression:
                        columns.update(self._parse_expression(expression))

            # AggregateStep — group-by columns and aggregation input columns
            aggregate_step = step.get("AggregateStep")
            if aggregate_step:
                group_by = aggregate_step.get("GroupByColumnNames", [])
                for col in group_by:
                    if isinstance(col, str) and col:
                        columns.add(col)
                aggregations = aggregate_step.get("Aggregations", [])
                for agg in aggregations:
                    agg_func = agg.get("AggregationFunction", {})
                    simple_agg = agg_func.get("SimpleAggregation", {})
                    if simple_agg:
                        input_col = simple_agg.get("InputColumnName", "")
                        if input_col:
                            columns.add(input_col)
                    list_agg = agg_func.get("ListAggregation", {})
                    if list_agg:
                        input_col = list_agg.get("InputColumnName", "")
                        if input_col:
                            columns.add(input_col)

            # FiltersStep — columns referenced in filter conditions
            filters_step = step.get("FiltersStep")
            if filters_step:
                filter_ops = filters_step.get("FilterOperations", [])
                for filter_op in filter_ops:
                    # ConditionExpression (legacy format)
                    cond_expr = filter_op.get("ConditionExpression", "")
                    if cond_expr:
                        columns.update(self._parse_expression(cond_expr))
                    # Typed filter conditions reference ColumnName directly
                    for filter_type in ("StringFilterCondition", "NumericFilterCondition", "DateFilterCondition"):
                        typed_filter = filter_op.get(filter_type, {})
                        if typed_filter:
                            col_name = typed_filter.get("ColumnName", "")
                            if col_name:
                                columns.add(col_name)

            # PivotStep — group-by + label column + value column
            pivot_step = step.get("PivotStep")
            if pivot_step:
                group_by = pivot_step.get("GroupByColumnNames", [])
                for col in group_by:
                    if isinstance(col, str) and col:
                        columns.add(col)
                pivot_config = pivot_step.get("PivotConfiguration", {})
                label_col = pivot_config.get("LabelColumnName", "")
                if label_col:
                    columns.add(label_col)
                value_config = pivot_step.get("ValueColumnConfiguration", {})
                agg_func = value_config.get("AggregationFunction", {})
                for agg_type in ("SimpleAggregation", "ListAggregation"):
                    agg = agg_func.get(agg_type, {})
                    if agg:
                        input_col = agg.get("InputColumnName", "")
                        if input_col:
                            columns.add(input_col)

            # AppendStep — columns being appended
            append_step = step.get("AppendStep")
            if append_step:
                appended = append_step.get("AppendedColumns", [])
                for app_col in appended:
                    col_name = app_col.get("ColumnName", "")
                    if col_name:
                        columns.add(col_name)

            # UnpivotStep — columns being unpivoted
            unpivot_step = step.get("UnpivotStep")
            if unpivot_step:
                cols_to_unpivot = unpivot_step.get("ColumnsToUnpivot", [])
                for unpivot_col in cols_to_unpivot:
                    col_name = unpivot_col.get("ColumnName", "")
                    if col_name:
                        columns.add(col_name)

        return columns

    def get_all_child_referenced_columns(
        self, children: list[ChildDataset]
    ) -> set[str]:
        """Aggregate all referenced columns across all child datasets.

        Convenience method that unions the referenced_columns from all children
        into a single set.

        Args:
            children: List of discovered ChildDataset objects.

        Returns:
            Set of all parent column names referenced by any child dataset.
        """
        all_refs: set[str] = set()
        for child in children:
            all_refs.update(child.referenced_columns)
        return all_refs

    def _references_target(
        self, dataset_def: dict, target_arn: str, target_arn_suffix: str
    ) -> bool:
        """Check if a dataset definition references the target dataset as a source.

        Checks both legacy LogicalTableMap and new experience DataPrepConfiguration.

        Args:
            dataset_def: The full dataset definition from DescribeDataSet.
            target_arn: Full ARN of the target dataset.
            target_arn_suffix: The 'dataset/{id}' suffix for flexible matching.

        Returns:
            True if this dataset uses the target as a source.
        """
        # Check LogicalTableMap (legacy)
        logical_table_map = dataset_def.get("LogicalTableMap") or {}
        for _table_id, table_def in logical_table_map.items():
            if not isinstance(table_def, dict):
                continue
            source = table_def.get("Source", {})
            if not isinstance(source, dict):
                continue
            ds_arn = source.get("DataSetArn", "")
            if self._arn_matches(ds_arn, target_arn, target_arn_suffix):
                return True

        # Check DataPrepConfiguration (new experience)
        data_prep = dataset_def.get("DataPrepConfiguration") or {}
        source_table_map = data_prep.get("SourceTableMap") or {}
        for _source_id, source_table in source_table_map.items():
            if not isinstance(source_table, dict):
                continue
            ds_source = source_table.get("DataSet", {})
            if not isinstance(ds_source, dict):
                continue
            ds_arn = ds_source.get("DataSetArn", "")
            if self._arn_matches(ds_arn, target_arn, target_arn_suffix):
                return True

        return False

    def _extract_referenced_columns(
        self, dataset_def: dict, target_arn: str, target_arn_suffix: str
    ) -> set[str]:
        """Extract all parent columns referenced by a child dataset.

        Gathers columns from multiple locations:
        1. DataPrepConfiguration.SourceTableMap InputColumns (explicit declaration)
        2. LogicalTableMap transforms (joins, casts, renames, filters, projections)
        3. Dataset-level calculated field expressions
        4. PhysicalTableMap references (if the child re-uses the parent's source)

        If no explicit column list is found (e.g., the child uses all columns
        from the parent), returns an empty set — the caller should treat this
        conservatively as "all parent columns are needed."

        Args:
            dataset_def: The child dataset's full definition.
            target_arn: Full ARN of the parent/target dataset.
            target_arn_suffix: The 'dataset/{id}' suffix for flexible matching.

        Returns:
            Set of column names from the parent that the child references.
            Empty set means "could not determine" (conservative: assume all needed).
        """
        columns: set[str] = set()
        found_explicit_declaration = False

        # 1. Check DataPrepConfiguration.SourceTableMap for explicit InputColumns
        data_prep = dataset_def.get("DataPrepConfiguration") or {}
        source_table_map = data_prep.get("SourceTableMap") or {}
        for _source_id, source_table in source_table_map.items():
            if not isinstance(source_table, dict):
                continue
            ds_source = source_table.get("DataSet", {})
            if not isinstance(ds_source, dict):
                continue
            ds_arn = ds_source.get("DataSetArn", "")
            if self._arn_matches(ds_arn, target_arn, target_arn_suffix):
                input_columns = ds_source.get("InputColumns", [])
                if input_columns:
                    found_explicit_declaration = True
                    for col in input_columns:
                        col_name = col.get("Name", "")
                        if col_name:
                            columns.add(col_name)

        # 2. Check LogicalTableMap for transforms referencing parent columns
        logical_table_map = dataset_def.get("LogicalTableMap") or {}
        for _table_id, table_def in logical_table_map.items():
            if not isinstance(table_def, dict):
                continue

            source = table_def.get("Source", {})
            if not isinstance(source, dict):
                continue

            # Only inspect logical tables sourced from the target dataset
            ds_arn = source.get("DataSetArn", "")
            if not self._arn_matches(ds_arn, target_arn, target_arn_suffix):
                # Also check if this logical table joins with one that sources
                # from the target — join keys on the target side matter
                join_instruction = source.get("JoinInstruction")
                if join_instruction:
                    columns.update(
                        self._extract_join_columns_for_target(
                            join_instruction, logical_table_map,
                            target_arn, target_arn_suffix
                        )
                    )
                continue

            found_explicit_declaration = True

            # Extract columns from DataTransforms in this logical table
            transforms = table_def.get("DataTransforms", [])
            columns.update(self._extract_transform_columns(transforms))

        # 3. Check dataset-level calculated fields for parent column references
        calc_fields = dataset_def.get("CalculatedFields", [])
        columns.update(self._extract_calc_field_columns(calc_fields))

        # If we found no explicit declaration of which columns are used,
        # return empty set to signal "unknown — assume all are needed"
        if not found_explicit_declaration and not columns:
            return set()

        return columns

    def _extract_join_columns_for_target(
        self,
        join_instruction: dict,
        logical_table_map: dict,
        target_arn: str,
        target_arn_suffix: str,
    ) -> set[str]:
        """Extract join key columns that come from the target dataset side.

        When a logical table joins two operands, and one of those operands
        sources from the target dataset, the join key columns on that side
        are parent column references.

        Args:
            join_instruction: The JoinInstruction dict from a logical table source.
            logical_table_map: The full LogicalTableMap for resolution.
            target_arn: Full ARN of the target dataset.
            target_arn_suffix: The 'dataset/{id}' suffix.

        Returns:
            Set of column names from the target used as join keys.
        """
        columns: set[str] = set()

        on_clause = join_instruction.get("OnClause", "")
        left_operand_id = join_instruction.get("LeftOperand", "")
        right_operand_id = join_instruction.get("RightOperand", "")

        # Determine which operand(s) source from the target
        left_is_target = self._logical_table_sources_target(
            left_operand_id, logical_table_map, target_arn, target_arn_suffix
        )
        right_is_target = self._logical_table_sources_target(
            right_operand_id, logical_table_map, target_arn, target_arn_suffix
        )

        if not on_clause:
            return columns

        # Parse OnClause: "col_a = col_b" or multi-condition with AND
        # QuickSight may wrap column names in braces: {col_name} or {col_name[table]}
        conditions = re.split(r"\s+AND\s+", on_clause, flags=re.IGNORECASE)
        for condition in conditions:
            parts = condition.split("=")
            if len(parts) == 2:
                left_col = self._strip_on_clause_braces(parts[0].strip())
                right_col = self._strip_on_clause_braces(parts[1].strip())

                # The left column belongs to the left operand, right to right
                if left_is_target and left_col:
                    columns.add(left_col)
                if right_is_target and right_col:
                    columns.add(right_col)
            else:
                # Can't parse — add all parts conservatively
                for part in parts:
                    col = self._strip_on_clause_braces(part.strip())
                    if col:
                        columns.add(col)

        return columns

    @staticmethod
    def _strip_on_clause_braces(col_ref: str) -> str:
        """Strip braces and table qualifiers from an OnClause column reference.

        QuickSight OnClause can use formats like:
        - "col_name" (plain)
        - "{col_name}" (braces)
        - "{col_name[table_alias]}" (braces with table qualifier)

        This extracts just the column name.

        Args:
            col_ref: Raw column reference from OnClause.

        Returns:
            Clean column name without braces or table qualifier.
        """
        # Delegates to the shared parser so format knowledge lives in one place.
        # This returns only the bare column name; use
        # lambda_source.on_clause.candidate_names when the table-qualified literal
        # (e.g. the real column "part_num[parts]") also needs protecting.
        return strip_braces(col_ref)

    def _logical_table_sources_target(
        self,
        logical_table_id: str,
        logical_table_map: dict,
        target_arn: str,
        target_arn_suffix: str,
    ) -> bool:
        """Check if a logical table ultimately sources from the target dataset.

        Args:
            logical_table_id: ID of the logical table to check.
            logical_table_map: The full LogicalTableMap.
            target_arn: Full ARN of the target.
            target_arn_suffix: Suffix for flexible matching.

        Returns:
            True if the logical table sources from the target dataset.
        """
        table_def = logical_table_map.get(logical_table_id, {})
        if not isinstance(table_def, dict):
            return False

        source = table_def.get("Source", {})
        if not isinstance(source, dict):
            return False

        ds_arn = source.get("DataSetArn", "")
        return self._arn_matches(ds_arn, target_arn, target_arn_suffix)

    def _extract_transform_columns(self, transforms: list[dict]) -> set[str]:
        """Extract all column names referenced in a list of DataTransforms.

        Handles: ProjectOperation, RenameColumnOperation, CastColumnTypeOperation,
        CreateColumnsOperation, FilterOperation, TagColumnOperation, UntagColumnOperation.

        Args:
            transforms: List of transform operation dicts.

        Returns:
            Set of column names found in the transforms.
        """
        columns: set[str] = set()

        for transform in transforms:
            if not isinstance(transform, dict):
                continue

            # ProjectOperation — projected columns are the ones the child uses
            project_op = transform.get("ProjectOperation", {})
            if project_op:
                projected = project_op.get("ProjectedColumns", [])
                for col in projected:
                    if isinstance(col, str) and col:
                        # Strip table qualifier: "col_name[table]" → "col_name"
                        clean_col = col.split("[")[0] if "[" in col else col
                        columns.add(clean_col)

            # RenameColumnOperation — the source column name
            rename_op = transform.get("RenameColumnOperation", {})
            if rename_op:
                col_name = rename_op.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

            # CastColumnTypeOperation — the column being cast
            cast_op = transform.get("CastColumnTypeOperation", {})
            if cast_op:
                col_name = cast_op.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

            # CreateColumnsOperation — parse expressions for column refs
            create_op = transform.get("CreateColumnsOperation", {})
            if create_op:
                calc_columns = create_op.get("Columns", [])
                for calc_col in calc_columns:
                    expression = calc_col.get("Expression", "")
                    if expression:
                        columns.update(self._parse_expression(expression))

            # FilterOperation — the column being filtered
            filter_op = transform.get("FilterOperation", {})
            if filter_op:
                cond_expr = filter_op.get("ConditionExpression", "")
                if cond_expr:
                    columns.update(self._parse_expression(cond_expr))

            # TagColumnOperation — column being tagged
            tag_op = transform.get("TagColumnOperation", {})
            if tag_op:
                col_name = tag_op.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

            # UntagColumnOperation — column being untagged
            untag_op = transform.get("UntagColumnOperation", {})
            if untag_op:
                col_name = untag_op.get("ColumnName", "")
                if col_name:
                    columns.add(col_name)

        return columns

    def _extract_calc_field_columns(self, calc_fields: list[dict]) -> set[str]:
        """Extract column names from dataset-level calculated field expressions.

        Args:
            calc_fields: List of calculated field definitions.

        Returns:
            Set of column names referenced in expressions.
        """
        columns: set[str] = set()

        for calc_field in calc_fields:
            if not isinstance(calc_field, dict):
                continue
            expression = calc_field.get("Expression", "")
            if expression:
                columns.update(self._parse_expression(expression))

        return columns

    def _parse_expression(self, expression: str) -> set[str]:
        """Parse a QuickSight expression string for column references.

        Handles both {DataSetIdentifier.ColumnName} and {[ColumnName]} patterns.

        Args:
            expression: The expression string to parse.

        Returns:
            Set of column names found in the expression.
        """
        columns: set[str] = set()

        # {DataSetIdentifier.ColumnName} pattern
        matches = _CALC_FIELD_PATTERN.findall(expression)
        for _ds_id, col_name in matches:
            if col_name:
                columns.add(col_name)

        # {[ColumnName]} pattern
        bracket_matches = _BRACKET_COL_PATTERN.findall(expression)
        for col_name in bracket_matches:
            if col_name:
                columns.add(col_name)

        return columns

    @staticmethod
    def _arn_matches(arn: str, target_arn: str, target_arn_suffix: str) -> bool:
        """Check if an ARN matches the target dataset.

        Uses both exact match and suffix match to handle cross-region or
        partition differences.

        Args:
            arn: The ARN to check.
            target_arn: The expected full ARN.
            target_arn_suffix: The 'dataset/{id}' suffix.

        Returns:
            True if the ARN refers to the target dataset.
        """
        if not arn:
            return False
        # Exact match first
        if arn == target_arn:
            return True
        # Suffix match: check that the ARN ends with dataset/{target_id}
        # and there's a colon or slash boundary before it
        if arn.endswith(target_arn_suffix):
            # Ensure it's a proper boundary (preceded by ':')
            prefix_end = len(arn) - len(target_arn_suffix)
            if prefix_end > 0 and arn[prefix_end - 1] == ":":
                return True
        return False
