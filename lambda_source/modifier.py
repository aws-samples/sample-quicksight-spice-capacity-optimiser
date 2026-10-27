# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Dataset modification via ProjectOperation for the SPICE Dataset Optimiser.

Provides functionality to build optimized dataset definitions by inserting
ProjectOperation entries into the LogicalTableMap to exclude unused columns.
ProjectOperation is placed at the end of the transformation chain (after
casts/renames/filters) to avoid breaking intermediate transforms that reference
excluded columns.

Also handles applying the modification via UpdateDataSet and triggering a
SPICE ingestion via CreateIngestion after a successful update.
"""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from lambda_source.qs_client import QSClient, QSClientError
from lambda_source.update_params import build_update_params

logger = logging.getLogger(__name__)


# QuickSight's wording when a dataset has no refreshable source (e.g. an uploaded
# file). Matched on the phrase rather than the error code because
# InvalidParameterValueException covers many unrelated problems. Kept in sync with
# handler._UNREFRESHABLE_MARKERS, which classifies the parent's own ingestion failure
# the same way; a test asserts the two agree so they cannot drift.
_UNREFRESHABLE_MARKERS = (
    "not refreshable because of the source type",
    "dataset is not accelerated",
)


def _refresh_unsupported(exc: BaseException) -> bool:
    """Whether this dataset simply has no refresh to start, not a failed one."""
    text = str(exc).lower()
    return any(marker in text for marker in _UNREFRESHABLE_MARKERS)


class ModificationError(Exception):
    """Raised when dataset modification fails."""

    def __init__(self, message: str, dataset_id: str | None = None, snapshot_key: str | None = None):
        super().__init__(message)
        self.dataset_id = dataset_id
        self.snapshot_key = snapshot_key


class ValidationError(Exception):
    """Raised when pre-modification validation fails."""

    def __init__(self, message: str, missing_columns: set[str] | None = None):
        super().__init__(message)
        self.missing_columns = missing_columns or set()


@dataclass
class ChildRefreshResult:
    """Outcome of attempting to refresh a single child dataset during a cascade."""

    dataset_id: str
    name: str = ""
    depth: int = 0
    refresh_started: bool = False
    """True when CreateIngestion was accepted for this child."""
    refresh_unsupported: bool = False
    """True when the child cannot be refreshed at all (e.g. an uploaded-file source).
    Not a failure — there is simply no ingestion to start."""
    ingestion_id: str | None = None
    error: str | None = None
    """Populated only on a genuine failure (not on refresh_unsupported)."""

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "depth": self.depth,
            "refresh_started": self.refresh_started,
            "refresh_unsupported": self.refresh_unsupported,
            "ingestion_id": self.ingestion_id,
            "error": self.error,
        }


class DatasetModifier:
    """Modifies dataset definitions by inserting ProjectOperation to exclude unused columns.

    Handles the full modification lifecycle:
    1. Build optimized definition (insert/update ProjectOperation)
    2. Validate retained columns are superset of consumer references
    3. Apply modification via UpdateDataSet API
    4. Trigger SPICE ingestion via CreateIngestion API
    """

    def __init__(self, qs_client: QSClient):
        """Initialize DatasetModifier.

        Args:
            qs_client: QSClient instance for QuickSight API calls.
        """
        self._qs_client = qs_client

    def build_optimized_definition(
        self,
        dataset_def: dict,
        unused_columns: list[str],
    ) -> dict:
        """Insert/update ProjectOperation to exclude unused columns.

        For each logical table in LogicalTableMap, determines which columns
        to keep (all columns minus unused) and inserts a ProjectOperation at
        position 0 of the DataTransforms list with ProjectedColumns set to
        the columns to keep.

        Args:
            dataset_def: The full dataset definition from DescribeDataSet.
            unused_columns: List of column names to exclude.

        Returns:
            A deep copy of the dataset definition with ProjectOperation inserted
            in each logical table's transformation chain.
        """
        if not unused_columns:
            logger.info("No unused columns to remove, returning definition unchanged")
            return copy.deepcopy(dataset_def)

        unused_set = set(unused_columns)
        modified_def = copy.deepcopy(dataset_def)

        logical_table_map = modified_def.get("LogicalTableMap", {})

        # Collect all columns from the dataset's OutputColumns or column schema
        all_columns = self._get_all_column_names(dataset_def)
        columns_to_keep = sorted([col for col in all_columns if col not in unused_set])

        if not columns_to_keep:
            logger.warning("All columns would be removed, skipping modification")
            return copy.deepcopy(dataset_def)

        if not logical_table_map:
            # New experience dataset: no LogicalTableMap available.
            # Use DataPrepConfiguration to add a ProjectStep after the final
            # transform, filtering to only retained columns.
            data_prep_config = modified_def.get("DataPrepConfiguration")
            if not data_prep_config:
                # Build a new DataPrepConfiguration with ProjectStep for new experience
                # datasets that have no existing DataPrepConfiguration.
                logger.info(
                    "New experience dataset has no DataPrepConfiguration, "
                    "building one with ProjectStep to exclude unused columns",
                    extra={"columns_to_keep": len(columns_to_keep)},
                )

                physical_table_map = modified_def.get("PhysicalTableMap", {})
                if not physical_table_map:
                    logger.warning("No PhysicalTableMap found, cannot modify dataset")
                    return copy.deepcopy(dataset_def)

                # For multi-table datasets without LogicalTableMap, we cannot safely
                # construct the join structure. Return unchanged with a warning.
                if len(physical_table_map) > 1:
                    logger.warning(
                        "Multi-table dataset without LogicalTableMap — join structure "
                        "unknown, skipping column removal. Dataset must be modified "
                        "manually via QuickSight console.",
                        extra={"physical_tables": len(physical_table_map)},
                    )
                    return copy.deepcopy(dataset_def)

                # Single physical table without LogicalTableMap: safe to add one
                table_id = list(physical_table_map.keys())[0]
                modified_def["LogicalTableMap"] = {
                    f"logical-{table_id[:8]}": {
                        "Alias": modified_def.get("Name", "Projected"),
                        "Source": {"PhysicalTableId": table_id},
                        "DataTransforms": [
                            {
                                "ProjectOperation": {
                                    "ProjectedColumns": columns_to_keep,
                                }
                            }
                        ],
                    }
                }

                return modified_def

            logger.info(
                "New experience dataset detected, inserting ProjectStep "
                "into DataPrepConfiguration",
                extra={"columns_to_keep": len(columns_to_keep)},
            )

            # Find the current DestinationTable's source (the final transform step)
            dest_map = data_prep_config.get("DestinationTableMap", {})
            transform_map = data_prep_config.get("TransformStepMap", {})

            # Build column ID lookup from OutputColumns
            output_columns = modified_def.get("OutputColumns", [])
            columns_to_keep_set = set(columns_to_keep)
            column_id_mappings = []
            for col in output_columns:
                if col.get("Name") in columns_to_keep_set and col.get("Id"):
                    column_id_mappings.append({
                        "SourceColumnId": col["Id"],
                        "TargetColumnId": col["Id"],
                    })

            for dest_id, dest_table in dest_map.items():
                current_source_id = dest_table.get("Source", {}).get("TransformOperationId")
                if not current_source_id:
                    continue

                # Insert a ProjectStep between the final transform and the destination
                project_step_id = "optimiser-project-step"
                transform_map[project_step_id] = {
                    "ProjectStep": {
                        "Alias": "SPICE Optimiser Column Projection",
                        "ProjectedColumns": columns_to_keep,
                        "Source": {
                            "TransformOperationId": current_source_id,
                            "ColumnIdMappings": column_id_mappings,
                        },
                    }
                }

                # Point the destination to our new ProjectStep
                dest_table["Source"]["TransformOperationId"] = project_step_id

            return modified_def

        # Legacy path: Insert ProjectOperation into each logical table
        for table_id, logical_table in logical_table_map.items():
            modified_table = self._insert_project_operation(logical_table, columns_to_keep)
            logical_table_map[table_id] = modified_table

        logger.info(
            "Built optimized definition",
            extra={
                "unused_columns_count": len(unused_columns),
                "retained_columns_count": len(columns_to_keep),
                "logical_tables_modified": len(logical_table_map),
            },
        )

        return modified_def

    def apply_modification(self, dataset_id: str, optimized_def: dict) -> dict:
        """Call UpdateDataSet with the modified definition.

        Handles both legacy datasets (LogicalTableMap) and new experience
        datasets (DataPrepConfiguration). Detects the dataset type based on
        whether LogicalTableMap is present in the definition.

        Args:
            dataset_id: QuickSight dataset identifier.
            optimized_def: The optimized dataset definition (from build_optimized_definition).

        Returns:
            The UpdateDataSet API response dict.

        Raises:
            ModificationError: If UpdateDataSet fails or dataset is unsupported
                (e.g., file-upload datasets have empty PhysicalTableMap).
        """
        # Guard: file-upload datasets have empty PhysicalTableMap entries and
        # cannot be modified via UpdateDataSet (documented API limitation).
        physical_table_map = optimized_def.get("PhysicalTableMap", {})
        if physical_table_map:
            has_empty_tables = any(
                not table_def for table_def in physical_table_map.values()
            )
            if has_empty_tables:
                logger.error(
                    "Dataset has empty PhysicalTableMap entries — likely a file-upload "
                    "dataset which cannot be modified via UpdateDataSet API",
                    extra={"dataset_id": dataset_id},
                )
                raise ModificationError(
                    f"Dataset {dataset_id} cannot be modified: PhysicalTableMap contains "
                    f"empty entries (file-upload datasets are not supported by UpdateDataSet)",
                    dataset_id=dataset_id,
                )

        update_params = self._build_update_params(optimized_def)

        logger.info(
            "Applying dataset modification",
            extra={
                "dataset_id": dataset_id,
                "uses_data_prep_config": "DataPrepConfiguration" in update_params,
            },
        )

        try:
            response = self._qs_client.update_dataset(dataset_id, update_params)
        except QSClientError as exc:
            logger.error(
                "UpdateDataSet failed",
                extra={
                    "dataset_id": dataset_id,
                    "error_code": exc.error_code,
                    "error": str(exc),
                },
            )
            raise ModificationError(
                f"UpdateDataSet failed for dataset {dataset_id}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        logger.info(
            "Dataset modification applied successfully",
            extra={"dataset_id": dataset_id},
        )

        return response

    def trigger_ingestion(self, dataset_id: str) -> dict:
        """Call CreateIngestion to refresh SPICE after modification.

        Generates a unique ingestion_id and triggers a new SPICE ingestion
        for the modified dataset.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The CreateIngestion API response dict.

        Raises:
            ModificationError: If CreateIngestion fails.
        """
        ingestion_id = f"optimiser-{uuid.uuid4().hex[:12]}"

        logger.info(
            "Triggering SPICE ingestion",
            extra={
                "dataset_id": dataset_id,
                "ingestion_id": ingestion_id,
            },
        )

        try:
            response = self._qs_client.create_ingestion(dataset_id, ingestion_id)
        except QSClientError as exc:
            logger.error(
                "CreateIngestion failed",
                extra={
                    "dataset_id": dataset_id,
                    "ingestion_id": ingestion_id,
                    "error_code": exc.error_code,
                    "error": str(exc),
                },
            )
            raise ModificationError(
                f"CreateIngestion failed for dataset {dataset_id}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        logger.info(
            "SPICE ingestion triggered successfully",
            extra={
                "dataset_id": dataset_id,
                "ingestion_id": ingestion_id,
            },
        )

        return response

    # QuickSight throttles CreateIngestion fairly aggressively, and a cascade fires one
    # per descendant with nothing between them. _call_with_retry backs off on a 429, but
    # only up to MAX_RETRIES before giving up on that child — so a short deliberate pause
    # between children keeps a wide tree from burning through those retries on the first
    # few and failing the rest. Small enough not to matter for typical trees of a handful
    # of children; a knob rather than a hardcode so a test can set it to zero.
    CASCADE_REFRESH_DELAY_SECONDS = 1.0

    def trigger_cascading_refresh(
        self,
        children: list[Any],
        *,
        delay_seconds: float | None = None,
    ) -> list[ChildRefreshResult]:
        """Trigger a SPICE ingestion for each descendant dataset, parents first.

        Called after a parent dataset has been modified and its own ingestion started.
        Child datasets still hold SPICE data shaped to the parent's old column set; once
        the parent refreshes, each child needs its own refresh so its cached data and
        any transforms re-resolve against the new parent schema.

        Order is by ascending ``depth`` (direct children before grandchildren), so each
        level refreshes only after the level above it. ``depth`` is the only ordering
        signal the discovery layer exposes, and level order is sufficient for the
        parent-before-child guarantee this needs.

        This never raises. The parent change is already committed by the time this runs,
        so a child that cannot be refreshed must be reported, not allowed to turn a
        successful optimisation into a failed one. Three per-child outcomes:

        - ``refresh_started`` — CreateIngestion accepted.
        - ``refresh_unsupported`` — the child has no refreshable source (an uploaded
          file). Not an error; QuickSight re-reads the stored file itself.
        - ``error`` — anything else (throttling exhausted, access denied, a transient
          server error). Recorded against that child; the cascade continues.

        Args:
            children: Discovered descendant datasets. Each needs ``dataset_id`` and,
                for ordering and reporting, ``depth`` and ``name``.
            delay_seconds: Pause between successive CreateIngestion calls. Defaults to
                ``CASCADE_REFRESH_DELAY_SECONDS``; pass ``0`` to disable (used by tests).

        Returns:
            One ``ChildRefreshResult`` per child, in the order they were refreshed.
        """
        if delay_seconds is None:
            delay_seconds = self.CASCADE_REFRESH_DELAY_SECONDS

        ordered = sorted(
            children or [],
            key=lambda c: getattr(c, "depth", 0),
        )

        results: list[ChildRefreshResult] = []
        for index, child in enumerate(ordered):
            dataset_id = getattr(child, "dataset_id", "")
            name = getattr(child, "name", "")
            depth = getattr(child, "depth", 0)

            if not dataset_id:
                continue

            # Space out calls to stay under the CreateIngestion rate limit. Skip the
            # pause before the first call and whenever disabled.
            if index > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)

            # Distinct prefix so a cascaded refresh is attributable to this tool and
            # distinguishable from the parent's own `optimiser-` ingestion and from a
            # rollback's `optimiser-rollback-` in the QuickSight ingestion history.
            ingestion_id = f"optimiser-cascade-{uuid.uuid4().hex[:12]}"
            result = ChildRefreshResult(
                dataset_id=dataset_id,
                name=name,
                depth=depth,
                ingestion_id=ingestion_id,
            )

            try:
                self._qs_client.create_ingestion(dataset_id, ingestion_id)
                result.refresh_started = True
                logger.info(
                    "Cascading SPICE refresh started for child dataset",
                    extra={
                        "dataset_id": dataset_id,
                        "ingestion_id": ingestion_id,
                        "depth": depth,
                        "operation": "cascade_refresh",
                    },
                )
            except Exception as exc:  # noqa: BLE001 — a child refresh must never raise
                if _refresh_unsupported(exc):
                    result.refresh_unsupported = True
                    result.ingestion_id = None
                    logger.info(
                        "Child dataset has no refreshable source; nothing to refresh",
                        extra={
                            "dataset_id": dataset_id,
                            "depth": depth,
                            "reason": str(exc),
                            "operation": "cascade_refresh",
                        },
                    )
                else:
                    result.error = f"{type(exc).__name__}: {exc}"
                    result.ingestion_id = None
                    logger.warning(
                        "Cascading SPICE refresh failed for child dataset "
                        "(parent change already applied)",
                        exc_info=True,
                        extra={
                            "dataset_id": dataset_id,
                            "depth": depth,
                            "error": result.error,
                            "operation": "cascade_refresh",
                        },
                    )

            results.append(result)

        logger.info(
            "Cascading SPICE refresh complete",
            extra={
                "children_total": len(results),
                "refresh_started": sum(1 for r in results if r.refresh_started),
                "refresh_unsupported": sum(
                    1 for r in results if r.refresh_unsupported
                ),
                "refresh_failed": sum(1 for r in results if r.error),
                "operation": "cascade_refresh",
            },
        )

        return results

    def validate_modification(
        self,
        retained_columns: set[str],
        consumer_references: set[str],
    ) -> bool:
        """Validate retained columns are superset of consumer references.

        Ensures that all columns referenced by consumers will still be
        present after the modification. This is the pre-modification safety
        check required by Requirement 8.4.

        Args:
            retained_columns: Set of column names that will be kept.
            consumer_references: Set of column names referenced by consumers.

        Returns:
            True if retained_columns is a superset of consumer_references.

        Raises:
            ValidationError: If retained_columns is NOT a superset (missing columns).
        """
        missing = consumer_references - retained_columns
        if missing:
            logger.error(
                "Pre-modification validation failed: retained columns missing consumer references",
                extra={
                    "missing_columns": sorted(missing),
                    "missing_count": len(missing),
                },
            )
            raise ValidationError(
                f"Retained columns do not cover all consumer references. "
                f"Missing {len(missing)} column(s): {sorted(missing)}",
                missing_columns=missing,
            )

        logger.info(
            "Pre-modification validation passed",
            extra={
                "retained_count": len(retained_columns),
                "consumer_refs_count": len(consumer_references),
            },
        )
        return True

    def _insert_project_operation(
        self,
        logical_table: dict,
        columns_to_keep: list[str],
    ) -> dict:
        """Insert ProjectOperation at the end of the transformation chain.

        Places ProjectOperation as the LAST transform in DataTransforms so it
        filters columns after all other operations (casts, renames, filters)
        have completed. This avoids breaking transforms that reference columns
        being excluded.

        If DataTransforms already has a ProjectOperation at the last position,
        it is replaced. Otherwise the new ProjectOperation is appended.

        Args:
            logical_table: A single logical table entry from LogicalTableMap.
            columns_to_keep: List of column names to include in the ProjectOperation.

        Returns:
            Modified logical table with ProjectOperation at the end of transforms.
        """
        project_operation = {
            "ProjectOperation": {
                "ProjectedColumns": list(columns_to_keep),
            }
        }

        # Get existing DataTransforms or create empty list
        data_transforms = logical_table.get("DataTransforms", [])

        if data_transforms:
            # Check if the last position already has a ProjectOperation
            if "ProjectOperation" in data_transforms[-1]:
                # Replace existing ProjectOperation at the end
                data_transforms[-1] = project_operation
            else:
                # Append new ProjectOperation after existing transforms
                data_transforms.append(project_operation)
        else:
            # No existing transforms, create with just our ProjectOperation
            data_transforms = [project_operation]

        logical_table["DataTransforms"] = data_transforms
        return logical_table

    def _get_all_column_names(self, dataset_def: dict) -> list[str]:
        """Extract all column names from the dataset definition.

        Uses OutputColumns if available, otherwise falls back to iterating
        through PhysicalTableMap column schemas.

        Args:
            dataset_def: The full dataset definition.

        Returns:
            List of all column names in the dataset.
        """
        # Try OutputColumns first (available in DescribeDataSet responses)
        output_columns = dataset_def.get("OutputColumns", [])
        if output_columns:
            return [col.get("Name", "") for col in output_columns if col.get("Name")]

        # Fallback: extract from PhysicalTableMap column schemas
        columns: list[str] = []
        physical_table_map = dataset_def.get("PhysicalTableMap", {})
        for _table_id, table_def in physical_table_map.items():
            # RelationalTable
            if "RelationalTable" in table_def:
                input_columns = table_def["RelationalTable"].get("InputColumns", [])
                columns.extend(col.get("Name", "") for col in input_columns if col.get("Name"))
            # CustomSql
            elif "CustomSql" in table_def:
                input_columns = table_def["CustomSql"].get("Columns", [])
                columns.extend(col.get("Name", "") for col in input_columns if col.get("Name"))
            # S3Source
            elif "S3Source" in table_def:
                input_columns = table_def["S3Source"].get("InputColumns", [])
                columns.extend(col.get("Name", "") for col in input_columns if col.get("Name"))

        return columns

    @staticmethod
    def _build_update_params(dataset_def: dict) -> dict:
        """Build UpdateDataSet API parameters from the modified definition.

        Delegates to ``lambda_source.update_params``, which selects LogicalTableMap for
        legacy datasets and DataPrepConfiguration for new-experience ones. ``snapshot.py``
        carried a second copy of this logic that had fallen behind; one function now
        serves both.

        Args:
            dataset_def: The modified dataset definition.

        Returns:
            Dict of parameters suitable for passing to QSClient.update_dataset.
        """
        return build_update_params(dataset_def)

    @staticmethod
    def _build_data_prep_config(
        physical_table_map: dict,
        projected_columns: list[str],
    ) -> dict:
        """Build a DataPrepConfiguration for new experience datasets.

        Constructs the pipeline: ImportTableStep(s) → ProjectStep → DestinationTable.
        Each physical table gets an ImportTableStep, then a single ProjectStep
        filters to only the projected columns, feeding into one DestinationTable.

        Args:
            physical_table_map: The PhysicalTableMap from the dataset definition.
            projected_columns: List of column names to keep after projection.

        Returns:
            A DataPrepConfiguration dict ready for the UpdateDataSet API.
        """
        source_table_map: dict[str, dict] = {}
        transform_step_map: dict[str, dict] = {}

        # Create ImportTableStep for each physical table
        import_step_ids: list[str] = []
        for table_id in physical_table_map:
            import_step_id = f"import-{table_id[:8]}"
            import_step_ids.append(import_step_id)

            source_table_map[table_id] = {
                "PhysicalTableId": table_id,
            }

            transform_step_map[import_step_id] = {
                "ImportTableStep": {
                    "Alias": f"Import {table_id[:8]}",
                    "Source": {
                        "SourceTableId": table_id,
                        "ColumnIdMappings": [],
                    },
                }
            }

        # If multiple tables, we need a JoinStep first, then project.
        # For simplicity with multiple tables, add the ProjectStep after
        # the last import step (QuickSight handles the join internally
        # for new experience datasets).
        # For a single table, project directly from the import step.
        if len(import_step_ids) == 1:
            project_source_id = import_step_ids[0]
        else:
            # For multi-table datasets, we need to reference the join
            # that QuickSight creates. Since the existing dataset already
            # has a join configured, we add the ProjectStep at the end.
            project_source_id = import_step_ids[-1]

        # Add ProjectStep to filter columns
        project_step_id = "project-optimiser"
        transform_step_map[project_step_id] = {
            "ProjectStep": {
                "Alias": "Column Optimization",
                "ProjectedColumns": projected_columns,
                "Source": {
                    "TransformOperationId": project_source_id,
                    "ColumnIdMappings": [],
                },
            }
        }

        # DestinationTable points to the project step
        destination_table_map = {
            "destination-1": {
                "Alias": "Output",
                "Source": {
                    "TransformOperationId": project_step_id,
                },
            }
        }

        return {
            "SourceTableMap": source_table_map,
            "TransformStepMap": transform_step_map,
            "DestinationTableMap": destination_table_map,
        }
