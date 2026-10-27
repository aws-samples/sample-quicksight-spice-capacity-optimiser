# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Core data models and type definitions for the SPICE Dataset Optimiser.

Defines dataclasses for configuration, consumer tracking, column classification,
quarantine state, and execution results. Also defines enumerations for execution
mode, column status, and consumer type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    """Execution mode for the optimiser."""

    DRY_RUN = "dry_run"
    APPLY = "apply"


class ColumnStatus(str, Enum):
    """Classification status for a dataset column."""

    USED = "used"
    UNUSED = "unused"
    JOIN_KEY = "join_key"


class ConsumerType(str, Enum):
    """Type of QuickSight consumer that references a dataset."""

    ANALYSIS = "analysis"
    DASHBOARD = "dashboard"


class ColumnUsageSource(str, Enum):
    """Why a column is protected from removal.

    The optimiser used to collapse every one of these into "referenced by
    consumer". That made the report unactionable: a column protected because a
    grandchild dataset's dashboard uses it read identically to one dropped
    straight into a visual, even though the remediation differs completely.
    """

    DIRECT_CONSUMER = "direct_consumer"
    """Referenced by an analysis or dashboard built directly on this dataset."""

    CHILD_CONSUMER = "child_consumer"
    """Referenced by a consumer of a child dataset, traced back to this one."""

    CHILD_JOIN_KEY = "child_join_key"
    """A join key inside a child dataset. Not removable without restructuring
    the child."""

    CHILD_PROJECTION = "child_projection"
    """Carried through a child's ProjectStep/RenameColumnsStep/etc. Potentially
    removable if the child is optimised first."""

    CHILD_STRUCTURAL = "child_structural"
    """Imported by a child dataset, but consumer tracing could not confirm that
    anything downstream displays it. Retained conservatively."""

    JOIN_KEY_LOGICAL_TABLE_MAP = "join_key_logical_table_map"
    """This dataset's own join key, declared in a legacy LogicalTableMap."""

    JOIN_KEY_DATA_PREP = "join_key_data_prep"
    """This dataset's own join key, declared in a new-experience
    DataPrepConfiguration JoinStep."""

    DATASET_CALC_FIELD = "dataset_calc_field"
    """Input to a calculated field defined on the dataset itself."""

    UNPARSEABLE_CALC_FIELD = "unparseable_calc_field"
    """A consumer calculated field could not be parsed, so every column was
    retained. Analysis is incomplete, not clean."""

    CHILD_REFS_UNKNOWN = "child_refs_unknown"
    """Child datasets exist but their column dependencies could not be
    determined, so every column was retained."""

    UNREACHABLE_CONSUMER = "unreachable_consumer"
    """A consumer or child dataset could not be read — usually AccessDenied — so its
    column usage is unknown and every column was retained. Distinct from "no
    consumers found", which it previously could not be told apart from: one means
    nothing uses this dataset, the other means we could not see what does, and the
    remediation is a permissions fix rather than a removal."""


# Human-readable description per usage source. Kept beside the enum so a new
# source cannot be added without deciding how it reads in a report.
_SOURCE_DESCRIPTIONS: dict[str, str] = {
    ColumnUsageSource.DIRECT_CONSUMER.value: "referenced by consumer",
    ColumnUsageSource.CHILD_CONSUMER.value: "referenced by child dataset consumer",
    ColumnUsageSource.CHILD_JOIN_KEY.value: "join key in child dataset",
    ColumnUsageSource.CHILD_PROJECTION.value: "projected by child dataset transform",
    ColumnUsageSource.CHILD_STRUCTURAL.value: "imported by child dataset",
    ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value: (
        "join key in LogicalTableMap JoinInstruction"
    ),
    ColumnUsageSource.JOIN_KEY_DATA_PREP.value: (
        "join key in DataPrepConfiguration JoinStep"
    ),
    ColumnUsageSource.DATASET_CALC_FIELD.value: (
        "input to dataset-level calculated field"
    ),
    ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value: (
        "retained: consumer calculated field could not be parsed"
    ),
    ColumnUsageSource.CHILD_REFS_UNKNOWN.value: (
        "retained: child dataset dependencies could not be determined"
    ),
    ColumnUsageSource.UNREACHABLE_CONSUMER.value: (
        "retained: a consumer could not be read, so its column usage is unknown"
    ),
}


@dataclass(frozen=True)
class ColumnAttribution:
    """One reason a specific column is protected, and who requires it.

    Frozen and hashable so duplicate attributions collapse in a set — the same
    child can be reached twice through a diamond dependency.
    """

    source: str
    """A ColumnUsageSource value."""

    requester_name: str = ""
    """Display name of the analysis, dashboard, or child dataset requiring it."""

    requester_id: str = ""
    """ARN or dataset ID of the requester, for unambiguous lookup."""

    detail: str = ""
    """Extra qualifier, e.g. 'depth 2' or the transform step type."""

    def describe(self) -> str:
        """Render as a single human-readable reason string."""
        base = _SOURCE_DESCRIPTIONS.get(self.source, self.source)

        qualifiers = [q for q in (self.requester_name, self.detail) if q]
        if not qualifiers:
            return base
        return f"{base} ({', '.join(qualifiers)})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimiserConfig:
    """Configuration parameters for an optimiser execution.

    Loaded from S3 JSON or inline event payload.
    """

    target_datasets: list[str]
    """Dataset IDs or tag filter values."""

    execution_mode: str = ExecutionMode.DRY_RUN.value
    """'dry_run' or 'apply'."""

    safety_threshold_pct: float = 0.7
    """Maximum percentage of columns removable (0.0–1.0)."""

    quarantine_period_days: int | None = None
    """Number of days a column must be unused before removal. None = disabled."""

    s3_bucket: str = ""
    """S3 bucket for snapshots and reports."""

    s3_prefix: str = ""
    """S3 key prefix for all stored objects."""

    sns_topic_arn: str | None = None
    """SNS topic ARN for notifications. None = no notifications."""

    dynamodb_table: str | None = None
    """DynamoDB table name for quarantine state. None = quarantine disabled."""

    aws_account_id: str = ""
    """AWS account ID where QuickSight resources reside."""

    aws_region: str = ""
    """AWS region for API calls."""

    skip_unreachable_consumers: bool = False
    """If True, skip unreachable consumers instead of halting. Use with caution."""

    cascade_child_refresh: bool = False
    """If True, after the parent's SPICE ingestion in apply mode, also trigger a SPICE
    ingestion for every descendant (child) dataset, in parent-before-child order. Opt-in
    because it fans out CreateIngestion calls across the dependency tree, which costs API
    quota and SPICE refresh capacity. Child refresh failures never fail the parent
    optimisation — the parent change is already committed. Ignored in dry_run mode."""


# ---------------------------------------------------------------------------
# Consumer Discovery
# ---------------------------------------------------------------------------


@dataclass
class Consumer:
    """A QuickSight analysis or dashboard that references a dataset."""

    arn: str
    """Full ARN of the consumer resource."""

    consumer_type: str
    """'analysis' or 'dashboard'."""

    name: str
    """Display name of the consumer."""

    dataset_identifier: str
    """The DataSetIdentifier used in the consumer's definition."""

    status: str
    """Current status of the consumer (e.g., 'CREATION_SUCCESSFUL', 'DELETED')."""


# ---------------------------------------------------------------------------
# Column Reference Extraction
# ---------------------------------------------------------------------------


@dataclass
class ColumnReference:
    """A single reference to a dataset column found in a consumer definition."""

    column_name: str
    """Name of the referenced column."""

    source: str
    """Where the reference was found (e.g., 'visual_field_well', 'calculated_field', 'filter')."""

    consumer_arn: str
    """ARN of the consumer containing this reference."""

    dataset_identifier: str
    """The DataSetIdentifier this reference is scoped to."""


# ---------------------------------------------------------------------------
# Column Classification
# ---------------------------------------------------------------------------


@dataclass
class ColumnClassification:
    """Classification result for a single dataset column."""

    column_name: str
    """Name of the column."""

    data_type: str
    """QuickSight data type (e.g., 'STRING', 'INTEGER', 'DECIMAL', 'DATETIME')."""

    status: str
    """Classification: 'used', 'unused', or 'join_key'."""

    reasons: list[str] = field(default_factory=list)
    """Human-readable reasons for the classification."""

    attributions: list[ColumnAttribution] = field(default_factory=list)
    """Structured provenance: which specific consumers and child datasets
    require this column, and why. Empty when attribution was not collected,
    in which case ``reasons`` carries the coarse legacy strings."""


# ---------------------------------------------------------------------------
# Quarantine State
# ---------------------------------------------------------------------------


@dataclass
class QuarantineEntry:
    """Tracks when an unused column was first and last detected."""

    dataset_id: str
    """ID of the dataset containing the column."""

    column_name: str
    """Name of the unused column."""

    first_detected: str
    """ISO-8601 timestamp of first detection as unused."""

    last_confirmed: str
    """ISO-8601 timestamp of most recent confirmation as unused."""


# ---------------------------------------------------------------------------
# Execution Results
# ---------------------------------------------------------------------------


@dataclass
class DatasetResult:
    """Result of processing a single dataset through the optimisation pipeline."""

    dataset_id: str
    """ID of the processed dataset."""

    dataset_name: str
    """Display name of the dataset."""

    status: str
    """Outcome status.

    Apply mode: 'optimized', 'skipped', 'failed', 'threshold_breached'.
    Dry run: 'scanned', 'skipped', 'failed', 'threshold_breached'.

    Dry run previously reported 'optimized', which is a claim that a dataset was changed.
    A scan changes nothing, so it now says what it did.
    """

    total_columns: int = 0
    """Total number of columns in the dataset."""

    used_columns_count: int = 0
    """Number of columns classified as used."""

    unused_columns_count: int = 0
    """Number of columns classified as unused."""

    columns_affected: list[str] = field(default_factory=list)
    """Columns the run acted on: removed in apply mode, identified for removal in dry run.

    Named for what it holds in both modes, because it was previously ``columns_removed``
    documented as "apply mode only" and populated by the dry-run path anyway. That leaked
    all the way out to callers, who were told a read-only scan had removed columns and
    freed capacity. The serialiser names this per mode, so nothing downstream has to
    infer whether a removal happened.
    """

    estimated_spice_savings_bytes: int = 0
    """Estimated SPICE capacity savings in bytes."""

    risk_indicators: list[str] = field(default_factory=list)
    """Risk indicators (e.g., 'shared dataset', 'no consumers found')."""

    error_message: str | None = None
    """Error message if processing failed."""

    report_s3_key: str | None = None
    """S3 key where the optimisation report was stored."""

    snapshot_s3_key: str | None = None
    """S3 key where the dataset snapshot was stored (apply mode only)."""

    child_refresh_results: list[dict] = field(default_factory=list)
    """Per-child cascading-refresh outcomes (apply mode, when cascade_child_refresh is
    on). Each entry is a serialised ChildRefreshResult: dataset_id, name, depth,
    refresh_started, refresh_unsupported, ingestion_id, error. Empty when cascading
    refresh was not requested or there were no child datasets."""


@dataclass
class ExecutionResult:
    """Aggregate result of a full optimiser execution across all target datasets."""

    execution_id: str
    """Unique identifier for this execution run."""

    timestamp: str
    """ISO-8601 timestamp when execution started."""

    datasets_processed: int = 0
    """Number of datasets successfully processed."""

    datasets_failed: int = 0
    """Number of datasets that encountered errors."""

    total_columns_affected: int = 0
    """Total columns acted on: removed in apply mode, identified in dry run."""

    total_spice_freed_bytes: int = 0
    """Total SPICE capacity in bytes: freed in apply mode, projected in dry run."""

    datasets_blocked: int = 0
    """Datasets refused by the safety threshold.

    Counted apart from failures, because nothing went wrong: the dataset was examined and
    deliberately left alone. Counted apart from successes too, because re-running changes
    nothing — the same threshold refuses again. A caller that cannot distinguish the three
    will either report work that did not happen or an error that did not occur.
    """

    execution_mode: str = ""
    """The mode this run executed in — ``dry_run`` or ``apply``.

    Carried on the result so the response serialiser can name its output fields for what
    actually happened. Without it the serialiser had no way to tell the two apart and
    emitted apply-mode vocabulary for both.
    """

    dataset_results: list[DatasetResult] = field(default_factory=list)
    """Per-dataset results."""
