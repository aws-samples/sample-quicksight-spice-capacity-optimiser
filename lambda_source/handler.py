# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Lambda entry point and orchestrator for the SPICE Dataset Optimiser.

Orchestrates the full optimisation pipeline:
Config → Discovery → Child Discovery → Scanner → Analyzer → Quarantine → Modifier → Reporter

Implements structured JSON logging, CloudWatch metrics emission, partial failure
handling, and both dry_run and apply execution modes.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3

from lambda_source.analyzer import UsageAnalyzer
from lambda_source.attribution import UsageAttribution
from lambda_source.child_discovery import (
    ChildDataset,
    ChildDatasetDiscovery,
    ChildDiscoveryError,
)
from lambda_source.config import load_config, validate_config
from lambda_source.dataset_lock import (
    DatasetLock,
    DatasetLockError,
    DatasetLockedError,
)
from lambda_source.discovery import (
    ConsumerDiscovery,
    ConsumerDiscoveryError,
    DiscoveryError,
)
from lambda_source.metrics import METRIC_NAMESPACE, emit_metric
from lambda_source.models import (
    ColumnStatus,
    ColumnUsageSource,
    DatasetResult,
    ExecutionMode,
    ExecutionResult,
    OptimiserConfig,
)
from lambda_source.modifier import DatasetModifier, ModificationError, ValidationError
from lambda_source.notifications import NotificationPublisher
from lambda_source.quarantine import QuarantineManager
from lambda_source.qs_client import QSClient, QSClientError
from lambda_source.reporter import ReportGenerator
from lambda_source.scanner import ColumnScanner
from lambda_source.snapshot import SnapshotError, SnapshotManager

logger = logging.getLogger(__name__)


def _is_scheduled_event(event: dict) -> bool:
    """Whether this is an EventBridge scheduled invocation.

    A bare scheduled event carries none of the pipeline or tool keys, so under the old
    key-presence routing it fell through to the tool dispatcher and returned
    "cannot identify tool". The weekly rule had been a 3ms no-op since at least
    2026-08-10 and nothing surfaced that.
    """
    return event.get("source") == "aws.events" or (
        event.get("detail-type") == "Scheduled Event"
    )


def route_event(event: dict, context: Any) -> str:
    """Decide which branch handles this event, by declaration rather than by guessing.

    Precedence, most explicit first:

    1. ``invocation_source`` — set by anything this codebase invokes itself with.
    2. A gateway tool name on the context — an unambiguous signal of a tool call.
    3. ``callback_job_id`` — only the pipeline is ever invoked with a callback, and
       treating it as anything else is what orphaned jobs on ``running`` for 7 days.
    4. ``target_datasets`` + ``execution_mode`` — a pipeline event. Note ``s3_bucket``
       is deliberately *not* required: it used to be, so an otherwise valid pipeline
       event missing it was silently reinterpreted as a different tool call. Config
       validation already rejects a missing bucket, with a legible error.
    5. An EventBridge scheduled event.
    6. Otherwise a tool call, resolved by payload shape as a last resort.

    Args:
        event: The raw Lambda event.
        context: The Lambda context, which carries the gateway tool name if any.

    Returns:
        Either ``"pipeline"`` or ``"tool"``.
    """
    declared = event.get("invocation_source")
    if declared in ("mcp_job", "pipeline", "schedule"):
        return "pipeline"
    if declared == "tool":
        return "tool"

    from lambda_source.mcp_dispatcher import tool_name_from_context

    if tool_name_from_context(context):
        return "tool"

    if event.get("callback_job_id"):
        return "pipeline"

    if "target_datasets" in event and "execution_mode" in event:
        return "pipeline"

    if _is_scheduled_event(event):
        return "pipeline"

    return "tool"


def _scheduled_event_config(event: dict) -> dict:
    """Fill in a scheduled event that did not bring its own configuration.

    The EventBridge rule now carries a complete pipeline event in its ``Input``, set
    from the ``Schedule*`` stack parameters, so nothing needs filling in on that path.
    This handles a bare scheduled event instead — an older rule created without an
    ``Input``, or a manual test — by pointing it at the S3 config location the function
    already knows.

    Without this a bare event has no ``target_datasets`` at all. That used to be the
    normal case: the rule shipped with no ``Input`` and there was no config object
    either, so the schedule did nothing for weeks and said nothing about it.
    """
    enriched = dict(event)
    enriched["invocation_source"] = "schedule"

    if enriched.get("target_datasets"):
        # Self-describing already; leave it alone.
        return enriched

    enriched.setdefault("config_s3_bucket", os.environ.get("S3_BUCKET", ""))
    enriched.setdefault("config_s3_key", os.environ.get("CONFIG_S3_KEY", ""))
    return enriched


def handler(event: dict, context: Any) -> dict:
    """Lambda entry point.

    Routes between the tool dispatcher and the optimisation pipeline. See
    :func:`route_event` for the precedence and why it is declaration-based.

    Guarantees that any event carrying ``callback_job_id`` reaches a terminal state.
    Callback fields are read before routing and the whole body is guarded, so a
    misroute, an unrecognised tool, or an unexpected exception all still write back.
    Previously the callback was only reachable inside the pipeline branch, so a
    misrouted job silently stayed ``running`` until its 7-day TTL expired.
    """
    event = event or {}

    # Configured here, before routing, rather than inside the pipeline branch. Tool
    # invocations previously ran with no structured logging at all, so every
    # logger.warning on that path landed in whatever default handler Lambda provided.
    _configure_logging()

    # Read before routing: a job must reach a terminal state no matter which branch
    # ends up handling it, including the branches that should never have seen it.
    callback_job_id = event.get("callback_job_id")
    callback_table = event.get("callback_table")

    try:
        return _route_and_run(event, context, callback_job_id, callback_table)
    except Exception as exc:  # noqa: BLE001 - last line of defence
        logger.error(
            "Unhandled exception before or during routing",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        error_response = {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if callback_job_id and callback_table:
            _write_job_callback(callback_job_id, callback_table, error_response)
        return error_response


def _route_and_run(
    event: dict,
    context: Any,
    callback_job_id: str | None,
    callback_table: str | None,
) -> dict:
    """Dispatch to the tool surface or the pipeline, per :func:`route_event`."""
    route = route_event(event, context)
    # Structured rather than a bare print. The routing defect this replaced printed
    # "TOOL_DISPATCH: unknown" on every misroute for weeks, and because it was not JSON
    # no Logs Insights query would surface it. It was found by reading a log stream by
    # hand.
    logger.info(
        "Event routed",
        extra={
            "route": route,
            "event_keys": sorted(event.keys()),
            "callback_job_id": callback_job_id,
            "invocation_source": event.get("invocation_source"),
            "operation": "routing",
        },
    )

    if route == "tool":
        from lambda_source.mcp_dispatcher import dispatch

        response = dispatch(event, context)
        # A tool call should not carry a callback, but if one arrives here the job
        # still has to reach a terminal state rather than being left running.
        if callback_job_id and callback_table:
            logger.warning(
                "Tool route received a job callback; writing terminal state anyway",
                extra={"callback_job_id": callback_job_id},
            )
            _write_job_callback(callback_job_id, callback_table, response)
        return response

    if _is_scheduled_event(event):
        event = _scheduled_event_config(event)

    return _run_pipeline(event, callback_job_id, callback_table)


def _run_pipeline(
    event: dict,
    callback_job_id: str | None,
    callback_table: str | None,
) -> dict:
    """Load config, run the optimisation pipeline, and always report a terminal state."""
    start_time = time.time()

    logger.info(
        "SPICE Dataset Optimiser invoked",
        extra={"event_keys": list(event.keys())},
    )

    try:
        # Load and validate configuration
        config = load_config(event)
        validation_errors = validate_config(config)

        if validation_errors:
            logger.error(
                "Configuration validation failed",
                extra={"errors": validation_errors},
            )
            error_response = {
                "status": "error",
                "error": "Configuration validation failed",
                "validation_errors": validation_errors,
            }
            # Write failure callback if job tracking is active
            if callback_job_id and callback_table:
                _write_job_callback(callback_job_id, callback_table, error_response)
            return error_response

        logger.info(
            "Configuration loaded successfully",
            extra={
                "execution_mode": config.execution_mode,
                "target_datasets_count": len(config.target_datasets),
                "aws_account_id": config.aws_account_id,
                "aws_region": config.aws_region,
            },
        )

        # Create orchestrator and run
        orchestrator = Orchestrator(config)
        result = orchestrator.run()

        # Emit execution duration metric
        duration_ms = (time.time() - start_time) * 1000
        orchestrator.emit_metric("ExecutionDuration", duration_ms, "Milliseconds")

        logger.info(
            "Execution complete",
            extra={
                "execution_id": result.execution_id,
                "execution_mode": result.execution_mode,
                "datasets_processed": result.datasets_processed,
                "datasets_failed": result.datasets_failed,
                "total_columns_affected": result.total_columns_affected,
                "total_spice_freed_bytes": result.total_spice_freed_bytes,
                "duration_ms": duration_ms,
            },
        )

        response = _build_response(result, duration_ms)

        # Write success/partial callback if job tracking is active
        if callback_job_id and callback_table:
            _write_job_callback(callback_job_id, callback_table, response)

        return response

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "Unhandled exception in handler",
            extra={
                "error": str(exc),
                "error_type": type(exc).__name__,
                "duration_ms": duration_ms,
            },
        )
        error_response = {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "duration_ms": duration_ms,
        }
        # Write failure callback if job tracking is active
        if callback_job_id and callback_table:
            _write_job_callback(callback_job_id, callback_table, error_response)
        return error_response


# QuickSight's wording when a dataset has no refreshable source. Matched on the phrase
# rather than on the error code, because InvalidParameterValueException covers a great
# many unrelated problems and treating all of them as "no refresh needed" would hide
# real failures.
_UNREFRESHABLE_MARKERS = (
    "not refreshable because of the source type",
    "dataset is not accelerated",
)


def _refresh_is_unsupported(exc: BaseException) -> bool:
    """Whether this dataset simply has no refresh to start, as opposed to a failed one.

    QuickSight cannot refresh a dataset built from an uploaded file: it keeps no
    connection or location metadata to re-read from, so ``CreateIngestion`` is rejected
    outright rather than failing transiently.
    ``https://docs.aws.amazon.com/quicksight/latest/user/refreshing-data.html``

    Read off QuickSight's own answer rather than re-derived from PhysicalTableMap. The
    service is the authority on whether a dataset is refreshable, and inspecting the
    definition ourselves would mean maintaining a second copy of that rule and getting
    it wrong whenever a new source variant appears - which is exactly the trap the
    FileSource variant already sprang once on the SDK floor.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _UNREFRESHABLE_MARKERS)


def execute_rollback(
    dataset_id: str,
    aws_account_id: str,
    aws_region: str,
    s3_bucket: str,
    s3_prefix: str,
    snapshot_key: str | None = None,
    trigger_refresh: bool = True,
) -> dict:
    """Restore a dataset from a snapshot, under the per-dataset lock.

    Lives here rather than in ``mcp_dispatcher`` for the same reason the apply-mode lock
    does: rollback is a describe-modify-write against a dataset, so it must not race a
    concurrent optimisation run. Two writers would mean one silently discarding the
    other's definition, and afterwards there would be no way to tell which state the
    dataset is actually in. Keeping the guarantee in the pipeline layer means it survives
    any future change to the tool surface.

    Synchronous rather than an async job. ``UpdateDataSet`` returns in seconds and
    ``CreateIngestion`` is itself asynchronous, so there is nothing worth polling for —
    and an undo that makes you poll is a worse undo.

    Args:
        dataset_id: Dataset to restore.
        aws_account_id: Account owning the dataset.
        aws_region: Region for API calls.
        s3_bucket: Bucket holding snapshots.
        s3_prefix: Key prefix for snapshots.
        snapshot_key: Specific snapshot to restore. Defaults to the most recent.
        trigger_refresh: Whether to start a SPICE ingestion afterwards. Restored columns
            hold no data until the dataset refreshes, so the default is on.

    Returns:
        A result dict with ``status`` and, on success, the snapshot restored and whether
        a refresh was started.
    """
    _configure_logging()

    qs_client = QSClient(account_id=aws_account_id, region=aws_region)
    snapshot_manager = SnapshotManager(
        s3_bucket=s3_bucket, s3_prefix=s3_prefix, qs_client=qs_client
    )
    lock = DatasetLock(s3_bucket=s3_bucket, s3_prefix=s3_prefix)
    execution_id = str(uuid.uuid4())

    try:
        with lock.held(dataset_id, execution_id=execution_id):
            restored_from = snapshot_key or snapshot_manager._find_most_recent_snapshot(
                dataset_id
            )
            snapshot_manager.rollback_dataset(dataset_id, snapshot_key=restored_from)

            refresh_started = False
            ingestion_error = None
            refresh_unsupported = False
            if trigger_refresh:
                # QuickSight requires a caller-supplied ingestion id. Prefixed the same
                # way modifier.py does it, so a refresh is attributable to this tool
                # whichever path started it, and marked as a rollback so the two are
                # distinguishable in the QuickSight ingestion history.
                ingestion_id = f"optimiser-rollback-{uuid.uuid4().hex[:12]}"
                try:
                    qs_client.create_ingestion(dataset_id, ingestion_id)
                    refresh_started = True
                except Exception as exc:  # noqa: BLE001
                    # The restore already succeeded and is the part that matters. A failed
                    # refresh leaves correct metadata with stale data, which is
                    # recoverable by refreshing manually — so report it rather than
                    # presenting the whole rollback as failed.
                    #
                    # Deliberately broad. This was `except QSClientError`, which is the
                    # narrow catch the guidance asks for, and it made the promise above
                    # untrue: the first failure here was a TypeError from calling
                    # create_ingestion with one argument instead of two, and because a
                    # TypeError is not a QSClientError it escaped to the top-level
                    # handler. The caller was told the rollback failed, seconds after
                    # UpdateDataSet had in fact restored the dataset - the single most
                    # misleading thing an undo can say.
                    #
                    # The catch has to be as wide as the promise. Anything at all that
                    # goes wrong while starting a refresh must not change the fact that
                    # the definition is already restored.
                    ingestion_error = f"{type(exc).__name__}: {exc}"
                    refresh_unsupported = _refresh_is_unsupported(exc)
                    logger.warning(
                        "Rollback restored the definition but could not start a refresh",
                        exc_info=True,
                        extra={
                            "dataset_id": dataset_id,
                            "error": ingestion_error,
                            "refresh_unsupported": refresh_unsupported,
                            "operation": "rollback_ingestion",
                        },
                    )

            logger.info(
                "Rollback complete",
                extra={
                    "dataset_id": dataset_id,
                    "snapshot_key": restored_from,
                    "refresh_started": refresh_started,
                    "execution_id": execution_id,
                    "operation": "rollback",
                },
            )

            if refresh_started:
                refresh_note = (
                    " A SPICE refresh has been started; restored columns hold no "
                    "data until it finishes."
                )
            elif refresh_unsupported:
                # Not a failure to report as one. QuickSight cannot refresh a dataset
                # built from an uploaded file, because it keeps no connection or
                # location metadata to re-read from - so there is no refresh to start
                # and nothing for the caller to do about it. Saying "columns will be
                # empty until the dataset is refreshed" here would be both alarming and
                # untrue, and it is what this branch exists to avoid.
                refresh_note = (
                    " No SPICE refresh was started, and none is needed: QuickSight does "
                    "not support explicitly refreshing a dataset built from an uploaded "
                    "file, and re-reads the stored file itself when the definition "
                    "changes. If the data looks stale, re-upload the file or move the "
                    "dataset to an S3 data source, which can be refreshed on a schedule."
                )
            else:
                refresh_note = (
                    " No refresh was started, so restored columns will be empty until "
                    "the dataset is refreshed."
                )

            result = {
                "status": "success",
                "dataset_id": dataset_id,
                "restored_from_snapshot": restored_from,
                "refresh_started": refresh_started,
                "message": f"Dataset restored from snapshot {restored_from}.{refresh_note}",
            }
            if ingestion_error:
                result["ingestion_error"] = ingestion_error
                # Distinguishes "the refresh broke" from "this dataset has no refresh",
                # so a caller does not have to pattern-match the error text to tell
                # whether anything is actually wrong.
                result["refresh_unsupported"] = refresh_unsupported
            return result

    except DatasetLockedError as exc:
        return {
            "status": "error",
            "error": (
                f"Dataset {dataset_id} is currently being optimised by another run "
                f"(holder: {exc.holder}). Refusing to roll back mid-flight, because two "
                "writers would leave the dataset in an indeterminate state. Retry once "
                "that run finishes."
            ),
        }
    except DatasetLockError as exc:
        return {
            "status": "error",
            "error": f"Could not acquire the dataset lock, so rollback was not attempted: {exc}",
        }
    except SnapshotError as exc:
        return {"status": "error", "error": str(exc)}
    except QSClientError as exc:
        return {
            "status": "error",
            "error": f"Rollback failed calling QuickSight: {exc}",
        }


class Orchestrator:
    """Orchestrates the full optimisation pipeline for all target datasets.

    Wires together all pipeline components: QSClient, ConsumerDiscovery,
    ChildDatasetDiscovery, ColumnScanner, UsageAnalyzer, QuarantineManager,
    SnapshotManager, DatasetModifier, ReportGenerator, and NotificationPublisher.
    """

    def __init__(self, config: OptimiserConfig):
        """Initialize the Orchestrator with all pipeline components.

        Args:
            config: Validated OptimiserConfig instance.
        """
        self._config = config

        # Initialize AWS clients
        self._qs_client = QSClient(
            account_id=config.aws_account_id,
            region=config.aws_region,
        )
        self._cloudwatch_client = boto3.client(
            "cloudwatch", region_name=config.aws_region
        )

        # Initialize pipeline components
        self._discovery = ConsumerDiscovery(
            qs_client=self._qs_client,
            account_id=config.aws_account_id,
        )
        self._child_discovery = ChildDatasetDiscovery(
            qs_client=self._qs_client,
            account_id=config.aws_account_id,
            region=config.aws_region,
        )
        self._scanner = ColumnScanner()
        self._analyzer = UsageAnalyzer()
        self._quarantine_manager = QuarantineManager(
            dynamodb_table=config.dynamodb_table,
        )
        self._snapshot_manager = SnapshotManager(
            s3_bucket=config.s3_bucket,
            s3_prefix=config.s3_prefix,
            qs_client=self._qs_client,
        )
        # Serialises apply-mode runs per dataset. Replaces the guarantee that
        # ReservedConcurrentExecutions: 1 used to provide, which was removed in
        # 15b3d39 because the async job pattern self-invokes this function.
        self._dataset_lock = DatasetLock(
            s3_bucket=config.s3_bucket,
            s3_prefix=config.s3_prefix,
        )
        self._modifier = DatasetModifier(qs_client=self._qs_client)
        self._reporter = ReportGenerator(
            s3_bucket=config.s3_bucket,
            s3_prefix=config.s3_prefix,
        )
        self._notifier = NotificationPublisher(sns_topic_arn=config.sns_topic_arn)

        # Recorded in each dataset lock so a human can trace which run holds it.
        # run() overwrites this; the default covers process_dataset being driven
        # directly, as the tests do.
        self._execution_id = str(uuid.uuid4())

    def run(self) -> ExecutionResult:
        """Execute the full optimisation pipeline for all target datasets.

        Processes each target dataset via process_dataset(), continues on
        failure (collecting errors), sends error notifications for failed
        datasets, and emits CloudWatch metrics.

        Returns:
            ExecutionResult with per-dataset results and aggregate counts.
        """
        execution_id = str(uuid.uuid4())
        self._execution_id = execution_id
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        result = ExecutionResult(
            execution_id=execution_id,
            timestamp=timestamp,
            execution_mode=self._config.execution_mode,
        )

        logger.info(
            "Starting optimisation run",
            extra={
                "execution_id": execution_id,
                "target_datasets": self._config.target_datasets,
                "execution_mode": self._config.execution_mode,
            },
        )

        errors_encountered = 0

        for dataset_id in self._config.target_datasets:
            try:
                dataset_result = self.process_dataset(dataset_id)
                result.dataset_results.append(dataset_result)

                if dataset_result.status == "failed":
                    result.datasets_failed += 1
                    errors_encountered += 1
                    # Send error notification for failed dataset
                    self._notifier.publish_error(
                        dataset_id=dataset_id,
                        error_message=dataset_result.error_message or "Unknown error",
                    )
                else:
                    result.datasets_processed += 1
                    result.total_columns_affected += len(dataset_result.columns_affected)
                    if dataset_result.status == "threshold_breached":
                        # A blocked dataset contributes no columns and no saving. Its
                        # estimated_spice_savings_bytes is what removal *would* free if
                        # the threshold were raised, which is meaningful in that
                        # dataset's own report — next to threshold_breached: true — and
                        # misleading in an aggregate that does not mention it.
                        #
                        # Observed live on a three-dataset scan: two were scanned and
                        # found 4 columns, one was blocked, and the total read "4 unused
                        # column(s) ... would free about 5432918 bytes" — a figure
                        # partly belonging to a dataset the tool refuses to touch, and
                        # which no amount of re-running will realise.
                        result.datasets_blocked += 1
                    else:
                        result.total_spice_freed_bytes += (
                            dataset_result.estimated_spice_savings_bytes
                        )

            except Exception as exc:
                # Catch-all for unexpected errors — continue processing
                logger.error(
                    "Unexpected error processing dataset",
                    extra={
                        "dataset_id": dataset_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                errors_encountered += 1
                result.datasets_failed += 1

                error_result = DatasetResult(
                    dataset_id=dataset_id,
                    dataset_name="",
                    status="failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
                result.dataset_results.append(error_result)

                # Send error notification
                self._notifier.publish_error(
                    dataset_id=dataset_id,
                    error_message=str(exc),
                )

        # Emit aggregate CloudWatch metrics.
        #
        # Separate metric names per mode. ColumnsRemoved and SpiceCapacityFreed were
        # emitted for dry runs too, so a scan added to the running total of columns
        # removed and capacity freed — making a dashboard of either one a sum of work
        # done and work merely contemplated, with no way to separate them after the fact.
        is_dry_run = result.execution_mode == ExecutionMode.DRY_RUN.value
        self.emit_metric("DatasetsProcessed", result.datasets_processed, "Count")
        self.emit_metric("ErrorsEncountered", errors_encountered, "Count")
        if is_dry_run:
            self.emit_metric("ColumnsIdentified", result.total_columns_affected, "Count")
            self.emit_metric(
                "SpiceCapacityIdentified", result.total_spice_freed_bytes, "Bytes"
            )
        else:
            self.emit_metric("ColumnsRemoved", result.total_columns_affected, "Count")
            self.emit_metric(
                "SpiceCapacityFreed", result.total_spice_freed_bytes, "Bytes"
            )

        logger.info(
            "Optimisation run complete",
            extra={
                "execution_id": execution_id,
                "execution_mode": result.execution_mode,
                "datasets_processed": result.datasets_processed,
                "datasets_failed": result.datasets_failed,
                "total_columns_affected": result.total_columns_affected,
                "total_spice_freed_bytes": result.total_spice_freed_bytes,
            },
        )

        return result

    def process_dataset(self, dataset_id: str) -> DatasetResult:
        """Process a single dataset through the optimisation pipeline.

        Pipeline steps:
        1. Describe dataset to get full column list
        2. Discover all consuming analyses/dashboards
        3. Scan each consumer to extract column references
        3b. Discover child datasets and merge their column references
        4. Extract join keys and dataset-level calc field inputs
        5. Classify columns (used/unused/join_key)
        6. Quarantine filtering (if enabled)
        7. Safety threshold check
        8. Dry run: generate report only (no write APIs)
        9. Apply: snapshot → validate → modify → ingest → report

        Args:
            dataset_id: QuickSight dataset identifier to process.

        Returns:
            DatasetResult with processing outcome and details.
        """
        logger.info(
            "Processing dataset",
            extra={"dataset_id": dataset_id, "operation": "process_dataset_start"},
        )

        # Step 1: Describe dataset to get column list
        try:
            dataset_def = self._qs_client.describe_dataset(dataset_id)
        except QSClientError as exc:
            logger.error(
                "Failed to describe dataset",
                extra={"dataset_id": dataset_id, "error": str(exc)},
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name="",
                status="failed",
                error_message=f"Failed to describe dataset: {exc}",
            )

        dataset_name = dataset_def.get("Name", dataset_id)
        all_columns = dataset_def.get("OutputColumns", [])
        logical_table_map = dataset_def.get("LogicalTableMap", {})
        consumed_spice_bytes = dataset_def.get("ConsumedSpiceCapacityInBytes")

        logger.info(
            "Dataset described",
            extra={
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "column_count": len(all_columns),
                "operation": "describe_dataset",
            },
        )

        # Step 2: Discover consumers (analyses and dashboards)
        #
        # The accumulator is what makes the skip_unreachable path safe. Without it the
        # method returned consumers with unreadable entries simply missing, so a dataset
        # whose only dashboard returned AccessDenied looked exactly like a dataset with
        # no dashboards — and its columns became removal candidates.
        unreachable_consumers: list[DiscoveryError] = []
        try:
            consumers = self._discovery.discover_consumers(
                dataset_id,
                skip_unreachable=self._config.skip_unreachable_consumers,
                unreachable=unreachable_consumers,
            )
        except ConsumerDiscoveryError as exc:
            logger.error(
                "Consumer discovery halted",
                extra={
                    "dataset_id": dataset_id,
                    "unreachable_count": len(exc.unreachable_consumers),
                    "error": str(exc),
                },
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(all_columns),
                error_message=f"Consumer discovery halted: {exc}",
            )
        except QSClientError as exc:
            logger.error(
                "Consumer discovery failed",
                extra={"dataset_id": dataset_id, "error": str(exc)},
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(all_columns),
                error_message=f"Consumer discovery failed: {exc}",
            )

        logger.info(
            "Consumer discovery complete",
            extra={
                "dataset_id": dataset_id,
                "consumer_count": len(consumers),
                "operation": "consumer_discovery",
            },
        )

        # Step 3: Scan each consumer definition for column references
        #
        # `used_columns` drives the removal decision and is built exactly as
        # before. `attribution` records the same references keyed by column so the
        # report can explain each one. It has no influence on classification —
        # that separation is deliberate, so a reporting change cannot alter what
        # gets removed.
        used_columns: set[str] = set()
        attribution = UsageAttribution()
        for consumer in consumers:
            try:
                # Get consumer definition
                if consumer.consumer_type == "analysis":
                    analysis_id = self._extract_id_from_arn(consumer.arn, "analysis")
                    definition = self._qs_client.describe_analysis_definition(analysis_id)
                else:
                    dashboard_id = self._extract_id_from_arn(consumer.arn, "dashboard")
                    definition = self._qs_client.describe_dashboard_definition(dashboard_id)

                # Scan definition for column references
                columns = self._scanner.scan_definition(
                    definition=definition,
                    dataset_identifier=consumer.dataset_identifier,
                    consumer_arn=consumer.arn,
                )

                # Handle unparseable calculated field (conservative fallback)
                if "__UNPARSEABLE__" in columns:
                    columns.discard("__UNPARSEABLE__")
                    # Include all dataset columns conservatively
                    all_col_names = {
                        col.get("Name", "") for col in all_columns if col.get("Name")
                    }
                    used_columns.update(all_col_names)
                    # Both facts get recorded. The columns the scanner did resolve
                    # are genuinely referenced; the rest are retained only because
                    # the expression defeated the parser. Attributing them
                    # identically is what made an incomplete analysis read as a
                    # clean one.
                    attribution.record_many(
                        columns,
                        ColumnUsageSource.DIRECT_CONSUMER.value,
                        requester_name=consumer.name,
                        requester_id=consumer.arn,
                    )
                    attribution.record_many(
                        all_col_names,
                        ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value,
                        requester_name=consumer.name,
                        requester_id=consumer.arn,
                    )
                else:
                    used_columns.update(columns)
                    attribution.record_many(
                        columns,
                        ColumnUsageSource.DIRECT_CONSUMER.value,
                        requester_name=consumer.name,
                        requester_id=consumer.arn,
                    )

            except QSClientError as exc:
                logger.warning(
                    "Failed to scan consumer, skipping",
                    extra={
                        "dataset_id": dataset_id,
                        "consumer_arn": consumer.arn,
                        "error": str(exc),
                    },
                )
                # Continue processing - non-fatal for individual consumer scan failure
                # (Discovery already validated reachability)

        logger.info(
            "Column scanning complete",
            extra={
                "dataset_id": dataset_id,
                "used_columns_count": len(used_columns),
                "operation": "column_scanning",
            },
        )

        # Step 3b: Discover child datasets (datasets using this one as a source)
        unreachable_children: list[str] = []
        try:
            child_datasets = self._child_discovery.discover_child_datasets(
                target_dataset_id=dataset_id,
                skip_unreachable=self._config.skip_unreachable_consumers,
                unreachable_out=unreachable_children,
            )
        except ChildDiscoveryError as exc:
            logger.error(
                "Child dataset discovery halted",
                extra={
                    "dataset_id": dataset_id,
                    "unreachable_count": len(exc.unreachable_datasets),
                    "error": str(exc),
                },
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(all_columns),
                error_message=f"Child dataset discovery halted: {exc}",
            )
        except QSClientError as exc:
            logger.error(
                "Child dataset discovery failed",
                extra={"dataset_id": dataset_id, "error": str(exc)},
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(all_columns),
                error_message=f"Child dataset discovery failed: {exc}",
            )

        if child_datasets:
            # First: get structural dependencies (columns the child needs from parent)
            child_refs = self._child_discovery.get_all_child_referenced_columns(
                child_datasets
            )

            if child_refs:
                # Scan child datasets' consumers to find which parent columns
                # are actually used end-to-end through the dependency chain.
                # Consumer tracing is more precise than structural deps because
                # InputColumns lists ALL columns a child imports, not just what's
                # actually needed by downstream visuals.
                try:
                    consumer_traced_refs = self._child_discovery.scan_child_consumers(
                        children=child_datasets,
                        target_dataset_id=dataset_id,
                        scanner=self._scanner,
                    )
                except Exception as exc:
                    logger.warning(
                        "Child consumer scanning failed, falling back to "
                        "structural dependencies",
                        extra={
                            "dataset_id": dataset_id,
                            "error": str(exc),
                        },
                    )
                    consumer_traced_refs = None

                if consumer_traced_refs is not None:
                    # Use consumer-traced refs (already includes join keys)
                    used_columns.update(consumer_traced_refs)
                    self._attribute_child_columns(
                        attribution, child_datasets, consumer_traced_refs
                    )
                    logger.info(
                        "Child dataset column references merged (consumer-traced)",
                        extra={
                            "dataset_id": dataset_id,
                            "child_count": len(child_datasets),
                            "max_depth": max((c.depth for c in child_datasets), default=0),
                            "structural_refs": len(child_refs),
                            "consumer_traced_refs": len(consumer_traced_refs),
                            "used_columns_after_merge": len(used_columns),
                            "operation": "child_dataset_discovery",
                        },
                    )
                else:
                    # Fallback: use structural deps (conservative)
                    used_columns.update(child_refs)
                    for child in child_datasets:
                        attribution.record_many(
                            child.referenced_columns,
                            ColumnUsageSource.CHILD_STRUCTURAL.value,
                            requester_name=child.name,
                            requester_id=child.dataset_id,
                            detail=f"depth {child.depth}",
                        )
                    logger.info(
                        "Child dataset column references merged (structural fallback)",
                        extra={
                            "dataset_id": dataset_id,
                            "child_count": len(child_datasets),
                            "structural_refs": len(child_refs),
                            "used_columns_after_merge": len(used_columns),
                            "operation": "child_dataset_discovery",
                        },
                    )
            else:
                # Empty set means we couldn't determine which columns are used —
                # conservatively mark ALL columns as used to avoid breaking children
                all_col_names = {
                    col.get("Name", "") for col in all_columns if col.get("Name")
                }
                used_columns.update(all_col_names)
                # Previously this only reached the log. A run where every column
                # was retained because the dependencies were unreadable looked
                # identical in the report to one where every column was genuinely
                # in use, which is the more dangerous of the two to misread.
                for child in child_datasets:
                    attribution.record_many(
                        all_col_names,
                        ColumnUsageSource.CHILD_REFS_UNKNOWN.value,
                        requester_name=child.name,
                        requester_id=child.dataset_id,
                        detail=f"depth {child.depth}",
                    )
                logger.warning(
                    "Child dataset column references unknown, "
                    "conservatively marking all columns as used",
                    extra={
                        "dataset_id": dataset_id,
                        "child_count": len(child_datasets),
                        "child_dataset_ids": [c.dataset_id for c in child_datasets],
                        "operation": "child_dataset_discovery",
                    },
                )
        else:
            logger.info(
                "No child datasets found",
                extra={
                    "dataset_id": dataset_id,
                    "operation": "child_dataset_discovery",
                },
            )

        # Step 3c: Protect everything if any consumer or child could not be read.
        #
        # This implements the behaviour discover_consumers' docstring has always
        # promised — "columns found only in unreachable consumers will be
        # conservatively marked as used" — and which nothing implemented. An unreadable
        # consumer must block removal, not look like an absent one: its columns were
        # never scanned, so without this they are classified unused and removing them
        # breaks the dashboard nobody could read.
        #
        # Placed after both discovery steps so it covers consumers and child datasets
        # with one rule.
        if unreachable_consumers or unreachable_children:
            all_col_names = {
                col.get("Name", "") for col in all_columns if col.get("Name")
            }
            used_columns.update(all_col_names)

            for failure in unreachable_consumers:
                attribution.record_many(
                    all_col_names,
                    ColumnUsageSource.UNREACHABLE_CONSUMER.value,
                    requester_name=failure.arn.rsplit("/", 1)[-1] or failure.arn,
                    requester_id=failure.arn,
                    detail=failure.error_code,
                )
            for child_id in unreachable_children:
                attribution.record_many(
                    all_col_names,
                    ColumnUsageSource.UNREACHABLE_CONSUMER.value,
                    requester_name=child_id,
                    requester_id=child_id,
                    detail="child dataset could not be described",
                )

            logger.warning(
                "Unreadable consumers or children; protecting every column",
                extra={
                    "dataset_id": dataset_id,
                    "unreachable_consumer_count": len(unreachable_consumers),
                    "unreachable_child_count": len(unreachable_children),
                    "unreachable_consumer_arns": [
                        f.arn for f in unreachable_consumers
                    ],
                    "unreachable_child_ids": unreachable_children,
                    "columns_protected": len(all_col_names),
                    "operation": "unreachable_consumer_guard",
                },
            )

        # Step 4: Extract join keys and dataset-level calc field inputs
        #
        # extract_all_join_keys covers both preparation experiences. The older
        # extract_join_keys reads only LogicalTableMap, so a new-experience dataset —
        # which keeps joins in DataPrepConfiguration.TransformStepMap — got no join key
        # protection at all. Results are intersected with the dataset's real columns,
        # because the table-qualified brace form yields two candidate readings.
        all_col_names_for_joins = {
            col.get("Name", "") for col in all_columns if col.get("Name")
        }
        # Split by declaring shape so the report can name where the operator will
        # actually find the join. The union is identical to extract_all_join_keys.
        join_keys_by_source = self._analyzer.extract_join_keys_by_source(
            dataset_def, all_col_names_for_joins
        )
        join_keys: set[str] = set()
        for source, columns_from_source in join_keys_by_source.items():
            join_keys.update(columns_from_source)
            attribution.record_many(
                columns_from_source, source, requester_name=dataset_name
            )

        calc_field_inputs = self._analyzer.extract_dataset_calc_field_inputs(dataset_def)
        attribution.record_many(
            calc_field_inputs,
            ColumnUsageSource.DATASET_CALC_FIELD.value,
            requester_name=dataset_name,
        )

        logger.info(
            "Join keys and calc field inputs extracted",
            extra={
                "dataset_id": dataset_id,
                "join_key_count": len(join_keys),
                "calc_field_input_count": len(calc_field_inputs),
                "operation": "join_key_extraction",
            },
        )

        # Step 5: Classify columns
        classifications = self._analyzer.classify_columns(
            all_columns=all_columns,
            used_columns=used_columns,
            join_key_columns=join_keys,
            calc_field_input_columns=calc_field_inputs,
            attribution=attribution,
        )

        unused_classifications = self._analyzer.get_unused_columns(classifications)
        unused_column_names = [c.column_name for c in unused_classifications]

        logger.info(
            "Column classification complete",
            extra={
                "dataset_id": dataset_id,
                "total_columns": len(classifications),
                "unused_columns": len(unused_column_names),
                "operation": "column_classification",
            },
        )

        # Step 6: Quarantine filtering (if enabled)
        #
        # Requested-but-unavailable is a refusal, not a skip. This compound condition
        # used to fall through silently when the manager was disabled, so a run that
        # asked for a 14-day waiting period removed columns immediately and reported
        # success. The operator had no way to know the guardrail they specified was
        # never applied. Quarantine remains entirely optional — omitting
        # quarantine_period_days is the supported way to run without it.
        if (
            self._config.quarantine_period_days is not None
            and not self._quarantine_manager.enabled
        ):
            logger.error(
                "Quarantine requested but no tracking table is configured",
                extra={
                    "dataset_id": dataset_id,
                    "quarantine_period_days": self._config.quarantine_period_days,
                    "dynamodb_table": self._config.dynamodb_table,
                    "operation": "quarantine_unavailable",
                },
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(classifications),
                used_columns_count=len(classifications) - len(unused_classifications),
                unused_columns_count=len(unused_classifications),
                risk_indicators=self._get_risk_indicators(
                    len(consumers), child_datasets=child_datasets
                ),
                error_message=(
                    f"quarantine_period_days={self._config.quarantine_period_days} was "
                    "requested but no quarantine tracking table is configured, so the "
                    "waiting period cannot be enforced. Refusing to remove columns "
                    "without the requested guardrail. Redeploy with "
                    "QuarantineEnabled=true, or omit quarantine_period_days."
                ),
            )

        if (
            self._config.quarantine_period_days is not None
            and self._quarantine_manager.enabled
        ):
            # Update tracking for all unused columns
            self._quarantine_manager.update_tracking(dataset_id, unused_column_names)

            # Remove used columns from tracking (they may have been unused before)
            used_col_names = [
                c.column_name for c in classifications
                if c.status != ColumnStatus.UNUSED.value
            ]
            self._quarantine_manager.remove_from_tracking(dataset_id, used_col_names)

            # Filter to only those past quarantine period
            eligible_columns = self._quarantine_manager.filter_by_quarantine(
                unused_columns=unused_column_names,
                dataset_id=dataset_id,
                quarantine_days=self._config.quarantine_period_days,
            )

            logger.info(
                "Quarantine filtering applied",
                extra={
                    "dataset_id": dataset_id,
                    "total_unused": len(unused_column_names),
                    "eligible_after_quarantine": len(eligible_columns),
                    "operation": "quarantine_filter",
                },
            )

            unused_column_names = eligible_columns

        # Step 7: Safety threshold check
        is_safe = self._analyzer.check_safety_threshold(
            classifications, self._config.safety_threshold_pct
        )
        threshold_breached = not is_safe

        if threshold_breached:
            logger.warning(
                "Safety threshold breached, halting optimization",
                extra={
                    "dataset_id": dataset_id,
                    "threshold": self._config.safety_threshold_pct,
                    "unused_count": len(unused_column_names),
                    "total_count": len(classifications),
                    "operation": "safety_threshold_check",
                },
            )

            # Generate report even on threshold breach
            report = self._reporter.generate_report(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                execution_mode=self._config.execution_mode,
                classifications=classifications,
                consumer_count=len(consumers),
                safety_threshold_pct=self._config.safety_threshold_pct,
                threshold_breached=True,
                consumers=consumers,
                consumed_spice_capacity_bytes=consumed_spice_bytes,
                child_datasets=child_datasets,
            )
            report_key = self._reporter.store_report(report, dataset_id)

            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="threshold_breached",
                total_columns=len(classifications),
                used_columns_count=len(classifications) - len(unused_classifications),
                unused_columns_count=len(unused_classifications),
                estimated_spice_savings_bytes=self._reporter.estimate_spice_savings(
                    unused_classifications,
                    total_columns=len(classifications),
                    consumed_spice_capacity_bytes=consumed_spice_bytes,
                ),
                risk_indicators=self._get_risk_indicators(
                    len(consumers), child_datasets=child_datasets
                ),
                report_s3_key=report_key,
            )

        # No columns to remove
        if not unused_column_names:
            logger.info(
                "No unused columns eligible for removal",
                extra={
                    "dataset_id": dataset_id,
                    "operation": "no_optimization_needed",
                },
            )

            report = self._reporter.generate_report(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                execution_mode=self._config.execution_mode,
                classifications=classifications,
                consumer_count=len(consumers),
                safety_threshold_pct=self._config.safety_threshold_pct,
                threshold_breached=False,
                consumers=consumers,
                consumed_spice_capacity_bytes=consumed_spice_bytes,
                child_datasets=child_datasets,
            )
            report_key = self._reporter.store_report(report, dataset_id)

            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="skipped",
                total_columns=len(classifications),
                used_columns_count=len(classifications) - len(unused_classifications),
                unused_columns_count=len(unused_classifications),
                estimated_spice_savings_bytes=0,
                risk_indicators=self._get_risk_indicators(
                    len(consumers), child_datasets=child_datasets
                ),
                report_s3_key=report_key,
            )

        # Step 8/9: Execute based on mode
        if self._config.execution_mode == ExecutionMode.DRY_RUN.value:
            return self._execute_dry_run(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                classifications=classifications,
                unused_classifications=unused_classifications,
                unused_column_names=unused_column_names,
                consumers=consumers,
                consumed_spice_bytes=consumed_spice_bytes,
                child_datasets=child_datasets,
            )
        else:
            # Held across the whole snapshot -> validate -> update -> ingest -> report
            # sequence, not just the write. Wrapping the call site rather than the
            # body means every one of the several early returns inside
            # _execute_apply releases the lock via the context manager's finally.
            try:
                with self._dataset_lock.held(dataset_id, execution_id=self._execution_id):
                    return self._execute_apply(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                        dataset_def=dataset_def,
                        classifications=classifications,
                        unused_classifications=unused_classifications,
                        unused_column_names=unused_column_names,
                        used_columns=used_columns,
                        consumers=consumers,
                        consumed_spice_bytes=consumed_spice_bytes,
                        child_datasets=child_datasets,
                    )
            except DatasetLockedError as exc:
                # Another run is mid-flight on this dataset. Refusing is correct:
                # proceeding would mean two describe-modify-write cycles racing, and
                # the loser's column removals would be silently discarded.
                logger.warning(
                    "Skipping dataset: another optimisation run holds its lock",
                    extra={
                        "dataset_id": dataset_id,
                        "holder": exc.holder,
                        "lock_expires_at": exc.expires_at,
                        "operation": "dataset_lock",
                    },
                )
                return DatasetResult(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    status="skipped",
                    total_columns=len(classifications),
                    used_columns_count=len(classifications)
                    - len(unused_classifications),
                    unused_columns_count=len(unused_classifications),
                    risk_indicators=self._get_risk_indicators(len(consumers)),
                    error_message=str(exc),
                )
            except DatasetLockError as exc:
                logger.error(
                    "Dataset lock unavailable, refusing to modify without it",
                    extra={"dataset_id": dataset_id, "error": str(exc)},
                )
                return DatasetResult(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    status="failed",
                    total_columns=len(classifications),
                    error_message=f"Dataset lock unavailable: {exc}",
                )

    def _execute_dry_run(
        self,
        dataset_id: str,
        dataset_name: str,
        classifications: list,
        unused_classifications: list,
        unused_column_names: list[str],
        consumers: list,
        consumed_spice_bytes: int | None = None,
        child_datasets: list | None = None,
    ) -> DatasetResult:
        """Execute dry run mode: generate report without calling write APIs.

        Args:
            dataset_id: Dataset identifier.
            dataset_name: Dataset display name.
            classifications: Full column classifications.
            unused_classifications: Unused column classifications.
            unused_column_names: Names of unused columns eligible for removal.
            consumers: List of discovered consumers.
            consumed_spice_bytes: Actual SPICE capacity, for savings estimation.
            child_datasets: Discovered child datasets, for report provenance.

        Returns:
            DatasetResult for the dry run.
        """
        logger.info(
            "Executing dry run",
            extra={
                "dataset_id": dataset_id,
                "unused_columns_count": len(unused_column_names),
                "operation": "dry_run",
            },
        )

        estimated_savings = self._reporter.estimate_spice_savings(
            unused_classifications,
            total_columns=len(classifications),
            consumed_spice_capacity_bytes=consumed_spice_bytes,
        )

        report = self._reporter.generate_report(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            execution_mode=ExecutionMode.DRY_RUN.value,
            classifications=classifications,
            consumer_count=len(consumers),
            safety_threshold_pct=self._config.safety_threshold_pct,
            threshold_breached=False,
            consumers=consumers,
            consumed_spice_capacity_bytes=consumed_spice_bytes,
            child_datasets=child_datasets,
        )
        report_key = self._reporter.store_report(report, dataset_id)

        # Publish summary notification
        self._notifier.publish_summary(
            dataset_id=dataset_id,
            columns_removed=len(unused_column_names),
            spice_savings_bytes=estimated_savings,
            execution_mode=ExecutionMode.DRY_RUN.value,
        )

        return DatasetResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            # Not "optimized". Nothing was optimised — this path describes the dataset
            # and writes a report, and the dataset it describes is untouched.
            status="scanned",
            total_columns=len(classifications),
            used_columns_count=len(classifications) - len(unused_classifications),
            unused_columns_count=len(unused_classifications),
            columns_affected=unused_column_names,
            estimated_spice_savings_bytes=estimated_savings,
            risk_indicators=self._get_risk_indicators(
                len(consumers), child_datasets=child_datasets
            ),
            report_s3_key=report_key,
        )

    def _execute_apply(
        self,
        dataset_id: str,
        dataset_name: str,
        dataset_def: dict,
        classifications: list,
        unused_classifications: list,
        unused_column_names: list[str],
        used_columns: set[str],
        consumers: list,
        consumed_spice_bytes: int | None = None,
        child_datasets: list | None = None,
    ) -> DatasetResult:
        """Execute apply mode: snapshot → validate → modify → ingest → report.

        Args:
            dataset_id: Dataset identifier.
            dataset_name: Dataset display name.
            dataset_def: Full dataset definition from DescribeDataSet.
            classifications: Full column classifications.
            unused_classifications: Unused column classifications.
            unused_column_names: Names of unused columns eligible for removal.
            used_columns: Set of all used column names (from consumers).
            consumers: List of discovered consumers.
            consumed_spice_bytes: Actual SPICE capacity, for savings estimation.
            child_datasets: Discovered child datasets, for report provenance.

        Returns:
            DatasetResult for the apply execution.
        """
        logger.info(
            "Executing apply mode",
            extra={
                "dataset_id": dataset_id,
                "unused_columns_count": len(unused_column_names),
                "operation": "apply_mode",
            },
        )

        # Step 9a: Capture snapshot before modification
        try:
            snapshot_key = self._snapshot_manager.capture_snapshot(dataset_id)
        except SnapshotError as exc:
            logger.error(
                "Snapshot capture failed, aborting optimization",
                extra={
                    "dataset_id": dataset_id,
                    "error": str(exc),
                    "operation": "snapshot_capture",
                },
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(classifications),
                error_message=f"Snapshot capture failed: {exc}",
            )

        logger.info(
            "Snapshot captured",
            extra={
                "dataset_id": dataset_id,
                "snapshot_key": snapshot_key,
                "operation": "snapshot_capture",
            },
        )

        # Step 9b: Pre-modification validation
        all_col_names = {
            col.get("Name", "") for col in dataset_def.get("OutputColumns", [])
            if col.get("Name")
        }
        retained_columns = all_col_names - set(unused_column_names)

        try:
            # Only validate against columns that exist in this dataset —
            # child consumer scanning may return column names from the child's
            # namespace that don't exist in the parent's OutputColumns.
            validatable_refs = used_columns & all_col_names
            self._modifier.validate_modification(
                retained_columns=retained_columns,
                consumer_references=validatable_refs,
            )
        except ValidationError as exc:
            logger.error(
                "Pre-modification validation failed",
                extra={
                    "dataset_id": dataset_id,
                    "missing_columns": sorted(exc.missing_columns),
                    "operation": "validation",
                },
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(classifications),
                error_message=f"Pre-modification validation failed: {exc}",
                snapshot_s3_key=snapshot_key,
            )

        # Step 9c: Build optimized definition
        optimized_def = self._modifier.build_optimized_definition(
            dataset_def=dataset_def,
            unused_columns=unused_column_names,
        )

        # Step 9d: Apply modification via UpdateDataSet
        try:
            update_response = self._modifier.apply_modification(dataset_id, optimized_def)
        except ModificationError as exc:
            logger.error(
                "Dataset modification failed",
                extra={
                    "dataset_id": dataset_id,
                    "snapshot_key": snapshot_key,
                    "error": str(exc),
                    "operation": "apply_modification",
                },
            )
            # Send error alert
            self._notifier.publish_error(
                dataset_id=dataset_id,
                error_message=f"UpdateDataSet failed: {exc}",
            )
            return DatasetResult(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                status="failed",
                total_columns=len(classifications),
                error_message=f"UpdateDataSet failed: {exc}",
                snapshot_s3_key=snapshot_key,
            )

        logger.info(
            "Dataset modified successfully",
            extra={
                "dataset_id": dataset_id,
                "operation": "apply_modification",
            },
        )

        # Step 9e: Trigger SPICE ingestion
        try:
            self._modifier.trigger_ingestion(dataset_id)
        except ModificationError as exc:
            if _refresh_is_unsupported(exc):
                # Not a failure. QuickSight cannot refresh a dataset built from an
                # uploaded file — it keeps no connection or location metadata to re-read
                # from — so CreateIngestion is rejected outright and there was never a
                # refresh to start. It re-reads the stored file itself when the
                # definition changes, which an apply has just done.
                #
                # This used to raise an SNS error alert on every apply against a
                # file-backed dataset. Harmless only because the topic has no
                # subscribers; the moment anyone sets NotificationEmail it becomes a
                # recurring alert for a non-event, and alerts that cry wolf stop being
                # read. The same case is already handled this way in execute_rollback.
                logger.info(
                    "No SPICE refresh started, and none available for this source type",
                    extra={
                        "dataset_id": dataset_id,
                        "reason": str(exc),
                        "refresh_unsupported": True,
                        "operation": "trigger_ingestion",
                    },
                )
            else:
                logger.warning(
                    "Ingestion trigger failed (dataset was modified successfully)",
                    exc_info=True,
                    extra={
                        "dataset_id": dataset_id,
                        "error": str(exc),
                        "operation": "trigger_ingestion",
                    },
                )
                # Non-fatal: dataset is modified, admin can trigger manually
                self._notifier.publish_error(
                    dataset_id=dataset_id,
                    error_message=(
                        f"CreateIngestion failed after successful UpdateDataSet: {exc}"
                    ),
                )
        else:
            # Only on the path where it actually was. This line was previously outside
            # the try entirely, so it claimed success immediately after the warning that
            # said the trigger had failed — two adjacent log lines contradicting each
            # other, which is worse than either one alone.
            logger.info(
                "SPICE ingestion triggered",
                extra={
                    "dataset_id": dataset_id,
                    "operation": "trigger_ingestion",
                },
            )

        # Step 9e-bis: Cascade the refresh to descendant datasets (opt-in).
        #
        # Removing columns from this parent and refreshing it leaves every child holding
        # SPICE data shaped to the old column set. Their next scheduled refresh — or a
        # transform that references a now-absent parent column — can then fail or show
        # errors. Refreshing the tree here, parents before children, re-resolves each
        # level against the new parent schema proactively.
        #
        # Off unless cascade_child_refresh is set, because it fans CreateIngestion out
        # across the whole dependency tree, which spends API quota and SPICE refresh
        # capacity. Never fatal: the parent change is already committed, so a child that
        # cannot be refreshed is reported, not raised.
        child_refresh_results: list = []
        cascade_requested = getattr(self._config, "cascade_child_refresh", False)
        if cascade_requested and child_datasets:
            child_refresh_results = self._modifier.trigger_cascading_refresh(
                child_datasets
            )
            refresh_failures = [r for r in child_refresh_results if r.error]
            if refresh_failures:
                # Surfaced as a non-fatal alert so an operator can refresh those
                # children by hand. Not publish_error per child — one summary line.
                self._notifier.publish_error(
                    dataset_id=dataset_id,
                    error_message=(
                        f"{len(refresh_failures)} of {len(child_refresh_results)} child "
                        f"dataset refresh(es) failed after the parent was optimised. "
                        f"The parent change is applied; refresh these children manually: "
                        f"{', '.join(r.dataset_id for r in refresh_failures)}"
                    ),
                )
            logger.info(
                "Cascading child refresh finished",
                extra={
                    "dataset_id": dataset_id,
                    "children_refreshed": sum(
                        1 for r in child_refresh_results if r.refresh_started
                    ),
                    "children_unsupported": sum(
                        1 for r in child_refresh_results if r.refresh_unsupported
                    ),
                    "children_failed": len(refresh_failures),
                    "operation": "cascade_refresh",
                },
            )
        elif cascade_requested:
            logger.info(
                "Cascading child refresh requested but no child datasets to refresh",
                extra={"dataset_id": dataset_id, "operation": "cascade_refresh"},
            )

        # Step 9f: Generate and store report
        estimated_savings = self._reporter.estimate_spice_savings(
            unused_classifications,
            total_columns=len(classifications),
            consumed_spice_capacity_bytes=consumed_spice_bytes,
        )

        report = self._reporter.generate_report(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            execution_mode=ExecutionMode.APPLY.value,
            classifications=classifications,
            consumer_count=len(consumers),
            safety_threshold_pct=self._config.safety_threshold_pct,
            threshold_breached=False,
            consumers=consumers,
            snapshot_s3_key=snapshot_key,
            update_response=update_response,
            consumed_spice_capacity_bytes=consumed_spice_bytes,
            child_datasets=child_datasets,
            child_refresh_results=child_refresh_results,
        )
        report_key = self._reporter.store_report(report, dataset_id)

        # Publish summary notification
        self._notifier.publish_summary(
            dataset_id=dataset_id,
            columns_removed=len(unused_column_names),
            spice_savings_bytes=estimated_savings,
            execution_mode=ExecutionMode.APPLY.value,
        )

        return DatasetResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            status="optimized",
            total_columns=len(classifications),
            used_columns_count=len(classifications) - len(unused_classifications),
            unused_columns_count=len(unused_classifications),
            columns_affected=unused_column_names,
            estimated_spice_savings_bytes=estimated_savings,
            risk_indicators=self._get_risk_indicators(
                len(consumers), child_datasets=child_datasets
            ),
            report_s3_key=report_key,
            snapshot_s3_key=snapshot_key,
            child_refresh_results=[r.to_dict() for r in child_refresh_results],
        )

    def emit_metric(self, metric_name: str, value: float, unit: str) -> None:
        """Emit a CloudWatch metric for the optimiser.

        Delegates to ``lambda_source.metrics`` so the namespace is defined once. This
        method used to hardcode its own, and the dispatcher hardcoded a third, while the
        IAM policy permitted a fourth — so every metric was silently denied.

        Failures are logged, never raised.

        Args:
            metric_name: Name of the metric (e.g., 'DatasetsProcessed').
            value: Metric value.
            unit: CloudWatch unit (e.g., 'Count', 'Bytes', 'Milliseconds').
        """
        emit_metric(
            metric_name,
            value,
            unit,
            dimensions={"FunctionName": METRIC_NAMESPACE},
            client=self._cloudwatch_client,
        )

    @staticmethod
    def _attribute_child_columns(
        attribution: UsageAttribution,
        child_datasets: list[ChildDataset],
        traced_refs: set[str],
    ) -> None:
        """Attribute child-protected columns to the specific child requiring them.

        ``scan_child_consumers`` returns one flat union, which is the right input
        for the removal decision but tells an operator nothing about which child to
        edit. Each child now retains the three categories separately, so this maps
        them onto the columns that actually survived into ``traced_refs``.

        The distinction is the actionable part. A column held by a child's
        ``ProjectStep`` can usually be freed by editing that child; one held by the
        child's join key cannot be freed without restructuring the join.

        Args:
            attribution: Collector to record into.
            child_datasets: Children discovered for the target dataset.
            traced_refs: The union actually merged into ``used_columns``. Categories
                are intersected with this so the report never claims protection for
                a column the pipeline did not in fact protect.
        """
        for child in child_datasets:
            requester = {
                "requester_name": child.name,
                "requester_id": child.dataset_id,
                "detail": f"depth {child.depth}",
            }

            attribution.record_many(
                child.join_key_columns & traced_refs,
                ColumnUsageSource.CHILD_JOIN_KEY.value,
                **requester,
            )
            attribution.record_many(
                child.consumer_traced_columns & traced_refs,
                ColumnUsageSource.CHILD_CONSUMER.value,
                **requester,
            )
            attribution.record_many(
                child.projected_columns & traced_refs,
                ColumnUsageSource.CHILD_PROJECTION.value,
                **requester,
            )

            if child.structural_fallback:
                # Consumer tracing could not run for this child, so its imported
                # columns are protected without proof that anything displays them.
                attribution.record_many(
                    child.referenced_columns & traced_refs,
                    ColumnUsageSource.CHILD_STRUCTURAL.value,
                    **requester,
                )

    @staticmethod
    def _extract_id_from_arn(arn: str, resource_type: str) -> str:
        """Extract the resource ID from a QuickSight ARN.

        ARN format: arn:aws:quicksight:{region}:{account}:{type}/{id}

        Args:
            arn: Full ARN string.
            resource_type: Expected resource type ('analysis' or 'dashboard').

        Returns:
            The extracted resource ID.
        """
        # ARN format: arn:aws:quicksight:region:account:resource_type/resource_id
        parts = arn.split("/")
        if len(parts) >= 2:
            return parts[-1]
        # Fallback: return the arn itself (shouldn't happen with valid ARNs)
        return arn

    @staticmethod
    def _get_risk_indicators(
        consumer_count: int, child_datasets: list | None = None
    ) -> list[str]:
        """Determine risk indicators for the DatasetResult summary.

        Kept in step with :meth:`ReportGenerator._get_risk_indicators` so the
        summary an operator sees in the tool response does not disagree with the
        stored report.

        Args:
            consumer_count: Number of consumers discovered.
            child_datasets: Discovered child datasets, if any.

        Returns:
            List of risk indicator strings.
        """
        indicators: list[str] = []
        if consumer_count > 1:
            indicators.append("shared dataset")
        elif consumer_count == 0:
            indicators.append("no consumers found")

        if child_datasets:
            indicators.append(f"{len(child_datasets)} child dataset(s) depend on this")
            if any(
                getattr(child, "structural_fallback", False)
                for child in child_datasets
            ):
                indicators.append(
                    "child dependency tracing incomplete: columns retained unproven"
                )

        return indicators


def _configure_logging() -> None:
    """Install structured JSON logging, whether or not a handler already exists.

    This function did nothing in the one environment it was written for. It was guarded
    by ``if not root_logger.handlers:``, and the Python Lambda runtime installs a root
    handler before any of our code runs — so the guard was always false, the level was
    never raised to INFO, and :class:`_JsonFormatter` was never installed.

    The cost was not a cosmetic one. Every ``logger.*(..., extra={...})`` call in this
    codebase writes its diagnostic detail into ``extra``, and the default Lambda
    formatter ignores it entirely. So the fields existed, were populated, and were
    discarded on the way out:

    - ``Unhandled exception before or during routing`` arrived with no traceback, which
      turned a one-line signature bug into an afternoon
    - ``Cannot release dataset lock`` arrived without the ``error`` field attached
      specifically to explain why, so the cause had to be established by IAM simulation
      instead of being read off the log line written to carry it
    - ``Dataset lock acquired`` and ``Dataset lock released`` never appeared at all,
      being INFO

    Reformats the existing handler rather than adding a second one, which would double
    every line. Respects ``AWS_LAMBDA_LOG_LEVEL`` if the function sets it, so Lambda's
    advanced logging controls still win — overriding a level someone deliberately
    configured would be its own small dishonesty.
    """
    root_logger = logging.getLogger()

    requested = os.environ.get("AWS_LAMBDA_LOG_LEVEL") or os.environ.get("LOG_LEVEL")
    level = logging.getLevelName((requested or "INFO").upper())
    # getLevelName returns the string "Level X" for anything it does not recognise, so a
    # typo in the env var must not silently disable logging.
    if not isinstance(level, int):
        level = logging.INFO
    root_logger.setLevel(level)

    formatter = _JsonFormatter()
    if root_logger.handlers:
        for existing in root_logger.handlers:
            existing.setFormatter(formatter)
    else:
        # Local runs and tests, where nothing has set logging up yet.
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root_logger.addHandler(stream)

    # Raising the root level to INFO also unmuted the AWS SDKs, which are chatty at INFO
    # — "Found credentials in environment variables" on every cold start, plus a
    # ``tenant_id`` attribute on every record. CloudWatch Logs bills on ingested volume,
    # so leaving that on would charge for noise and bury the lines worth reading. Their
    # warnings and errors still come through, and a throttle or a retry storm is
    # something we do want to see.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for Lambda/CloudWatch Logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Includes timestamp, level, message, logger name, and any extra fields.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # The Lambda runtime puts the request id on the record, and its own formatter is
        # what normally surfaces it. Since we replace that formatter, lift it to the top
        # level here — otherwise it would drop into ``details`` alongside the domain
        # fields, and the one identifier that ties a line to an invocation is worth more
        # visibility than that.
        request_id = getattr(record, "aws_request_id", None)
        if request_id:
            log_entry["request_id"] = request_id

        # Include extra fields passed via the 'extra' parameter
        # Standard LogRecord attributes to exclude
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
            # Promoted to the top level above, so not repeated inside details.
            "aws_request_id",
            # Injected onto every record by the Lambda runtime, not by us, and almost
            # always None. Left in, it added a null field to every line we emit — which
            # is billable volume for no information, and one more thing between a reader
            # and the fields that matter.
            "tenant_id",
        }
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extra_fields:
            log_entry["details"] = extra_fields

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def _build_response(result: ExecutionResult, duration_ms: float) -> dict:
    """Build the Lambda response dict from an ExecutionResult.

    Args:
        result: The completed execution result.
        duration_ms: Execution duration in milliseconds.

    Returns:
        Response dict suitable for Lambda return value.
    """
    # Field names describe what the run actually did. A dry run reports what it found;
    # only an apply reports what it removed and freed.
    #
    # This is the whole point of the fix. Previously both modes emitted
    # total_columns_removed and total_spice_freed_bytes, so a read-only scan came back
    # saying "8 columns removed, 147 MB freed" about a dataset it had not touched. The
    # only thing distinguishing the two was the job's separate top-level `mode` field,
    # which a reader — human or agent — has no particular reason to consult when the body
    # already states, in the past tense, that the work is done.
    is_dry_run = result.execution_mode == ExecutionMode.DRY_RUN.value
    count_key = "columns_identified_count" if is_dry_run else "columns_removed_count"
    total_count_key = "total_columns_identified" if is_dry_run else "total_columns_removed"
    savings_key = (
        "projected_spice_savings_bytes" if is_dry_run else "total_spice_freed_bytes"
    )

    dataset_summaries = []
    for dr in result.dataset_results:
        summary: dict[str, Any] = {
            "dataset_id": dr.dataset_id,
            "dataset_name": dr.dataset_name,
            "status": dr.status,
            "total_columns": dr.total_columns,
            count_key: len(dr.columns_affected),
            "estimated_spice_savings_bytes": dr.estimated_spice_savings_bytes,
        }
        if dr.error_message:
            summary["error_message"] = dr.error_message
        if dr.report_s3_key:
            summary["report_s3_key"] = dr.report_s3_key
        if dr.snapshot_s3_key:
            summary["snapshot_s3_key"] = dr.snapshot_s3_key
        # Only present in apply mode when a cascade actually ran. A count of children
        # refreshed is the headline; the per-child detail lives in the stored report.
        if dr.child_refresh_results:
            crr = dr.child_refresh_results
            summary["cascading_refresh"] = {
                "children_total": len(crr),
                "refresh_started_count": sum(
                    1 for r in crr if r.get("refresh_started")
                ),
                "refresh_unsupported_count": sum(
                    1 for r in crr if r.get("refresh_unsupported")
                ),
                "refresh_failed_count": sum(1 for r in crr if r.get("error")),
            }
        dataset_summaries.append(summary)

    response = {
        "status": "success" if result.datasets_failed == 0 else "partial_failure",
        "execution_id": result.execution_id,
        "timestamp": result.timestamp,
        "duration_ms": round(duration_ms, 2),
        # Echoed into the body, not left only on the job record, so any consumer of this
        # dict alone can tell which vocabulary it is reading.
        "execution_mode": result.execution_mode,
        "datasets_processed": result.datasets_processed,
        "datasets_failed": result.datasets_failed,
        total_count_key: result.total_columns_affected,
        savings_key: result.total_spice_freed_bytes,
        "dataset_results": dataset_summaries,
    }
    # Surfaced only when it happened, so a clean run is not cluttered by a zero.
    if result.datasets_blocked:
        response["datasets_blocked"] = result.datasets_blocked

    if is_dry_run:
        acted_on = result.datasets_processed - result.datasets_blocked
        if acted_on <= 0 and result.datasets_blocked:
            # Every dataset was refused, so there is nothing to apply. The old message
            # ended "Re-run with action='remove' to apply", which invited an action that
            # refuses identically — the threshold is a property of the dataset, not of
            # the mode you asked for.
            response["message"] = (
                f"Dry run: no dataset was modified. "
                f"{result.datasets_blocked} dataset(s) were refused by the safety "
                f"threshold because too large a share of their columns looked unused. "
                f"Removing nothing is the correct outcome here — re-running with "
                f"action='remove' would refuse again. Either raise "
                f"safety_threshold_pct deliberately, or work out why so many columns "
                f"appear unreferenced: a dataset whose consumers reference none of its "
                f"columns usually means the scan could not resolve them, not that the "
                f"columns are genuinely dead."
            )
        else:
            blocked_note = ""
            if result.datasets_blocked:
                blocked_note = (
                    f" A further {result.datasets_blocked} dataset(s) were refused by "
                    f"the safety threshold and are excluded from these figures."
                )
            response["message"] = (
                f"Dry run: no dataset was modified. Identified "
                f"{result.total_columns_affected} unused column(s) across "
                f"{acted_on} dataset(s), which would free about "
                f"{result.total_spice_freed_bytes} bytes if removed."
                f"{blocked_note} Re-run with action='remove' to apply."
            )
    return response


def _write_job_callback(job_id: str, table_name: str, result: dict) -> None:
    """Write execution result back to the Job Store DynamoDB table.

    Called when the Lambda is invoked asynchronously via the MCP server with
    callback_job_id and callback_table in the event. Writes the final status
    and result summary to the job record.

    Uses a conditional write to prevent writing to non-existent jobs. Errors
    are logged but never raised — callback failures must not crash the Lambda.

    Args:
        job_id: The job ID to update in the Job Store.
        table_name: Name of the DynamoDB Job Store table.
        result: The Lambda execution result dict.
    """
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(table_name)

        # Determine status based on the execution result
        status = "completed" if result.get("status") != "error" else "failed"

        # Build a compact callback result with key metrics.
        #
        # Copies whichever vocabulary _build_response used rather than reaching for
        # apply-mode keys and defaulting them to 0. Asking for "total_columns_removed" on
        # a dry-run response used to succeed with the dry-run number in it; asking for it
        # now misses, and silently reporting 0 columns for a scan that found 8 would just
        # be a quieter version of the same lie.
        mode = result.get("execution_mode", "")
        is_dry_run = mode == ExecutionMode.DRY_RUN.value
        count_key = "total_columns_identified" if is_dry_run else "total_columns_removed"
        savings_key = (
            "projected_spice_savings_bytes" if is_dry_run else "total_spice_freed_bytes"
        )
        per_dataset_count_key = (
            "columns_identified_count" if is_dry_run else "columns_removed_count"
        )

        callback_result: dict[str, Any] = {
            "execution_mode": mode,
            "datasets_processed": result.get("datasets_processed", 0),
            count_key: result.get(count_key, 0),
            savings_key: result.get(savings_key, 0),
            "status": result.get("status", "unknown"),
        }
        if result.get("message"):
            callback_result["message"] = result["message"]
        # Copied explicitly, because this callback builds a chosen subset rather than
        # forwarding the response. A field added to _build_response and not added here is
        # simply absent from what the agent reads - which is how the first live check of
        # the blocked count came back as None despite the response carrying it.
        if result.get("datasets_blocked"):
            callback_result["datasets_blocked"] = result["datasets_blocked"]

        # Include report_s3_key from the first dataset result if available
        dataset_results = result.get("dataset_results", [])
        if dataset_results:
            first_report_key = dataset_results[0].get("report_s3_key")
            if first_report_key:
                callback_result["report_s3_key"] = first_report_key

            # Include compact per-dataset outcomes so callers (e.g. the MCP
            # chat agent) can explain a 0-column result: distinguishes
            # optimized / threshold_breached / skipped / failed.
            callback_result["dataset_outcomes"] = [
                {
                    "dataset_id": dr.get("dataset_id", ""),
                    "status": dr.get("status", "unknown"),
                    per_dataset_count_key: dr.get(per_dataset_count_key, 0),
                }
                for dr in dataset_results
            ]

        # Include error details if the execution failed
        if status == "failed":
            callback_result["error"] = result.get("error", "Unknown error")

        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :status, #r = :result, #u = :updated_at",
            ExpressionAttributeNames={
                "#s": "status",
                "#r": "result",
                "#u": "updated_at",
            },
            ExpressionAttributeValues={
                ":status": status,
                ":result": callback_result,
                ":updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ConditionExpression="attribute_exists(job_id)",
        )

        logger.info(
            "Job callback written successfully",
            extra={"job_id": job_id, "status": status},
        )

    except Exception as exc:
        # Check if this is a ConditionalCheckFailedException (job doesn't exist)
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if error_code == "ConditionalCheckFailedException":
            logger.warning(
                "Job callback skipped: job_id does not exist in Job Store",
                extra={"job_id": job_id, "table_name": table_name},
            )
        else:
            logger.warning(
                "Job callback write failed",
                extra={
                    "job_id": job_id,
                    "table_name": table_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
