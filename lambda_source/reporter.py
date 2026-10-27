# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Optimisation report generation and storage for the SPICE Dataset Optimiser.

Generates structured Optimisation_Report JSON documents containing column analysis
results, SPICE savings estimates, risk indicators, and execution metadata. Reports
are stored to S3 with the key pattern: {prefix}reports/{execution-date}/{dataset_id}.json
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambda_source.models import ColumnClassification, ColumnUsageSource, Consumer

logger = logging.getLogger(__name__)

# Default SPICE size estimates per data type (bytes per row)
# These are approximate averages for columnar storage in SPICE
DEFAULT_SPICE_BYTES_PER_ROW = {
    "STRING": 100,
    "INTEGER": 8,
    "DECIMAL": 16,
    "DATETIME": 8,
}

# Default assumption for row count when estimating savings
DEFAULT_AVERAGE_ROWS = 1_000_000

# Usage sources that mean "the analysis could not rule this column out", as
# opposed to "something demonstrably needs it". A column held only by these is
# retained on the safe side of an unknown, and the report should say so.
_UNPROVEN_SOURCES = frozenset(
    {
        ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value,
        ColumnUsageSource.CHILD_REFS_UNKNOWN.value,
        ColumnUsageSource.CHILD_STRUCTURAL.value,
        ColumnUsageSource.UNREACHABLE_CONSUMER.value,
    }
)


class ReportGenerator:
    """Generates and stores Optimisation Reports for dataset analysis results."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        s3_client: Any | None = None,
        average_rows: int = DEFAULT_AVERAGE_ROWS,
    ):
        """Initialize ReportGenerator.

        Args:
            s3_bucket: S3 bucket name for storing reports.
            s3_prefix: S3 key prefix for all stored objects.
            s3_client: Optional pre-configured boto3 S3 client (for testing).
            average_rows: Average row count to use for SPICE savings estimation.
        """
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._s3_client = s3_client or boto3.client("s3")
        self._average_rows = average_rows

    def generate_report(
        self,
        dataset_id: str,
        dataset_name: str,
        execution_mode: str,
        classifications: list[ColumnClassification],
        consumer_count: int,
        safety_threshold_pct: float,
        threshold_breached: bool,
        consumers: list[Consumer],
        snapshot_s3_key: str | None = None,
        update_response: dict | None = None,
        consumed_spice_capacity_bytes: int | None = None,
        child_datasets: list[Any] | None = None,
        child_refresh_results: list[Any] | None = None,
    ) -> dict:
        """Generate a complete Optimisation_Report dict.

        Args:
            dataset_id: QuickSight dataset identifier.
            dataset_name: Display name of the dataset.
            execution_mode: 'dry_run' or 'apply'.
            classifications: List of column classification results.
            consumer_count: Number of consumers found for this dataset.
            safety_threshold_pct: Configured safety threshold (0.0-1.0).
            threshold_breached: Whether the safety threshold was breached.
            consumers: List of Consumer objects for inclusion in the report.
            snapshot_s3_key: S3 key of the dataset snapshot (apply mode only).
            update_response: UpdateDataSet API response dict (apply mode only).
            consumed_spice_capacity_bytes: Actual SPICE capacity from the API.
            child_datasets: Discovered ChildDataset objects. Previously these were
                logged and then dropped, so a report could not show that a column
                was protected by a downstream dataset — or which one.
            child_refresh_results: Per-child cascading-refresh outcomes (apply mode,
                when cascade_child_refresh is on). Each is either a ChildRefreshResult
                (with a ``to_dict``) or an already-serialised dict. Summarised into a
                ``cascading_refresh`` block so a report records which descendants were
                refreshed, which have no refreshable source, and which failed.

        Returns:
            Complete Optimisation_Report dict with all required fields.
        """
        execution_id = str(uuid.uuid4())
        execution_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Count columns by status
        used_count = sum(
            1 for c in classifications if c.status in ("used", "join_key")
        )
        unused_count = sum(1 for c in classifications if c.status == "unused")
        join_key_count = sum(1 for c in classifications if c.status == "join_key")
        total_columns = len(classifications)

        # Get unused columns for savings estimation
        unused_columns = [c for c in classifications if c.status == "unused"]
        estimated_savings = self.estimate_spice_savings(
            unused_columns,
            total_columns=total_columns,
            consumed_spice_capacity_bytes=consumed_spice_capacity_bytes,
        )

        # Determine risk indicators
        risk_indicators = self._get_risk_indicators(
            consumer_count,
            classifications=classifications,
            child_datasets=child_datasets,
        )

        # Build column details
        column_details = [
            self._build_column_detail(c) for c in classifications
        ]

        # Build child dataset summary
        child_dataset_list = self._build_child_summary(child_datasets)

        # Summarise the cascading refresh, if one ran
        cascading_refresh = self._build_cascade_summary(child_refresh_results)

        # Build consumer list for report
        consumer_list = [
            {
                "arn": consumer.arn,
                "type": consumer.consumer_type,
                "name": consumer.name,
            }
            for consumer in consumers
        ]

        # Build the report
        report: dict[str, Any] = {
            "execution_id": execution_id,
            "execution_timestamp": execution_timestamp,
            "execution_mode": execution_mode,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "total_columns": total_columns,
            "used_columns_count": used_count,
            "unused_columns_count": unused_count,
            "join_key_columns_count": join_key_count,
            "estimated_spice_savings_bytes": estimated_savings,
            "safety_threshold_pct": safety_threshold_pct,
            "threshold_breached": threshold_breached,
            "risk_indicators": risk_indicators,
            "consumers": consumer_list,
            "child_datasets": child_dataset_list,
            "cascading_refresh": cascading_refresh,
            "column_details": column_details,
            "snapshot_s3_key": snapshot_s3_key if execution_mode == "apply" else None,
            "update_api_response": update_response if execution_mode == "apply" else None,
        }

        logger.info(
            "Generated optimisation report",
            extra={
                "dataset_id": dataset_id,
                "execution_id": execution_id,
                "execution_mode": execution_mode,
                "total_columns": total_columns,
                "unused_columns_count": unused_count,
                "estimated_spice_savings_bytes": estimated_savings,
            },
        )

        return report

    def store_report(self, report: dict, dataset_id: str) -> str:
        """Store report to S3.

        Args:
            report: The complete Optimisation_Report dict.
            dataset_id: QuickSight dataset identifier.

        Returns:
            The S3 key where the report was stored.

        Raises:
            ClientError: If S3 put operation fails.
        """
        s3_key = self._generate_s3_key(dataset_id)

        try:
            report_body = json.dumps(report, default=str, indent=2)
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=s3_key,
                Body=report_body,
                ContentType="application/json",
            )
        except ClientError as exc:
            logger.error(
                "Failed to store optimisation report to S3",
                extra={
                    "dataset_id": dataset_id,
                    "s3_bucket": self._s3_bucket,
                    "s3_key": s3_key,
                    "error": str(exc),
                },
            )
            raise

        logger.info(
            "Optimisation report stored successfully",
            extra={
                "dataset_id": dataset_id,
                "s3_bucket": self._s3_bucket,
                "s3_key": s3_key,
            },
        )

        return s3_key

    def estimate_spice_savings(
        self,
        unused_columns: list[ColumnClassification],
        total_columns: int = 0,
        consumed_spice_capacity_bytes: int | None = None,
    ) -> int:
        """Estimate SPICE savings in bytes for removing unused columns.

        When actual SPICE capacity is available (from DescribeDataSet API's
        ConsumedSpiceCapacityInBytes field), uses proportional estimation:
            savings = actual_bytes × (unused_columns / total_columns)

        Falls back to the heuristic-based estimation (bytes_per_row × average_rows)
        only when actual capacity data is not available.

        Args:
            unused_columns: List of unused column classifications.
            total_columns: Total number of columns in the dataset (required for
                proportional estimation).
            consumed_spice_capacity_bytes: Actual SPICE capacity consumed by the
                dataset in bytes (from QuickSight API). If provided and > 0,
                proportional estimation is used.

        Returns:
            Estimated savings in bytes (non-negative integer).
        """
        if not unused_columns:
            return 0

        # Prefer proportional estimation when actual SPICE capacity is known
        if consumed_spice_capacity_bytes and consumed_spice_capacity_bytes > 0 and total_columns > 0:
            proportion = len(unused_columns) / total_columns
            return int(consumed_spice_capacity_bytes * proportion)

        # Fallback: heuristic-based estimation using data types and assumed row count
        total_savings = 0
        for column in unused_columns:
            data_type = column.data_type.upper()
            bytes_per_row = DEFAULT_SPICE_BYTES_PER_ROW.get(data_type, 100)
            total_savings += bytes_per_row * self._average_rows

        return total_savings

    def _build_column_detail(self, classification: ColumnClassification) -> dict:
        """Render one column's classification for the report.

        ``reason`` stays a single joined string for backward compatibility.
        ``required_by`` is the structured form, which is what makes the result
        actionable — it names the specific dashboard or child dataset rather than
        stating only that something, somewhere, referenced the column.
        """
        detail: dict[str, Any] = {
            "column_name": classification.column_name,
            "data_type": classification.data_type,
            "status": classification.status,
            "reason": (
                "; ".join(classification.reasons)
                if classification.reasons
                else self._default_reason(classification.status)
            ),
        }

        if classification.attributions:
            detail["required_by"] = [
                {
                    "source": a.source,
                    "name": a.requester_name,
                    "id": a.requester_id,
                    "detail": a.detail,
                }
                for a in classification.attributions
            ]
            sources = {a.source for a in classification.attributions}
            # Flagged separately because "nothing proved this is needed, but we
            # could not prove it is not" warrants a different response from a
            # column that is genuinely in use.
            detail["retained_without_proof"] = bool(
                sources and sources <= _UNPROVEN_SOURCES
            )

        return detail

    @staticmethod
    def _build_child_summary(child_datasets: list[Any] | None) -> list[dict]:
        """Summarise child datasets and what each one requires from the parent.

        Lets an operator act on a blocked optimisation: if a parent column is only
        held by one child's projection, editing that child frees it.
        """
        summary: list[dict] = []
        for child in child_datasets or []:
            summary.append(
                {
                    "dataset_id": getattr(child, "dataset_id", ""),
                    "name": getattr(child, "name", ""),
                    "depth": getattr(child, "depth", 0),
                    "required_columns": sorted(
                        getattr(child, "referenced_columns", set()) or set()
                    ),
                    "join_key_columns": sorted(
                        getattr(child, "join_key_columns", set()) or set()
                    ),
                    "consumer_traced_columns": sorted(
                        getattr(child, "consumer_traced_columns", set()) or set()
                    ),
                    "projected_columns": sorted(
                        getattr(child, "projected_columns", set()) or set()
                    ),
                    "structural_fallback": bool(
                        getattr(child, "structural_fallback", False)
                    ),
                }
            )
        return summary

    @staticmethod
    def _build_cascade_summary(child_refresh_results: list[Any] | None) -> dict:
        """Summarise a cascading child refresh for the report.

        Accepts either ChildRefreshResult objects (with ``to_dict``) or already-plain
        dicts, so it works whether called from the pipeline or reconstructed from a
        stored result. ``attempted`` is False when no cascade ran, which a reader should
        not confuse with "ran and refreshed nothing".
        """
        results = child_refresh_results or []
        normalised: list[dict] = []
        for r in results:
            if hasattr(r, "to_dict"):
                normalised.append(r.to_dict())
            elif isinstance(r, dict):
                normalised.append(r)

        return {
            "attempted": bool(normalised),
            "children_total": len(normalised),
            "refresh_started_count": sum(
                1 for r in normalised if r.get("refresh_started")
            ),
            "refresh_unsupported_count": sum(
                1 for r in normalised if r.get("refresh_unsupported")
            ),
            "refresh_failed_count": sum(1 for r in normalised if r.get("error")),
            "results": normalised,
        }

    def _get_risk_indicators(
        self,
        consumer_count: int,
        classifications: list[ColumnClassification] | None = None,
        child_datasets: list[Any] | None = None,
    ) -> list[str]:
        """Determine risk indicators for the dataset.

        Consumer-count behaviour is unchanged. The additional indicators surface
        conditions that previously appeared only in CloudWatch logs, the most
        important being that a conservative fallback retained columns — a run where
        the analysis could not complete used to be indistinguishable in the report
        from one where every column was genuinely in use.

        Args:
            consumer_count: Number of consumers discovered for the dataset.
            classifications: Column classifications, for fallback detection.
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

        sources = {
            a.source
            for c in classifications or []
            for a in c.attributions
        }
        if ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value in sources:
            indicators.append(
                "calculated field could not be parsed: all columns retained"
            )
        if ColumnUsageSource.CHILD_REFS_UNKNOWN.value in sources:
            indicators.append(
                "child dependencies undeterminable: all columns retained"
            )
        if ColumnUsageSource.UNREACHABLE_CONSUMER.value in sources:
            # Deliberately worded to be unmistakable for "no consumers found". The two
            # were previously indistinguishable and the remediation differs entirely:
            # this one is a permissions fix, not an optimisation opportunity.
            indicators.append(
                "a consumer could not be read: analysis incomplete, all columns "
                "retained. Check permissions before trusting this result"
            )

        return indicators

    def _generate_s3_key(self, dataset_id: str) -> str:
        """Generate S3 key with pattern: {prefix}reports/{execution-date}/{dataset_id}.json.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The generated S3 key string.
        """
        execution_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prefix = self._s3_prefix
        # Ensure prefix ends with '/' if non-empty
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        return f"{prefix}reports/{execution_date}/{dataset_id}.json"

    @staticmethod
    def _default_reason(status: str) -> str:
        """Provide a default reason string based on column status.

        Args:
            status: Column classification status.

        Returns:
            Human-readable default reason string.
        """
        if status == "unused":
            return "no consumer references"
        elif status == "join_key":
            return "retained: join key"
        elif status == "used":
            return "referenced by consumer"
        return "unknown"
