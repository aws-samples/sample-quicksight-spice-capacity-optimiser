# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Consumer discovery module for the SPICE Dataset Optimiser.

Discovers all QuickSight analyses and dashboards that reference a target dataset
by listing all consumers, fetching their definitions, and matching against
DataSetIdentifierDeclarations.

Implements Requirements 1.1–1.5:
- List all analyses/dashboards with pagination
- Match target dataset via DataSetIdentifierDeclarations
- Exclude DELETED consumers
- Associate each consumer with dataset identifier and ARN
- Halt processing on unreachable definitions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lambda_source.models import Consumer
from lambda_source.qs_client import (
    QSClient,
    QSClientError,
    AccessDeniedError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryError:
    """Represents an error encountered while retrieving a consumer definition."""

    arn: str
    """ARN of the unreachable consumer."""

    error_code: str
    """The error code received from the API."""

    message: str
    """Human-readable error description."""


class ConsumerDiscoveryError(Exception):
    """Raised when consumer discovery must halt due to unreachable definitions."""

    def __init__(self, message: str, unreachable_consumers: list[DiscoveryError]):
        super().__init__(message)
        self.unreachable_consumers = unreachable_consumers


class ConsumerDiscovery:
    """Discovers all analyses and dashboards referencing a given dataset.

    Uses the QSClient to list and describe consumers, matching them
    against the target dataset by inspecting DataSetIdentifierDeclarations
    in each definition.
    """

    def __init__(self, qs_client: QSClient, account_id: str):
        """Initialize ConsumerDiscovery.

        Args:
            qs_client: Pre-configured QSClient instance for API calls.
            account_id: AWS account ID for constructing ARNs.
        """
        self._qs_client = qs_client
        self._account_id = account_id

    def discover_consumers(
        self,
        dataset_id: str,
        *,
        skip_unreachable: bool = False,
        unreachable: list[DiscoveryError] | None = None,
    ) -> list[Consumer]:
        """Find all analyses and dashboards referencing the given dataset.

        Calls ListAnalyses and ListDashboards with pagination (handled by QSClient),
        then inspects each consumer's definition for a DataSetIdentifierDeclaration
        matching the target dataset_id.

        Consumers with DELETED status are excluded from results.

        If any consumer definition cannot be retrieved (due to access denied or
        other errors), processing is halted and a ConsumerDiscoveryError is raised,
        unless skip_unreachable is True.

        On the ``skip_unreachable=True`` path the caller **must** pass ``unreachable``
        and act on it. Previously that path logged a warning and returned
        ``list[Consumer]`` with the unreadable entries simply missing, giving the caller
        no way to tell a dataset whose only dashboard returned ``AccessDenied`` from a
        dataset that genuinely has no dashboards. The consequence ran in the dangerous
        direction: the unreadable dashboard's columns were never scanned, so they were
        classified unused and became removal candidates. Removing them breaks the
        dashboard nobody could read.

        This docstring already promised the opposite — "columns found only in
        unreachable consumers will be conservatively marked as used" — and nothing
        implemented it. ``handler.process_dataset`` now does, based on this list.

        Args:
            dataset_id: The QuickSight dataset ID to search for.
            skip_unreachable: If True, skip unreachable consumers instead of halting.
            unreachable: Optional accumulator. When supplied, every consumer that could
                not be read is appended to it, whether or not the run halts. Follows the
                same out-parameter shape ``ChildDatasetDiscovery`` uses internally.

        Returns:
            List of Consumer objects referencing the target dataset.

        Raises:
            ConsumerDiscoveryError: If any consumer definition is unreachable
                and skip_unreachable is False.
        """
        errors: list[DiscoveryError] = []
        consumers: list[Consumer] = []

        # Discover analyses
        analyses = self._list_analyses_paginated()
        for analysis_summary in analyses:
            status = analysis_summary.get("Status", "")
            if status == "DELETED":
                continue

            analysis_id = analysis_summary.get("AnalysisId", "")
            arn = analysis_summary.get("Arn", "")
            name = analysis_summary.get("Name", "")

            try:
                definition = self._qs_client.describe_analysis_definition(analysis_id)
            except ResourceNotFoundError:
                # Resource was deleted between list and describe; skip silently
                logger.warning(
                    "Analysis not found during definition retrieval, skipping",
                    extra={"analysis_id": analysis_id, "arn": arn},
                )
                continue
            except (AccessDeniedError, QSClientError) as exc:
                error_code = getattr(exc, "error_code", "Unknown") or "Unknown"
                errors.append(DiscoveryError(
                    arn=arn,
                    error_code=error_code,
                    message=str(exc),
                ))
                logger.error(
                    "Cannot retrieve analysis definition",
                    extra={
                        "analysis_id": analysis_id,
                        "arn": arn,
                        "error_code": error_code,
                    },
                )
                continue

            match, dataset_identifier = self._matches_dataset(definition, dataset_id)
            if match:
                consumers.append(Consumer(
                    arn=arn,
                    consumer_type="analysis",
                    name=name,
                    dataset_identifier=dataset_identifier,
                    status=status,
                ))

        # Discover dashboards
        dashboards = self._list_dashboards_paginated()
        for dashboard_summary in dashboards:
            status = dashboard_summary.get("Status", "")
            if status == "DELETED":
                continue

            dashboard_id = dashboard_summary.get("DashboardId", "")
            arn = dashboard_summary.get("Arn", "")
            name = dashboard_summary.get("Name", "")

            try:
                definition = self._qs_client.describe_dashboard_definition(dashboard_id)
            except ResourceNotFoundError:
                # Resource was deleted between list and describe; skip silently
                logger.warning(
                    "Dashboard not found during definition retrieval, skipping",
                    extra={"dashboard_id": dashboard_id, "arn": arn},
                )
                continue
            except (AccessDeniedError, QSClientError) as exc:
                error_code = getattr(exc, "error_code", "Unknown") or "Unknown"
                errors.append(DiscoveryError(
                    arn=arn,
                    error_code=error_code,
                    message=str(exc),
                ))
                logger.error(
                    "Cannot retrieve dashboard definition",
                    extra={
                        "dashboard_id": dashboard_id,
                        "arn": arn,
                        "error_code": error_code,
                    },
                )
                continue

            match, dataset_identifier = self._matches_dataset(definition, dataset_id)
            if match:
                consumers.append(Consumer(
                    arn=arn,
                    consumer_type="dashboard",
                    name=name,
                    dataset_identifier=dataset_identifier,
                    status=status,
                ))

        # Surface unreachable consumers to the caller regardless of which path is
        # taken. Populated before the raise so the accumulator is usable even when the
        # caller catches ConsumerDiscoveryError.
        if unreachable is not None:
            unreachable.extend(errors)

        # Halt processing if any definitions were unreachable (Req 1.5)
        if errors:
            if skip_unreachable:
                logger.warning(
                    "Skipping %d unreachable consumer(s) as requested",
                    len(errors),
                    extra={
                        "unreachable_count": len(errors),
                        "unreachable_arns": [e.arn for e in errors],
                    },
                )
            else:
                raise ConsumerDiscoveryError(
                    message=(
                        f"Cannot proceed: {len(errors)} consumer definition(s) unreachable. "
                        f"ARNs: {[e.arn for e in errors]}"
                    ),
                    unreachable_consumers=errors,
                )

        logger.info(
            "Consumer discovery complete",
            extra={
                "dataset_id": dataset_id,
                "analysis_count": sum(1 for c in consumers if c.consumer_type == "analysis"),
                "dashboard_count": sum(1 for c in consumers if c.consumer_type == "dashboard"),
            },
        )

        return consumers

    def _list_analyses_paginated(self) -> list[dict]:
        """Return all analyses using QSClient pagination.

        Returns:
            List of analysis summary dicts from the QuickSight API.
        """
        return self._qs_client.list_analyses()

    def _list_dashboards_paginated(self) -> list[dict]:
        """Return all dashboards using QSClient pagination.

        Returns:
            List of dashboard summary dicts from the QuickSight API.
        """
        return self._qs_client.list_dashboards()

    @staticmethod
    def _matches_dataset(definition: dict, dataset_id: str) -> tuple[bool, str]:
        """Check if a definition references the target dataset_id.

        Inspects the DataSetIdentifierDeclarations list in the definition
        for an entry whose DataSetArn contains the dataset_id or whose
        DataSetId matches directly.

        Args:
            definition: The analysis or dashboard definition dict.
            dataset_id: The target dataset ID to match against.

        Returns:
            Tuple of (match_found, dataset_identifier). If no match,
            dataset_identifier is an empty string.
        """
        declarations = definition.get("DataSetIdentifierDeclarations", [])

        for declaration in declarations:
            decl_dataset_arn = declaration.get("DataSetArn", "")
            decl_identifier = declaration.get("Identifier", "")

            # Match on the dataset ID at the ARN resource boundary, not as a
            # substring. A plain `dataset_id in arn` check causes false
            # positives (e.g. "prod-sales" matching "prod-sales-backup"),
            # which would associate the wrong consumers and incorrectly mark
            # columns as used.
            # ARN format: arn:aws:quicksight:{region}:{account}:dataset/{dataset_id}
            if (
                decl_dataset_arn.endswith(f"/{dataset_id}")
                or f"/{dataset_id}/" in decl_dataset_arn
            ):
                return True, decl_identifier

        return False, ""
