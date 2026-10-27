# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the consumer discovery module.

Tests pagination handling, error handling when DescribeAnalysisDefinition fails,
correct association of dataset identifiers, and DELETED status filtering.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lambda_source.discovery import (
    ConsumerDiscovery,
    ConsumerDiscoveryError,
    DiscoveryError,
)
from lambda_source.qs_client import (
    AccessDeniedError,
    QSClientError,
    ResourceNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TARGET_DATASET_ID = "abc12345-1234-5678-abcd-123456789012"
ACCOUNT_ID = "123456789012"


def _make_analysis_summary(analysis_id: str, name: str, status: str = "CREATION_SUCCESSFUL") -> dict:
    """Create a mock analysis summary dict."""
    return {
        "AnalysisId": analysis_id,
        "Arn": f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:analysis/{analysis_id}",
        "Name": name,
        "Status": status,
    }


def _make_dashboard_summary(dashboard_id: str, name: str, status: str = "CREATION_SUCCESSFUL") -> dict:
    """Create a mock dashboard summary dict."""
    return {
        "DashboardId": dashboard_id,
        "Arn": f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:dashboard/{dashboard_id}",
        "Name": name,
        "Status": status,
    }


def _make_definition_with_dataset(dataset_id: str, identifier: str = "ds_1") -> dict:
    """Create a definition that references the given dataset_id."""
    return {
        "DataSetIdentifierDeclarations": [
            {
                "Identifier": identifier,
                "DataSetArn": f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:dataset/{dataset_id}",
            }
        ]
    }


def _make_definition_without_dataset() -> dict:
    """Create a definition that does NOT reference the target dataset."""
    return {
        "DataSetIdentifierDeclarations": [
            {
                "Identifier": "other_ds",
                "DataSetArn": f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:dataset/other-dataset-id",
            }
        ]
    }


def _create_mock_qs_client(
    analyses: list[dict] | None = None,
    dashboards: list[dict] | None = None,
    analysis_definitions: dict[str, dict] | None = None,
    dashboard_definitions: dict[str, dict] | None = None,
    analysis_errors: dict[str, Exception] | None = None,
    dashboard_errors: dict[str, Exception] | None = None,
) -> MagicMock:
    """Create a mock QSClient."""
    mock_client = MagicMock()
    mock_client.list_analyses.return_value = analyses or []
    mock_client.list_dashboards.return_value = dashboards or []

    analysis_defs = analysis_definitions or {}
    dashboard_defs = dashboard_definitions or {}
    a_errors = analysis_errors or {}
    d_errors = dashboard_errors or {}

    def describe_analysis_def(analysis_id):
        if analysis_id in a_errors:
            raise a_errors[analysis_id]
        return analysis_defs.get(analysis_id, {"DataSetIdentifierDeclarations": []})

    def describe_dashboard_def(dashboard_id):
        if dashboard_id in d_errors:
            raise d_errors[dashboard_id]
        return dashboard_defs.get(dashboard_id, {"DataSetIdentifierDeclarations": []})

    mock_client.describe_analysis_definition.side_effect = describe_analysis_def
    mock_client.describe_dashboard_definition.side_effect = describe_dashboard_def

    return mock_client


# ---------------------------------------------------------------------------
# Tests: Basic discovery
# ---------------------------------------------------------------------------


class TestDiscoverConsumers:
    """Tests for ConsumerDiscovery.discover_consumers."""

    def test_discovers_analysis_referencing_dataset(self):
        """An analysis that references the target dataset is included."""
        mock_client = _create_mock_qs_client(
            analyses=[_make_analysis_summary("a-1", "Sales Analysis")],
            analysis_definitions={"a-1": _make_definition_with_dataset(TARGET_DATASET_ID, "my_ds")},
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 1
        assert consumers[0].arn == f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:analysis/a-1"
        assert consumers[0].consumer_type == "analysis"
        assert consumers[0].name == "Sales Analysis"
        assert consumers[0].dataset_identifier == "my_ds"
        assert consumers[0].status == "CREATION_SUCCESSFUL"

    def test_discovers_dashboard_referencing_dataset(self):
        """A dashboard that references the target dataset is included."""
        mock_client = _create_mock_qs_client(
            dashboards=[_make_dashboard_summary("d-1", "Revenue Dashboard")],
            dashboard_definitions={"d-1": _make_definition_with_dataset(TARGET_DATASET_ID, "revenue_ds")},
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 1
        assert consumers[0].arn == f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:dashboard/d-1"
        assert consumers[0].consumer_type == "dashboard"
        assert consumers[0].name == "Revenue Dashboard"
        assert consumers[0].dataset_identifier == "revenue_ds"

    def test_excludes_analysis_not_referencing_dataset(self):
        """An analysis that does not reference the target dataset is excluded."""
        mock_client = _create_mock_qs_client(
            analyses=[_make_analysis_summary("a-1", "Other Analysis")],
            analysis_definitions={"a-1": _make_definition_without_dataset()},
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 0

    def test_discovers_multiple_consumers(self):
        """Multiple analyses and dashboards are correctly discovered."""
        mock_client = _create_mock_qs_client(
            analyses=[
                _make_analysis_summary("a-1", "Analysis 1"),
                _make_analysis_summary("a-2", "Analysis 2"),
            ],
            dashboards=[
                _make_dashboard_summary("d-1", "Dashboard 1"),
            ],
            analysis_definitions={
                "a-1": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_a"),
                "a-2": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_b"),
            },
            dashboard_definitions={
                "d-1": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_c"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 3
        types = {c.consumer_type for c in consumers}
        assert types == {"analysis", "dashboard"}

    def test_no_consumers_found_returns_empty_list(self):
        """Returns empty list when no consumers reference the dataset."""
        mock_client = _create_mock_qs_client()

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert consumers == []


# ---------------------------------------------------------------------------
# Tests: DELETED status filtering (Req 1.3)
# ---------------------------------------------------------------------------


class TestDeletedStatusFiltering:
    """Tests for filtering out DELETED consumers."""

    def test_excludes_deleted_analysis(self):
        """Analyses with DELETED status are excluded."""
        mock_client = _create_mock_qs_client(
            analyses=[
                _make_analysis_summary("a-1", "Deleted Analysis", status="DELETED"),
                _make_analysis_summary("a-2", "Active Analysis", status="CREATION_SUCCESSFUL"),
            ],
            analysis_definitions={
                "a-2": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_active"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 1
        assert consumers[0].name == "Active Analysis"
        # Ensure describe was not called for the deleted analysis
        mock_client.describe_analysis_definition.assert_called_once_with("a-2")

    def test_excludes_deleted_dashboard(self):
        """Dashboards with DELETED status are excluded."""
        mock_client = _create_mock_qs_client(
            dashboards=[
                _make_dashboard_summary("d-1", "Deleted Dashboard", status="DELETED"),
            ],
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert len(consumers) == 0
        mock_client.describe_dashboard_definition.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Error handling (Req 1.5)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for halting on unreachable consumer definitions."""

    def test_halts_on_access_denied_for_analysis(self):
        """Raises ConsumerDiscoveryError when analysis definition is access denied."""
        mock_client = _create_mock_qs_client(
            analyses=[_make_analysis_summary("a-1", "Restricted Analysis")],
            analysis_errors={
                "a-1": AccessDeniedError(
                    "Access denied",
                    error_code="AccessDeniedException",
                    api_name="describe_analysis_definition",
                )
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)

        with pytest.raises(ConsumerDiscoveryError) as exc_info:
            discovery.discover_consumers(TARGET_DATASET_ID)

        err = exc_info.value
        assert len(err.unreachable_consumers) == 1
        assert err.unreachable_consumers[0].error_code == "AccessDeniedException"
        assert "a-1" in err.unreachable_consumers[0].arn

    def test_halts_on_access_denied_for_dashboard(self):
        """Raises ConsumerDiscoveryError when dashboard definition is access denied."""
        mock_client = _create_mock_qs_client(
            dashboards=[_make_dashboard_summary("d-1", "Restricted Dashboard")],
            dashboard_errors={
                "d-1": AccessDeniedError(
                    "Access denied",
                    error_code="AccessDeniedException",
                    api_name="describe_dashboard_definition",
                )
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)

        with pytest.raises(ConsumerDiscoveryError) as exc_info:
            discovery.discover_consumers(TARGET_DATASET_ID)

        err = exc_info.value
        assert len(err.unreachable_consumers) == 1
        assert err.unreachable_consumers[0].error_code == "AccessDeniedException"

    def test_halts_with_multiple_unreachable_consumers(self):
        """Error includes all unreachable consumers, not just the first."""
        mock_client = _create_mock_qs_client(
            analyses=[
                _make_analysis_summary("a-1", "Restricted 1"),
                _make_analysis_summary("a-2", "Restricted 2"),
            ],
            analysis_errors={
                "a-1": AccessDeniedError(
                    "Access denied", error_code="AccessDeniedException", api_name="test"
                ),
                "a-2": QSClientError(
                    "Unknown error", error_code="InternalFailure", api_name="test"
                ),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)

        with pytest.raises(ConsumerDiscoveryError) as exc_info:
            discovery.discover_consumers(TARGET_DATASET_ID)

        err = exc_info.value
        assert len(err.unreachable_consumers) == 2
        error_codes = {e.error_code for e in err.unreachable_consumers}
        assert "AccessDeniedException" in error_codes
        assert "InternalFailure" in error_codes

    def test_resource_not_found_skipped_not_halted(self):
        """ResourceNotFoundError is skipped (race between list and describe)."""
        mock_client = _create_mock_qs_client(
            analyses=[
                _make_analysis_summary("a-1", "Gone Analysis"),
                _make_analysis_summary("a-2", "Active Analysis"),
            ],
            analysis_errors={
                "a-1": ResourceNotFoundError(
                    "Not found", error_code="ResourceNotFoundException", api_name="test"
                ),
            },
            analysis_definitions={
                "a-2": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_1"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        # Should not raise, just skip the not-found one
        assert len(consumers) == 1
        assert consumers[0].name == "Active Analysis"


# ---------------------------------------------------------------------------
# Tests: Pagination via QSClient delegation (Req 1.1, 1.2)
# ---------------------------------------------------------------------------


class TestPaginationHandling:
    """Tests verifying discovery correctly uses QSClient pagination."""

    def test_calls_list_analyses_for_pagination(self):
        """Verifies list_analyses is called to retrieve all pages."""
        mock_client = _create_mock_qs_client(
            analyses=[
                _make_analysis_summary("a-1", "Analysis Page 1"),
                _make_analysis_summary("a-2", "Analysis Page 2"),
                _make_analysis_summary("a-3", "Analysis Page 3"),
            ],
            analysis_definitions={
                "a-1": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_1"),
                "a-2": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_2"),
                "a-3": _make_definition_without_dataset(),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        # QSClient.list_analyses handles pagination internally; verify it was called
        mock_client.list_analyses.assert_called_once()
        # Two out of three analyses reference the target dataset
        assert len(consumers) == 2
        names = {c.name for c in consumers}
        assert names == {"Analysis Page 1", "Analysis Page 2"}

    def test_calls_list_dashboards_for_pagination(self):
        """Verifies list_dashboards is called to retrieve all pages."""
        mock_client = _create_mock_qs_client(
            dashboards=[
                _make_dashboard_summary("d-1", "Dashboard Page 1"),
                _make_dashboard_summary("d-2", "Dashboard Page 2"),
            ],
            dashboard_definitions={
                "d-1": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_a"),
                "d-2": _make_definition_with_dataset(TARGET_DATASET_ID, "ds_b"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        # QSClient.list_dashboards handles pagination internally; verify it was called
        mock_client.list_dashboards.assert_called_once()
        assert len(consumers) == 2

    def test_processes_large_result_set_from_paginated_list(self):
        """Simulates a large paginated result set returned by QSClient."""
        # Simulate many analyses as if returned across multiple pages
        analyses = [_make_analysis_summary(f"a-{i}", f"Analysis {i}") for i in range(50)]
        definitions = {
            f"a-{i}": _make_definition_with_dataset(TARGET_DATASET_ID, f"ds_{i}")
            if i % 2 == 0
            else _make_definition_without_dataset()
            for i in range(50)
        }

        mock_client = _create_mock_qs_client(
            analyses=analyses,
            analysis_definitions=definitions,
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        # Even-indexed analyses (0, 2, 4, ..., 48) reference the target dataset = 25 items
        assert len(consumers) == 25
        mock_client.list_analyses.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Dataset matching (_matches_dataset)
# ---------------------------------------------------------------------------


class TestMatchesDataset:
    """Tests for the _matches_dataset static method."""

    def test_matches_when_dataset_id_in_arn(self):
        """Returns True when dataset ID is found in DataSetArn."""
        definition = _make_definition_with_dataset(TARGET_DATASET_ID, "my_identifier")

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is True
        assert identifier == "my_identifier"

    def test_no_match_when_different_dataset(self):
        """Returns False when definition references a different dataset."""
        definition = _make_definition_without_dataset()

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is False
        assert identifier == ""

    def test_no_match_when_declarations_empty(self):
        """Returns False when DataSetIdentifierDeclarations is empty."""
        definition = {"DataSetIdentifierDeclarations": []}

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is False
        assert identifier == ""

    def test_no_match_when_declarations_missing(self):
        """Returns False when DataSetIdentifierDeclarations key is absent."""
        definition = {}

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is False
        assert identifier == ""

    def test_matches_first_matching_declaration(self):
        """When multiple declarations exist, returns the first match."""
        definition = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": "other_ds",
                    "DataSetArn": "arn:aws:quicksight:us-east-1:123:dataset/other-id",
                },
                {
                    "Identifier": "target_ds",
                    "DataSetArn": f"arn:aws:quicksight:us-east-1:123:dataset/{TARGET_DATASET_ID}",
                },
            ]
        }

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is True
        assert identifier == "target_ds"

    def test_no_match_on_substring_superset_arn(self):
        """A dataset whose ID has the target as a prefix must NOT match.

        Guards against substring false positives, e.g. target 'prod-sales'
        must not match an ARN for 'prod-sales-backup'.
        """
        definition = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": "backup_ds",
                    "DataSetArn": (
                        "arn:aws:quicksight:us-east-1:123456789012:"
                        f"dataset/{TARGET_DATASET_ID}-backup"
                    ),
                }
            ]
        }

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is False
        assert identifier == ""

    def test_no_match_when_target_is_suffix_of_other_id(self):
        """The target appearing only as a suffix of another ID must NOT match."""
        definition = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": "other_ds",
                    "DataSetArn": (
                        "arn:aws:quicksight:us-east-1:123456789012:"
                        f"dataset/prefix-{TARGET_DATASET_ID}"
                    ),
                }
            ]
        }

        match, identifier = ConsumerDiscovery._matches_dataset(definition, TARGET_DATASET_ID)

        assert match is False
        assert identifier == ""


# ---------------------------------------------------------------------------
# Tests: Association (Req 1.4)
# ---------------------------------------------------------------------------


class TestConsumerAssociation:
    """Tests that consumers are correctly associated with dataset identifier and ARN."""

    def test_consumer_has_correct_arn(self):
        """Consumer ARN matches the analysis/dashboard summary ARN."""
        expected_arn = f"arn:aws:quicksight:us-east-1:{ACCOUNT_ID}:analysis/a-123"
        mock_client = _create_mock_qs_client(
            analyses=[{
                "AnalysisId": "a-123",
                "Arn": expected_arn,
                "Name": "Test",
                "Status": "CREATION_SUCCESSFUL",
            }],
            analysis_definitions={
                "a-123": _make_definition_with_dataset(TARGET_DATASET_ID, "identifier_x"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert consumers[0].arn == expected_arn

    def test_consumer_has_correct_dataset_identifier(self):
        """Consumer dataset_identifier matches the definition's Identifier field."""
        mock_client = _create_mock_qs_client(
            dashboards=[_make_dashboard_summary("d-99", "My Dashboard")],
            dashboard_definitions={
                "d-99": _make_definition_with_dataset(TARGET_DATASET_ID, "custom_identifier_42"),
            },
        )

        discovery = ConsumerDiscovery(mock_client, ACCOUNT_ID)
        consumers = discovery.discover_consumers(TARGET_DATASET_ID)

        assert consumers[0].dataset_identifier == "custom_identifier_42"
