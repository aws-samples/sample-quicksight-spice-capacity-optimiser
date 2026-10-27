# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the Consumer Discovery module.

Tests correctness properties related to consumer discovery, including
dataset matching and DELETED consumer exclusion.

Feature: spice-dataset-optimiser
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from lambda_source.discovery import ConsumerDiscovery
from lambda_source.qs_client import QSClient
from tests.property.strategies import (
    QS_CONSUMER_STATUSES,
    st_analysis_definition,
    st_arn,
    st_dataset_id,
    st_dataset_identifier,
)


# ---------------------------------------------------------------------------
# Property 1: Definition-dataset matching correctness
# ---------------------------------------------------------------------------


class TestDefinitionDatasetMatchingCorrectness:
    """Property 1: Definition-dataset matching correctness.

    For any consumer definition (analysis or dashboard) and any target dataset ID,
    the discovery logic SHALL return the definition as a match if and only if the
    target dataset ID appears within the definition's DataSetIdentifierDeclarations list.

    Feature: spice-dataset-optimiser, Property 1: Definition-dataset matching correctness
    **Validates: Requirements 1.1, 1.2**
    """

    @given(
        target_dataset_id=st_dataset_id(),
        definition=st_analysis_definition(),
    )
    @settings(max_examples=100)
    def test_matches_when_dataset_id_in_declarations(
        self,
        target_dataset_id: str,
        definition: dict,
    ) -> None:
        """When the target dataset_id IS in DataSetIdentifierDeclarations (in a
        DataSetArn), _matches_dataset should return (True, identifier).

        **Validates: Requirements 1.1, 1.2**
        """
        # Arrange: Inject target_dataset_id into the definition's first declaration
        identifier = definition["DataSetIdentifierDeclarations"][0]["Identifier"]
        definition["DataSetIdentifierDeclarations"][0]["DataSetArn"] = (
            f"arn:aws:quicksight:us-east-1:123456789012:dataset/{target_dataset_id}"
        )

        # Act
        match, returned_identifier = ConsumerDiscovery._matches_dataset(
            definition, target_dataset_id
        )

        # Assert: Should match and return the correct identifier
        assert match is True
        assert returned_identifier == identifier

    @given(
        target_dataset_id=st_dataset_id(),
        other_dataset_id=st_dataset_id(),
        definition=st_analysis_definition(),
    )
    @settings(max_examples=100)
    def test_no_match_when_dataset_id_not_in_declarations(
        self,
        target_dataset_id: str,
        other_dataset_id: str,
        definition: dict,
    ) -> None:
        """When the target dataset_id IS NOT in DataSetIdentifierDeclarations,
        _matches_dataset should return (False, "").

        **Validates: Requirements 1.1, 1.2**
        """
        # Skip if IDs happen to be the same or one is a substring of the other
        if target_dataset_id == other_dataset_id:
            return
        if target_dataset_id in other_dataset_id:
            return

        # Replace all DataSetArns so none contain the target
        for decl in definition["DataSetIdentifierDeclarations"]:
            decl["DataSetArn"] = (
                f"arn:aws:quicksight:us-east-1:123456789012:dataset/{other_dataset_id}"
            )

        # Act
        match, returned_identifier = ConsumerDiscovery._matches_dataset(
            definition, target_dataset_id
        )

        # Assert: Should NOT match
        assert match is False
        assert returned_identifier == ""

    @given(
        target_dataset_id=st_dataset_id(),
    )
    @settings(max_examples=100)
    def test_empty_declarations_returns_no_match(
        self,
        target_dataset_id: str,
    ) -> None:
        """When DataSetIdentifierDeclarations is empty, _matches_dataset
        should return (False, "").

        **Validates: Requirements 1.1, 1.2**
        """
        # Arrange: Definition with no declarations
        definition: dict = {"DataSetIdentifierDeclarations": []}

        # Act
        match, returned_identifier = ConsumerDiscovery._matches_dataset(
            definition, target_dataset_id
        )

        # Assert
        assert match is False
        assert returned_identifier == ""

    @given(
        target_dataset_id=st_dataset_id(),
        dataset_identifier=st_dataset_identifier(),
    )
    @settings(max_examples=100)
    def test_match_returns_correct_identifier_from_declaration(
        self,
        target_dataset_id: str,
        dataset_identifier: str,
    ) -> None:
        """The returned identifier must correspond to the declaration entry
        that contains the matching DataSetArn.

        **Validates: Requirements 1.1, 1.2**
        """
        # Arrange: Build a definition with the target in a specific declaration
        definition: dict = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": dataset_identifier,
                    "DataSetArn": f"arn:aws:quicksight:us-east-1:123456789012:dataset/{target_dataset_id}",
                }
            ]
        }

        # Act
        match, returned_identifier = ConsumerDiscovery._matches_dataset(
            definition, target_dataset_id
        )

        # Assert
        assert match is True
        assert returned_identifier == dataset_identifier


# ---------------------------------------------------------------------------
# Strategies for consumer summaries
# ---------------------------------------------------------------------------


@st.composite
def st_analysis_summary(draw: st.DrawFn, status: str | None = None) -> dict:
    """Generate an analysis summary dict as returned by ListAnalyses."""
    return {
        "AnalysisId": draw(st_dataset_id()),
        "Arn": draw(st_arn(resource_type="analysis")),
        "Name": f"Analysis_{draw(st.integers(min_value=1, max_value=9999))}",
        "Status": status if status is not None else draw(st.sampled_from(QS_CONSUMER_STATUSES)),
    }


@st.composite
def st_dashboard_summary(draw: st.DrawFn, status: str | None = None) -> dict:
    """Generate a dashboard summary dict as returned by ListDashboards."""
    return {
        "DashboardId": draw(st_dataset_id()),
        "Arn": draw(st_arn(resource_type="dashboard")),
        "Name": f"Dashboard_{draw(st.integers(min_value=1, max_value=9999))}",
        "Status": status if status is not None else draw(st.sampled_from(QS_CONSUMER_STATUSES)),
    }


# ---------------------------------------------------------------------------
# Property 2: DELETED consumer exclusion
# ---------------------------------------------------------------------------


class TestDeletedConsumerExclusion:
    """Property 2: DELETED consumer exclusion.

    For any list of discovered consumers with arbitrary status values,
    filtering SHALL produce a result containing zero consumers with status
    DELETED and all consumers with non-DELETED status preserved.

    **Validates: Requirements 1.3**
    """

    @given(
        analysis_summaries=st.lists(st_analysis_summary(), min_size=0, max_size=10),
        dashboard_summaries=st.lists(st_dashboard_summary(), min_size=0, max_size=10),
        target_dataset_id=st_dataset_id(),
        target_ds_identifier=st_dataset_identifier(),
    )
    @settings(max_examples=100)
    def test_no_deleted_consumers_in_results(
        self,
        analysis_summaries: list[dict],
        dashboard_summaries: list[dict],
        target_dataset_id: str,
        target_ds_identifier: str,
    ) -> None:
        """No consumer with DELETED status shall appear in discovery results.

        Feature: spice-dataset-optimiser, Property 2: DELETED consumer exclusion
        **Validates: Requirements 1.3**
        """
        # Build a definition that matches the target dataset
        matching_definition = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": target_ds_identifier,
                    "DataSetArn": f"arn:aws:quicksight:us-east-1:123456789012:dataset/{target_dataset_id}",
                }
            ]
        }

        # Mock the QSClient
        mock_client = MagicMock(spec=QSClient)
        mock_client.list_analyses.return_value = analysis_summaries
        mock_client.list_dashboards.return_value = dashboard_summaries

        # All describe calls return the matching definition
        mock_client.describe_analysis_definition.return_value = matching_definition
        mock_client.describe_dashboard_definition.return_value = matching_definition

        discovery = ConsumerDiscovery(qs_client=mock_client, account_id="123456789012")
        consumers = discovery.discover_consumers(target_dataset_id)

        # PROPERTY: No consumer in results should have DELETED status
        for consumer in consumers:
            assert consumer.status != "DELETED", (
                f"Consumer with ARN {consumer.arn} has DELETED status but should have been excluded"
            )

    @given(
        analysis_summaries=st.lists(st_analysis_summary(), min_size=0, max_size=10),
        dashboard_summaries=st.lists(st_dashboard_summary(), min_size=0, max_size=10),
        target_dataset_id=st_dataset_id(),
        target_ds_identifier=st_dataset_identifier(),
    )
    @settings(max_examples=100)
    def test_all_non_deleted_consumers_preserved(
        self,
        analysis_summaries: list[dict],
        dashboard_summaries: list[dict],
        target_dataset_id: str,
        target_ds_identifier: str,
    ) -> None:
        """All consumers with non-DELETED status that match the dataset shall be preserved.

        Feature: spice-dataset-optimiser, Property 2: DELETED consumer exclusion
        **Validates: Requirements 1.3**
        """
        # Build a definition that matches the target dataset
        matching_definition = {
            "DataSetIdentifierDeclarations": [
                {
                    "Identifier": target_ds_identifier,
                    "DataSetArn": f"arn:aws:quicksight:us-east-1:123456789012:dataset/{target_dataset_id}",
                }
            ]
        }

        # Mock the QSClient
        mock_client = MagicMock(spec=QSClient)
        mock_client.list_analyses.return_value = analysis_summaries
        mock_client.list_dashboards.return_value = dashboard_summaries

        # All describe calls return the matching definition
        mock_client.describe_analysis_definition.return_value = matching_definition
        mock_client.describe_dashboard_definition.return_value = matching_definition

        discovery = ConsumerDiscovery(qs_client=mock_client, account_id="123456789012")
        consumers = discovery.discover_consumers(target_dataset_id)

        # Count expected non-DELETED consumers (those that match the dataset)
        non_deleted_analyses = [
            s for s in analysis_summaries if s.get("Status") != "DELETED"
        ]
        non_deleted_dashboards = [
            s for s in dashboard_summaries if s.get("Status") != "DELETED"
        ]

        expected_count = len(non_deleted_analyses) + len(non_deleted_dashboards)

        # PROPERTY: All non-DELETED consumers that match the dataset should be in results
        assert len(consumers) == expected_count, (
            f"Expected {expected_count} consumers (non-DELETED that match dataset), "
            f"but got {len(consumers)}. "
            f"Analyses: {len(non_deleted_analyses)} non-DELETED, "
            f"Dashboards: {len(non_deleted_dashboards)} non-DELETED"
        )

        # Verify all result ARNs come from non-DELETED summaries
        result_arns = {c.arn for c in consumers}
        expected_arns = {s["Arn"] for s in non_deleted_analyses} | {
            s["Arn"] for s in non_deleted_dashboards
        }
        assert result_arns == expected_arns, (
            f"Result ARNs do not match expected non-DELETED ARNs. "
            f"Missing: {expected_arns - result_arns}, "
            f"Extra: {result_arns - expected_arns}"
        )
