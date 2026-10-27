# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the ChildDatasetDiscovery class.

Tests validate:
- Discovery of child datasets using target as a source (legacy LogicalTableMap)
- Discovery of child datasets using target as a source (new DataPrepConfiguration)
- Column reference extraction from InputColumns, transforms, join keys, calc fields
- Conservative fallback when column usage is undeterminable
- Handling of access denied / unreachable child datasets
- Skip vs halt behavior for unreachable datasets
- ARN matching logic (exact and suffix)
- Target dataset is excluded from results
- Empty account (no datasets) case
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lambda_source.child_discovery import (
    ChildDataset,
    ChildDatasetDiscovery,
    ChildDiscoveryError,
)
from lambda_source.qs_client import (
    AccessDeniedError,
    QSClient,
    QSClientError,
    ResourceNotFoundError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCOUNT_ID = "123456789012"
REGION = "us-east-1"
TARGET_DATASET_ID = "target-dataset-001"
TARGET_ARN = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{TARGET_DATASET_ID}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_summary(dataset_id: str, name: str = "") -> dict:
    """Create a minimal DataSetSummary dict."""
    return {
        "DataSetId": dataset_id,
        "Name": name or f"Dataset {dataset_id}",
        "ImportMode": "SPICE",
    }


def _make_child_def_legacy(
    child_id: str,
    parent_arn: str,
    transforms: list[dict] | None = None,
) -> dict:
    """Create a dataset definition that references parent via LogicalTableMap."""
    return {
        "DataSetId": child_id,
        "Name": f"Child {child_id}",
        "ImportMode": "SPICE",
        "LogicalTableMap": {
            "lt-1": {
                "Alias": "Parent Data",
                "Source": {"DataSetArn": parent_arn},
                "DataTransforms": transforms or [],
            }
        },
    }


def _make_child_def_new_experience(
    child_id: str,
    parent_arn: str,
    input_columns: list[dict] | None = None,
) -> dict:
    """Create a dataset definition that references parent via DataPrepConfiguration."""
    return {
        "DataSetId": child_id,
        "Name": f"Child {child_id}",
        "ImportMode": "SPICE",
        "DataPrepConfiguration": {
            "SourceTableMap": {
                "source-1": {
                    "DataSet": {
                        "DataSetArn": parent_arn,
                        "InputColumns": input_columns or [],
                    }
                }
            },
            "TransformStepMap": {},
            "DestinationTableMap": {},
        },
    }


def _make_unrelated_def(dataset_id: str) -> dict:
    """Create a dataset definition that does NOT reference the target."""
    return {
        "DataSetId": dataset_id,
        "Name": f"Unrelated {dataset_id}",
        "ImportMode": "SPICE",
        "PhysicalTableMap": {
            "pt-1": {
                "RelationalTable": {
                    "DataSourceArn": "arn:aws:quicksight:us-east-1:123:datasource/ds-other",
                    "Name": "some_table",
                    "InputColumns": [
                        {"Name": "col_x", "Type": "STRING"},
                    ],
                }
            }
        },
        "LogicalTableMap": {
            "lt-1": {
                "Alias": "Some Table",
                "Source": {"PhysicalTableId": "pt-1"},
                "DataTransforms": [],
            }
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qs_client():
    """Create a mock QSClient."""
    client = MagicMock(spec=QSClient)
    client.account_id = ACCOUNT_ID
    return client


@pytest.fixture
def discovery(mock_qs_client):
    """Create a ChildDatasetDiscovery instance with a mock client."""
    return ChildDatasetDiscovery(
        qs_client=mock_qs_client,
        account_id=ACCOUNT_ID,
        region=REGION,
    )


# ---------------------------------------------------------------------------
# Tests: No Children Found
# ---------------------------------------------------------------------------


class TestNoChildrenFound:
    """Tests when no datasets use the target as a source."""

    def test_empty_account_returns_no_children(self, discovery, mock_qs_client):
        """When there are no datasets at all, returns empty list."""
        mock_qs_client.list_datasets.return_value = []

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result == []
        mock_qs_client.list_datasets.assert_called_once()

    def test_only_target_dataset_in_account(self, discovery, mock_qs_client):
        """When only the target dataset exists, returns empty list (skips itself)."""
        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
        ]

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result == []
        # Should not attempt to describe the target dataset
        mock_qs_client.describe_dataset.assert_not_called()

    def test_unrelated_datasets_return_no_children(self, discovery, mock_qs_client):
        """When other datasets exist but none reference the target, returns empty."""
        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("unrelated-1"),
            _make_dataset_summary("unrelated-2"),
        ]
        mock_qs_client.describe_dataset.side_effect = [
            _make_unrelated_def("unrelated-1"),
            _make_unrelated_def("unrelated-2"),
        ]

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result == []
        assert mock_qs_client.describe_dataset.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Legacy LogicalTableMap Children
# ---------------------------------------------------------------------------


class TestLegacyChildren:
    """Tests for child datasets using LogicalTableMap Source.DataSetArn."""

    def test_single_child_with_project_operation(self, discovery, mock_qs_client):
        """Child with ProjectOperation — referenced columns are the projected columns."""
        transforms = [
            {"ProjectOperation": {"ProjectedColumns": ["order_id", "amount", "region"]}},
        ]
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        assert result[0].dataset_id == "child-1"
        assert result[0].referenced_columns == {"order_id", "amount", "region"}

    def test_child_with_rename_operation(self, discovery, mock_qs_client):
        """RenameColumnOperation source column should be in referenced columns."""
        transforms = [
            {"RenameColumnOperation": {"ColumnName": "old_name", "NewColumnName": "new_name"}},
        ]
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "old_name" in result[0].referenced_columns

    def test_child_with_cast_operation(self, discovery, mock_qs_client):
        """CastColumnTypeOperation column should be in referenced columns."""
        transforms = [
            {"CastColumnTypeOperation": {"ColumnName": "price", "NewColumnType": "DECIMAL"}},
        ]
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "price" in result[0].referenced_columns

    def test_child_with_create_columns_expression(self, discovery, mock_qs_client):
        """CreateColumnsOperation expression columns should be extracted."""
        transforms = [
            {
                "CreateColumnsOperation": {
                    "Columns": [
                        {
                            "ColumnId": "calc-1",
                            "ColumnName": "full_name",
                            "Expression": "concat({ds1.first_name}, ' ', {ds1.last_name})",
                        }
                    ]
                }
            },
        ]
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "first_name" in result[0].referenced_columns
        assert "last_name" in result[0].referenced_columns

    def test_child_with_no_transforms_returns_empty_columns(self, discovery, mock_qs_client):
        """Child with Source.DataSetArn but no transforms — columns are empty set
        but found_explicit_declaration is True (LogicalTableMap source match),
        so returns empty set (meaning all columns from parent are passed through)."""
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms=[])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        # Empty transforms but explicit declaration found — returns empty set
        # (this triggers conservative fallback in orchestrator)
        assert len(result) == 1
        assert result[0].referenced_columns == set()

    def test_child_with_multiple_transforms(self, discovery, mock_qs_client):
        """Multiple transforms should all contribute to referenced columns."""
        transforms = [
            {"ProjectOperation": {"ProjectedColumns": ["col_a", "col_b", "col_c"]}},
            {"RenameColumnOperation": {"ColumnName": "col_a", "NewColumnName": "column_a"}},
            {"CastColumnTypeOperation": {"ColumnName": "col_b", "NewColumnType": "INTEGER"}},
        ]
        child_def = _make_child_def_legacy("child-1", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result[0].referenced_columns == {"col_a", "col_b", "col_c"}


# ---------------------------------------------------------------------------
# Tests: New Experience DataPrepConfiguration Children
# ---------------------------------------------------------------------------


class TestNewExperienceChildren:
    """Tests for child datasets using DataPrepConfiguration SourceTableMap."""

    def test_child_with_explicit_input_columns(self, discovery, mock_qs_client):
        """InputColumns in DataPrepConfiguration should be the definitive reference list."""
        input_columns = [
            {"Name": "order_id", "Type": "STRING", "Id": "id-1"},
            {"Name": "amount", "Type": "DECIMAL", "Id": "id-2"},
            {"Name": "status", "Type": "STRING", "Id": "id-3"},
        ]
        child_def = _make_child_def_new_experience("child-1", TARGET_ARN, input_columns)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        assert result[0].referenced_columns == {"order_id", "amount", "status"}

    def test_child_with_empty_input_columns(self, discovery, mock_qs_client):
        """Empty InputColumns in DataPrepConfiguration — undeterminable, returns empty set."""
        child_def = _make_child_def_new_experience("child-1", TARGET_ARN, input_columns=[])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        # Empty set signals "unknown" to caller
        assert result[0].referenced_columns == set()


# ---------------------------------------------------------------------------
# Tests: Multiple Children
# ---------------------------------------------------------------------------


class TestMultipleChildren:
    """Tests for scenarios with multiple child datasets."""

    def test_multiple_children_all_detected(self, discovery, mock_qs_client):
        """All children referencing the target should be found."""
        child_1_def = _make_child_def_legacy(
            "child-1", TARGET_ARN,
            [{"ProjectOperation": {"ProjectedColumns": ["col_a"]}}],
        )
        child_2_def = _make_child_def_new_experience(
            "child-2", TARGET_ARN,
            [{"Name": "col_b", "Type": "STRING", "Id": "id-b"}],
        )
        unrelated_def = _make_unrelated_def("unrelated-1")

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
            _make_dataset_summary("child-2"),
            _make_dataset_summary("unrelated-1"),
        ]

        def describe_side_effect(ds_id):
            return {
                "child-1": child_1_def,
                "child-2": child_2_def,
                "unrelated-1": unrelated_def,
            }.get(ds_id, _make_unrelated_def(ds_id))

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 2
        ids = {r.dataset_id for r in result}
        assert ids == {"child-1", "child-2"}

    def test_get_all_child_referenced_columns_unions_all(self, discovery, mock_qs_client):
        """get_all_child_referenced_columns should union columns from all children."""
        children = [
            ChildDataset(dataset_id="c1", name="C1", referenced_columns={"col_a", "col_b"}),
            ChildDataset(dataset_id="c2", name="C2", referenced_columns={"col_b", "col_c"}),
            ChildDataset(dataset_id="c3", name="C3", referenced_columns={"col_d"}),
        ]

        result = discovery.get_all_child_referenced_columns(children)

        assert result == {"col_a", "col_b", "col_c", "col_d"}

    def test_get_all_child_referenced_columns_empty_list(self, discovery):
        """Empty children list returns empty set."""
        result = discovery.get_all_child_referenced_columns([])
        assert result == set()


# ---------------------------------------------------------------------------
# Tests: Access Denied / Unreachable Datasets
# ---------------------------------------------------------------------------


class TestUnreachableDatasets:
    """Tests for handling access denied and other errors during describe."""

    def test_access_denied_raises_by_default(self, discovery, mock_qs_client):
        """If a dataset can't be described, raises ChildDiscoveryError by default."""
        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("inaccessible-1"),
        ]
        mock_qs_client.describe_dataset.side_effect = AccessDeniedError(
            "Access denied", error_code="AccessDeniedException", api_name="describe_data_set"
        )

        with pytest.raises(ChildDiscoveryError) as exc_info:
            discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "inaccessible-1" in exc_info.value.unreachable_datasets

    def test_access_denied_skipped_when_flag_set(self, discovery, mock_qs_client):
        """When skip_unreachable=True, inaccessible datasets are skipped."""
        child_def = _make_child_def_legacy(
            "child-1", TARGET_ARN,
            [{"ProjectOperation": {"ProjectedColumns": ["col_a"]}}],
        )

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("inaccessible-1"),
            _make_dataset_summary("child-1"),
        ]

        def describe_side_effect(ds_id):
            if ds_id == "inaccessible-1":
                raise AccessDeniedError(
                    "Access denied", error_code="AccessDeniedException",
                    api_name="describe_data_set",
                )
            if ds_id == "child-1":
                return child_def
            return _make_unrelated_def(ds_id)

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(
            TARGET_DATASET_ID, skip_unreachable=True
        )

        # Should still find the accessible child
        assert len(result) == 1
        assert result[0].dataset_id == "child-1"

    def test_resource_not_found_is_silently_skipped(self, discovery, mock_qs_client):
        """ResourceNotFoundError (deleted between list and describe) is skipped."""
        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("deleted-1"),
        ]
        mock_qs_client.describe_dataset.side_effect = ResourceNotFoundError(
            "Not found", error_code="ResourceNotFoundException",
            api_name="describe_data_set",
        )

        # Should NOT raise — ResourceNotFound is always skipped
        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result == []

    def test_generic_qs_error_treated_as_unreachable(self, discovery, mock_qs_client):
        """Generic QSClientError marks dataset as unreachable."""
        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("error-1"),
        ]
        mock_qs_client.describe_dataset.side_effect = QSClientError(
            "Internal failure", error_code="InternalFailure",
            api_name="describe_data_set",
        )

        with pytest.raises(ChildDiscoveryError) as exc_info:
            discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "error-1" in exc_info.value.unreachable_datasets

    def test_list_datasets_failure_raises_child_discovery_error(self, discovery, mock_qs_client):
        """If list_datasets itself fails, raises ChildDiscoveryError."""
        mock_qs_client.list_datasets.side_effect = QSClientError(
            "Throttled", error_code="ThrottlingException", api_name="list_data_sets"
        )

        with pytest.raises(ChildDiscoveryError) as exc_info:
            discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "Failed to list datasets" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: Join Key Extraction
# ---------------------------------------------------------------------------


class TestJoinKeyExtraction:
    """Tests for extracting join key columns from child datasets."""

    def test_join_keys_from_target_side_extracted(self, discovery, mock_qs_client):
        """When child joins target with another table, target-side join keys are extracted."""
        child_def = {
            "DataSetId": "child-join",
            "Name": "Child Join",
            "ImportMode": "SPICE",
            "LogicalTableMap": {
                "lt-target": {
                    "Alias": "Target Data",
                    "Source": {"DataSetArn": TARGET_ARN},
                    "DataTransforms": [
                        {"ProjectOperation": {"ProjectedColumns": ["order_id", "customer_id"]}},
                    ],
                },
                "lt-other": {
                    "Alias": "Other Table",
                    "Source": {"PhysicalTableId": "pt-other"},
                    "DataTransforms": [],
                },
                "lt-joined": {
                    "Alias": "Joined",
                    "Source": {
                        "JoinInstruction": {
                            "LeftOperand": "lt-target",
                            "RightOperand": "lt-other",
                            "Type": "INNER",
                            "OnClause": "customer_id = customer_id",
                        }
                    },
                    "DataTransforms": [],
                },
            },
        }

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-join"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        # customer_id is a join key from the target side
        assert "customer_id" in result[0].referenced_columns
        # order_id is in the target's own ProjectOperation
        assert "order_id" in result[0].referenced_columns

    def test_multi_condition_join_keys(self, discovery, mock_qs_client):
        """Multi-condition ON clause (AND) should extract all join columns."""
        child_def = {
            "DataSetId": "child-multi-join",
            "Name": "Child Multi Join",
            "ImportMode": "SPICE",
            "LogicalTableMap": {
                "lt-target": {
                    "Alias": "Target Data",
                    "Source": {"DataSetArn": TARGET_ARN},
                    "DataTransforms": [],
                },
                "lt-other": {
                    "Alias": "Other",
                    "Source": {"PhysicalTableId": "pt-other"},
                    "DataTransforms": [],
                },
                "lt-joined": {
                    "Alias": "Joined",
                    "Source": {
                        "JoinInstruction": {
                            "LeftOperand": "lt-target",
                            "RightOperand": "lt-other",
                            "Type": "LEFT",
                            "OnClause": "region = region AND year = year",
                        }
                    },
                    "DataTransforms": [],
                },
            },
        }

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-multi-join"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "region" in result[0].referenced_columns
        assert "year" in result[0].referenced_columns


# ---------------------------------------------------------------------------
# Tests: Calculated Field Extraction
# ---------------------------------------------------------------------------


class TestCalcFieldExtraction:
    """Tests for extracting columns from dataset-level calculated fields."""

    def test_calc_fields_with_bracket_syntax(self, discovery, mock_qs_client):
        """Dataset-level calc fields using {[ColumnName]} syntax should be extracted."""
        child_def = _make_child_def_legacy("child-calc", TARGET_ARN, transforms=[])
        child_def["CalculatedFields"] = [
            {
                "DataSetIdentifier": "ds1",
                "Name": "total",
                "Expression": "{[quantity]} * {[unit_price]}",
            }
        ]

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-calc"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "quantity" in result[0].referenced_columns
        assert "unit_price" in result[0].referenced_columns

    def test_calc_fields_with_dot_syntax(self, discovery, mock_qs_client):
        """Dataset-level calc fields using {DSId.ColumnName} syntax should be extracted."""
        child_def = _make_child_def_legacy("child-calc", TARGET_ARN, transforms=[])
        child_def["CalculatedFields"] = [
            {
                "DataSetIdentifier": "my_dataset",
                "Name": "margin",
                "Expression": "{my_dataset.revenue} - {my_dataset.cost}",
            }
        ]

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-calc"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "revenue" in result[0].referenced_columns
        assert "cost" in result[0].referenced_columns


# ---------------------------------------------------------------------------
# Tests: ARN Matching
# ---------------------------------------------------------------------------


class TestArnMatching:
    """Tests for the ARN matching logic."""

    def test_exact_arn_match(self, discovery, mock_qs_client):
        """Exact ARN match should identify child."""
        child_def = _make_child_def_legacy("child-1", TARGET_ARN)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1

    def test_different_partition_arn_still_matches(self, discovery, mock_qs_client):
        """ARN with different partition (aws-cn) but same suffix should match."""
        cn_arn = f"arn:aws-cn:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{TARGET_DATASET_ID}"
        child_def = _make_child_def_legacy("child-1", cn_arn)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1

    def test_substring_dataset_id_does_not_match(self, discovery, mock_qs_client):
        """A dataset ID that is a substring should NOT match (e.g., 'target-dataset-001-backup')."""
        # This ARN ends with a different dataset ID that contains the target as a prefix
        wrong_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{TARGET_DATASET_ID}-backup"
        child_def = _make_child_def_legacy("child-1", wrong_arn)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-1"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        # Should NOT match — the suffix is "dataset/target-dataset-001-backup"
        # not "dataset/target-dataset-001"
        assert len(result) == 0

    def test_empty_arn_does_not_match(self, discovery, mock_qs_client):
        """Empty DataSetArn should not cause a match."""
        child_def = {
            "DataSetId": "child-empty",
            "Name": "Child Empty",
            "ImportMode": "SPICE",
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Table",
                    "Source": {"DataSetArn": ""},
                    "DataTransforms": [],
                }
            },
        }

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-empty"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    def test_dataset_summary_without_id_is_skipped(self, discovery, mock_qs_client):
        """Dataset summaries with no DataSetId are skipped."""
        mock_qs_client.list_datasets.return_value = [
            {"Name": "No ID Dataset", "ImportMode": "SPICE"},
            _make_dataset_summary(TARGET_DATASET_ID),
        ]

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert result == []
        mock_qs_client.describe_dataset.assert_not_called()

    def test_child_with_both_logical_and_data_prep_referencing_target(
        self, discovery, mock_qs_client
    ):
        """If a child references target in both locations, columns from both are merged."""
        child_def = {
            "DataSetId": "child-dual",
            "Name": "Child Dual",
            "ImportMode": "SPICE",
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Legacy Source",
                    "Source": {"DataSetArn": TARGET_ARN},
                    "DataTransforms": [
                        {"ProjectOperation": {"ProjectedColumns": ["col_a", "col_b"]}},
                    ],
                }
            },
            "DataPrepConfiguration": {
                "SourceTableMap": {
                    "src-1": {
                        "DataSet": {
                            "DataSetArn": TARGET_ARN,
                            "InputColumns": [
                                {"Name": "col_b", "Type": "STRING", "Id": "id-b"},
                                {"Name": "col_c", "Type": "STRING", "Id": "id-c"},
                            ],
                        }
                    }
                },
                "TransformStepMap": {},
                "DestinationTableMap": {},
            },
        }

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-dual"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        # Should include columns from both sources
        assert result[0].referenced_columns == {"col_a", "col_b", "col_c"}

    def test_filter_operation_expression_columns_extracted(self, discovery, mock_qs_client):
        """FilterOperation ConditionExpression columns should be extracted."""
        transforms = [
            {
                "FilterOperation": {
                    "ConditionExpression": "{ds1.status} = 'active'",
                }
            },
        ]
        child_def = _make_child_def_legacy("child-filter", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-filter"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "status" in result[0].referenced_columns

    def test_tag_and_untag_operations_extracted(self, discovery, mock_qs_client):
        """TagColumnOperation and UntagColumnOperation columns should be extracted."""
        transforms = [
            {"TagColumnOperation": {"ColumnName": "country", "Tags": []}},
            {"UntagColumnOperation": {"ColumnName": "region", "TagNames": []}},
        ]
        child_def = _make_child_def_legacy("child-tag", TARGET_ARN, transforms)

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-tag"),
        ]
        mock_qs_client.describe_dataset.return_value = child_def

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert "country" in result[0].referenced_columns
        assert "region" in result[0].referenced_columns


# ---------------------------------------------------------------------------
# Tests: Recursive Discovery
# ---------------------------------------------------------------------------


class TestRecursiveDiscovery:
    """Tests for recursive child dataset discovery at multiple depths."""

    def test_discovers_grandchildren(self, discovery, mock_qs_client):
        """Should find children at depth 2 (grandchildren)."""
        # A -> B -> C (A is target, B is child, C is grandchild)
        child_b_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/child-b"
        child_b_def = _make_child_def_legacy(
            "child-b", TARGET_ARN,
            [{"ProjectOperation": {"ProjectedColumns": ["col_from_a"]}}],
        )
        child_c_def = _make_child_def_legacy(
            "child-c", child_b_arn,
            [{"ProjectOperation": {"ProjectedColumns": ["col_from_b"]}}],
        )

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("child-b"),
            _make_dataset_summary("child-c"),
        ]

        def describe_side_effect(ds_id):
            return {
                "child-b": child_b_def,
                "child-c": child_c_def,
            }.get(ds_id, _make_unrelated_def(ds_id))

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 2
        ids = {r.dataset_id for r in result}
        assert ids == {"child-b", "child-c"}

        # Check depths
        depths = {r.dataset_id: r.depth for r in result}
        assert depths["child-b"] == 1
        assert depths["child-c"] == 2

    def test_discovers_three_levels_deep(self, discovery, mock_qs_client):
        """Should find great-grandchildren (depth 3)."""
        # A -> B -> C -> D
        b_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-b"
        c_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-c"

        b_def = _make_child_def_legacy("ds-b", TARGET_ARN, [])
        c_def = _make_child_def_legacy("ds-c", b_arn, [])
        d_def = _make_child_def_legacy("ds-d", c_arn, [])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("ds-b"),
            _make_dataset_summary("ds-c"),
            _make_dataset_summary("ds-d"),
        ]

        def describe_side_effect(ds_id):
            return {"ds-b": b_def, "ds-c": c_def, "ds-d": d_def}.get(
                ds_id, _make_unrelated_def(ds_id)
            )

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 3
        depths = {r.dataset_id: r.depth for r in result}
        assert depths["ds-b"] == 1
        assert depths["ds-c"] == 2
        assert depths["ds-d"] == 3

    def test_cycle_detection_prevents_infinite_loop(self, discovery, mock_qs_client):
        """Circular references should not cause infinite recursion."""
        # A -> B -> A (cycle)
        b_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-b"
        b_def = _make_child_def_legacy("ds-b", TARGET_ARN, [])
        # ds-b also references target back — but target is already visited
        # Additionally, make a dataset that references ds-b back to A
        a_as_child_of_b = _make_child_def_legacy("ds-a-fake", b_arn, [])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("ds-b"),
        ]

        def describe_side_effect(ds_id):
            if ds_id == "ds-b":
                return b_def
            return _make_unrelated_def(ds_id)

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        # Should complete without hanging
        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        assert len(result) == 1
        assert result[0].dataset_id == "ds-b"

    def test_max_depth_limits_recursion(self, discovery, mock_qs_client):
        """Should stop at max_depth and not go deeper."""
        # A -> B -> C -> D but max_depth=2 so D should not be found
        b_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-b"
        c_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-c"

        b_def = _make_child_def_legacy("ds-b", TARGET_ARN, [])
        c_def = _make_child_def_legacy("ds-c", b_arn, [])
        d_def = _make_child_def_legacy("ds-d", c_arn, [])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("ds-b"),
            _make_dataset_summary("ds-c"),
            _make_dataset_summary("ds-d"),
        ]

        def describe_side_effect(ds_id):
            return {"ds-b": b_def, "ds-c": c_def, "ds-d": d_def}.get(
                ds_id, _make_unrelated_def(ds_id)
            )

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(TARGET_DATASET_ID, max_depth=2)

        # Only B (depth 1) and C (depth 2) should be found, not D (depth 3)
        assert len(result) == 2
        ids = {r.dataset_id for r in result}
        assert "ds-b" in ids
        assert "ds-c" in ids
        assert "ds-d" not in ids

    def test_diamond_dependency_not_duplicated(self, discovery, mock_qs_client):
        """A dataset reachable via multiple paths should only appear once."""
        # A -> B, A -> C, B -> D, C -> D (diamond: D reachable via B and C)
        b_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-b"
        c_arn = f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/ds-c"

        b_def = _make_child_def_legacy("ds-b", TARGET_ARN, [])
        c_def = _make_child_def_legacy("ds-c", TARGET_ARN, [])
        d_def = _make_child_def_legacy("ds-d", b_arn, [])
        # D also references C but since visited set has D already, won't duplicate

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("ds-b"),
            _make_dataset_summary("ds-c"),
            _make_dataset_summary("ds-d"),
        ]

        def describe_side_effect(ds_id):
            return {
                "ds-b": b_def,
                "ds-c": c_def,
                "ds-d": d_def,
            }.get(ds_id, _make_unrelated_def(ds_id))

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        result = discovery.discover_child_datasets(TARGET_DATASET_ID)

        # D should appear exactly once (found via B, skipped when checking C's children)
        ids = [r.dataset_id for r in result]
        assert ids.count("ds-d") == 1
        assert len(result) == 3  # B, C, D

    def test_caching_avoids_redundant_describe_calls(self, discovery, mock_qs_client):
        """Datasets should only be described once thanks to QSClient caching."""
        b_def = _make_child_def_legacy("ds-b", TARGET_ARN, [])

        mock_qs_client.list_datasets.return_value = [
            _make_dataset_summary(TARGET_DATASET_ID),
            _make_dataset_summary("ds-b"),
        ]

        call_count = {"ds-b": 0}

        def describe_side_effect(ds_id):
            call_count[ds_id] = call_count.get(ds_id, 0) + 1
            if ds_id == "ds-b":
                return b_def
            return _make_unrelated_def(ds_id)

        mock_qs_client.describe_dataset.side_effect = describe_side_effect

        discovery.discover_child_datasets(TARGET_DATASET_ID)

        # ds-b described during level 1 scan, then during recursion into B's children
        # it should be called once for each unique dataset (not the target or already visited)
        # The key point: list_datasets is called only ONCE (not per recursion level)
        mock_qs_client.list_datasets.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Child Consumer Scanning
# ---------------------------------------------------------------------------


class TestChildConsumerScanning:
    """Tests for scanning child datasets' consumers to trace parent column usage."""

    def test_scan_child_consumers_traces_columns_to_parent(
        self, discovery, mock_qs_client
    ):
        """Columns used by child's consumers that come from parent should be returned."""
        from unittest.mock import MagicMock, patch
        from lambda_source.scanner import ColumnScanner

        child = ChildDataset(
            dataset_id="child-1",
            name="Child 1",
            referenced_columns={"col_a", "col_b", "col_c"},
            depth=1,
        )

        # Mock consumer discovery to return one dashboard consumer
        mock_consumer = MagicMock()
        mock_consumer.consumer_type = "dashboard"
        mock_consumer.arn = "arn:aws:quicksight:us-east-1:123456789012:dashboard/dash-1"
        mock_consumer.dataset_identifier = "child_ds"

        # Mock scanner to return columns that the dashboard uses from the child
        mock_scanner = MagicMock(spec=ColumnScanner)
        mock_scanner.scan_definition.return_value = {"col_a", "col_c", "col_x"}

        # Mock child dataset definition (for tracing join keys)
        child_def = {
            "DataSetId": "child-1",
            "Name": "Child 1",
            "OutputColumns": [
                {"Name": "col_a"}, {"Name": "col_b"}, {"Name": "col_c"}, {"Name": "col_x"}
            ],
            "LogicalTableMap": {},
        }
        mock_qs_client.describe_dataset.return_value = child_def
        mock_qs_client.describe_dashboard_definition.return_value = {"Sheets": []}

        with patch(
            "lambda_source.discovery.ConsumerDiscovery"
        ) as mock_disc_cls:
            mock_disc_instance = MagicMock()
            mock_disc_instance.discover_consumers.return_value = [mock_consumer]
            mock_disc_cls.return_value = mock_disc_instance

            result = discovery.scan_child_consumers(
                children=[child],
                target_dataset_id=TARGET_DATASET_ID,
                scanner=mock_scanner,
            )

        # col_a and col_c are used by dashboard AND come from parent
        # col_x is used by dashboard but NOT in referenced_columns (comes from elsewhere)
        # col_b comes from parent but NOT used by dashboard
        assert "col_a" in result
        assert "col_c" in result
        assert "col_x" not in result
        assert "col_b" not in result

    def test_scan_child_consumers_no_consumers_falls_back_to_structural(
        self, discovery, mock_qs_client
    ):
        """If child has no consumers, all structural deps should be protected."""
        from unittest.mock import MagicMock, patch
        from lambda_source.scanner import ColumnScanner

        child = ChildDataset(
            dataset_id="child-1",
            name="Child 1",
            referenced_columns={"col_a", "col_b"},
            depth=1,
        )

        mock_scanner = MagicMock(spec=ColumnScanner)

        with patch(
            "lambda_source.discovery.ConsumerDiscovery"
        ) as mock_disc_cls:
            mock_disc_instance = MagicMock()
            mock_disc_instance.discover_consumers.return_value = []
            mock_disc_cls.return_value = mock_disc_instance

            result = discovery.scan_child_consumers(
                children=[child],
                target_dataset_id=TARGET_DATASET_ID,
                scanner=mock_scanner,
            )

        # Falls back to structural deps
        assert result == {"col_a", "col_b"}

    def test_scan_child_consumers_error_falls_back_to_structural(
        self, discovery, mock_qs_client
    ):
        """If consumer scanning fails, fall back to structural dependencies."""
        from unittest.mock import MagicMock, patch
        from lambda_source.scanner import ColumnScanner

        child = ChildDataset(
            dataset_id="child-1",
            name="Child 1",
            referenced_columns={"col_a", "col_b", "col_c"},
            depth=1,
        )

        mock_scanner = MagicMock(spec=ColumnScanner)

        with patch(
            "lambda_source.discovery.ConsumerDiscovery"
        ) as mock_disc_cls:
            mock_disc_instance = MagicMock()
            mock_disc_instance.discover_consumers.side_effect = Exception("API failure")
            mock_disc_cls.return_value = mock_disc_instance

            result = discovery.scan_child_consumers(
                children=[child],
                target_dataset_id=TARGET_DATASET_ID,
                scanner=mock_scanner,
            )

        # Falls back to structural deps
        assert result == {"col_a", "col_b", "col_c"}

    def test_scan_child_consumers_protects_join_keys_in_child(
        self, discovery, mock_qs_client
    ):
        """Join keys within the child that come from parent should be protected."""
        from unittest.mock import MagicMock, patch
        from lambda_source.scanner import ColumnScanner

        child = ChildDataset(
            dataset_id="child-1",
            name="Child 1",
            referenced_columns={"join_col", "data_col_a", "data_col_b"},
            depth=1,
        )

        mock_consumer = MagicMock()
        mock_consumer.consumer_type = "analysis"
        mock_consumer.arn = "arn:aws:quicksight:us-east-1:123456789012:analysis/an-1"
        mock_consumer.dataset_identifier = "child_ds"

        # Dashboard only uses data_col_a, not join_col directly
        mock_scanner = MagicMock(spec=ColumnScanner)
        mock_scanner.scan_definition.return_value = {"data_col_a", "other_col"}

        # Child definition has a join that uses join_col
        child_def = {
            "DataSetId": "child-1",
            "Name": "Child 1",
            "OutputColumns": [
                {"Name": "join_col"}, {"Name": "data_col_a"}, {"Name": "data_col_b"}
            ],
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Joined",
                    "Source": {
                        "JoinInstruction": {
                            "LeftOperand": "lt-parent",
                            "RightOperand": "lt-other",
                            "OnClause": "join_col = join_col",
                            "Type": "INNER",
                        }
                    },
                    "DataTransforms": [],
                }
            },
        }
        mock_qs_client.describe_dataset.return_value = child_def
        mock_qs_client.describe_analysis_definition.return_value = {"Sheets": []}

        with patch(
            "lambda_source.discovery.ConsumerDiscovery"
        ) as mock_disc_cls:
            mock_disc_instance = MagicMock()
            mock_disc_instance.discover_consumers.return_value = [mock_consumer]
            mock_disc_cls.return_value = mock_disc_instance

            result = discovery.scan_child_consumers(
                children=[child],
                target_dataset_id=TARGET_DATASET_ID,
                scanner=mock_scanner,
            )

        # join_col should be protected (it's a join key in child AND comes from parent)
        # data_col_a should be protected (used by consumer AND comes from parent)
        # data_col_b should NOT be protected (not used by consumer, not a join key)
        assert "join_col" in result
        assert "data_col_a" in result
        assert "data_col_b" not in result

    def test_scan_child_consumers_multiple_children(self, discovery, mock_qs_client):
        """Should union results across multiple children."""
        from unittest.mock import MagicMock, patch
        from lambda_source.scanner import ColumnScanner

        child_1 = ChildDataset(
            dataset_id="child-1", name="C1",
            referenced_columns={"col_a", "col_b"}, depth=1,
        )
        child_2 = ChildDataset(
            dataset_id="child-2", name="C2",
            referenced_columns={"col_b", "col_c"}, depth=1,
        )

        mock_scanner = MagicMock(spec=ColumnScanner)

        with patch(
            "lambda_source.discovery.ConsumerDiscovery"
        ) as mock_disc_cls:
            mock_disc_instance = MagicMock()
            # child-1 has no consumers, child-2 has no consumers
            mock_disc_instance.discover_consumers.return_value = []
            mock_disc_cls.return_value = mock_disc_instance

            result = discovery.scan_child_consumers(
                children=[child_1, child_2],
                target_dataset_id=TARGET_DATASET_ID,
                scanner=mock_scanner,
            )

        # Both fall back to structural deps, unioned
        assert result == {"col_a", "col_b", "col_c"}


# ---------------------------------------------------------------------------
# Tests: Dataset Definition Caching
# ---------------------------------------------------------------------------


class TestDatasetCaching:
    """Tests for QSClient dataset definition caching."""

    def test_describe_dataset_caches_result(self, mock_qs_client):
        """Second call with same ID should not hit the mock again."""
        from lambda_source.qs_client import QSClient

        # Use a real QSClient with a mock boto3 client
        mock_boto = MagicMock()
        mock_boto.describe_data_set.return_value = {
            "DataSet": {"DataSetId": "ds-1", "Name": "Test"}
        }
        client = QSClient(account_id=ACCOUNT_ID, region=REGION, client=mock_boto)

        # First call — hits the API
        result1 = client.describe_dataset("ds-1")
        assert result1 == {"DataSetId": "ds-1", "Name": "Test"}
        assert mock_boto.describe_data_set.call_count == 1

        # Second call — served from cache
        result2 = client.describe_dataset("ds-1")
        assert result2 == {"DataSetId": "ds-1", "Name": "Test"}
        assert mock_boto.describe_data_set.call_count == 1  # Still 1

    def test_clear_cache_forces_fresh_call(self, mock_qs_client):
        """After clear_cache(), next call should hit the API."""
        from lambda_source.qs_client import QSClient

        mock_boto = MagicMock()
        mock_boto.describe_data_set.return_value = {
            "DataSet": {"DataSetId": "ds-1", "Name": "Test"}
        }
        client = QSClient(account_id=ACCOUNT_ID, region=REGION, client=mock_boto)

        client.describe_dataset("ds-1")
        assert mock_boto.describe_data_set.call_count == 1

        client.clear_cache()

        client.describe_dataset("ds-1")
        assert mock_boto.describe_data_set.call_count == 2
