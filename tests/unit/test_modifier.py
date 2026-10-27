# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the DatasetModifier class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lambda_source.modifier import DatasetModifier, ModificationError, ValidationError
from lambda_source.qs_client import QSClient, QSClientError


@pytest.fixture
def mock_qs_client():
    """Create a mock QSClient."""
    client = MagicMock(spec=QSClient)
    client.account_id = "123456789012"
    return client


@pytest.fixture
def modifier(mock_qs_client):
    """Create a DatasetModifier instance with a mock client."""
    return DatasetModifier(mock_qs_client)


@pytest.fixture
def sample_dataset_def():
    """Create a sample dataset definition for testing."""
    return {
        "Name": "Sales Data",
        "ImportMode": "SPICE",
        "PhysicalTableMap": {
            "physical-1": {
                "RelationalTable": {
                    "DataSourceArn": "arn:aws:quicksight:us-east-1:123:datasource/ds-1",
                    "Schema": "public",
                    "Name": "sales",
                    "InputColumns": [
                        {"Name": "order_id", "Type": "STRING"},
                        {"Name": "customer_id", "Type": "STRING"},
                        {"Name": "amount", "Type": "DECIMAL"},
                        {"Name": "unused_col_1", "Type": "STRING"},
                        {"Name": "unused_col_2", "Type": "INTEGER"},
                    ],
                }
            }
        },
        "LogicalTableMap": {
            "logical-1": {
                "Alias": "Sales",
                "Source": {"PhysicalTableId": "physical-1"},
                "DataTransforms": [
                    {"RenameColumnOperation": {"ColumnName": "amount", "NewColumnName": "total_amount"}},
                ],
            }
        },
        "OutputColumns": [
            {"Name": "order_id", "Type": "STRING"},
            {"Name": "customer_id", "Type": "STRING"},
            {"Name": "total_amount", "Type": "DECIMAL"},
            {"Name": "unused_col_1", "Type": "STRING"},
            {"Name": "unused_col_2", "Type": "INTEGER"},
        ],
    }


class TestBuildOptimizedDefinition:
    """Tests for build_optimized_definition."""

    def test_inserts_project_operation_after_existing_transforms(self, modifier, sample_dataset_def):
        """ProjectOperation should be placed at the end after other transforms."""
        unused = ["unused_col_1", "unused_col_2"]
        result = modifier.build_optimized_definition(sample_dataset_def, unused)

        logical_table = result["LogicalTableMap"]["logical-1"]
        transforms = logical_table["DataTransforms"]

        # RenameColumnOperation should still be first (original position)
        assert "RenameColumnOperation" in transforms[0]
        # ProjectOperation should be last (replaces the existing one at the end or appended)
        assert "ProjectOperation" in transforms[-1]

    def test_project_operation_excludes_unused_columns(self, modifier, sample_dataset_def):
        """ProjectedColumns should contain only retained columns."""
        unused = ["unused_col_1", "unused_col_2"]
        result = modifier.build_optimized_definition(sample_dataset_def, unused)

        logical_table = result["LogicalTableMap"]["logical-1"]
        projected = logical_table["DataTransforms"][-1]["ProjectOperation"]["ProjectedColumns"]

        assert "unused_col_1" not in projected
        assert "unused_col_2" not in projected
        assert "order_id" in projected
        assert "customer_id" in projected
        assert "total_amount" in projected

    def test_no_unused_columns_returns_unchanged_copy(self, modifier, sample_dataset_def):
        """When no columns are unused, return an unchanged deep copy."""
        result = modifier.build_optimized_definition(sample_dataset_def, [])

        # Should be a deep copy (not same object)
        assert result is not sample_dataset_def
        # Content should match original structure
        assert result["LogicalTableMap"]["logical-1"]["DataTransforms"] == sample_dataset_def["LogicalTableMap"]["logical-1"]["DataTransforms"]

    def test_replaces_existing_project_operation_at_last_position(self, modifier):
        """If last position already has a ProjectOperation, replace it."""
        dataset_def = {
            "OutputColumns": [
                {"Name": "col_a", "Type": "STRING"},
                {"Name": "col_b", "Type": "STRING"},
                {"Name": "col_c", "Type": "STRING"},
            ],
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Table",
                    "Source": {"PhysicalTableId": "pt-1"},
                    "DataTransforms": [
                        {"RenameColumnOperation": {"ColumnName": "col_a", "NewColumnName": "column_a"}},
                        {"ProjectOperation": {"ProjectedColumns": ["col_a", "col_b", "col_c"]}},
                    ],
                }
            },
        }

        result = modifier.build_optimized_definition(dataset_def, ["col_c"])

        transforms = result["LogicalTableMap"]["lt-1"]["DataTransforms"]
        # Should still have exactly 2 transforms (replaced, not appended)
        assert len(transforms) == 2
        assert "RenameColumnOperation" in transforms[0]
        assert "ProjectOperation" in transforms[-1]
        assert "col_c" not in transforms[-1]["ProjectOperation"]["ProjectedColumns"]
        assert "col_a" in transforms[-1]["ProjectOperation"]["ProjectedColumns"]

    def test_creates_data_transforms_when_missing(self, modifier):
        """When logical table has no DataTransforms, create one with ProjectOperation."""
        dataset_def = {
            "OutputColumns": [
                {"Name": "col_a", "Type": "STRING"},
                {"Name": "col_b", "Type": "STRING"},
            ],
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Table",
                    "Source": {"PhysicalTableId": "pt-1"},
                }
            },
        }

        result = modifier.build_optimized_definition(dataset_def, ["col_b"])

        transforms = result["LogicalTableMap"]["lt-1"]["DataTransforms"]
        assert len(transforms) == 1
        assert "ProjectOperation" in transforms[-1]
        assert transforms[-1]["ProjectOperation"]["ProjectedColumns"] == ["col_a"]

    def test_does_not_mutate_original_definition(self, modifier, sample_dataset_def):
        """The original dataset_def should not be modified."""
        import copy
        original = copy.deepcopy(sample_dataset_def)
        modifier.build_optimized_definition(sample_dataset_def, ["unused_col_1"])

        assert sample_dataset_def == original

    def test_handles_multiple_logical_tables(self, modifier):
        """ProjectOperation should be inserted into all logical tables."""
        dataset_def = {
            "OutputColumns": [
                {"Name": "col_a", "Type": "STRING"},
                {"Name": "col_b", "Type": "STRING"},
                {"Name": "col_c", "Type": "STRING"},
            ],
            "LogicalTableMap": {
                "lt-1": {
                    "Alias": "Table1",
                    "Source": {"PhysicalTableId": "pt-1"},
                    "DataTransforms": [],
                },
                "lt-2": {
                    "Alias": "Table2",
                    "Source": {"PhysicalTableId": "pt-2"},
                    "DataTransforms": [
                        {"CastColumnTypeOperation": {"ColumnName": "col_b", "NewColumnType": "INTEGER"}},
                    ],
                },
            },
        }

        result = modifier.build_optimized_definition(dataset_def, ["col_c"])

        # Both logical tables should have ProjectOperation at the end
        for table_id in ["lt-1", "lt-2"]:
            transforms = result["LogicalTableMap"][table_id]["DataTransforms"]
            assert "ProjectOperation" in transforms[-1]
            assert "col_c" not in transforms[-1]["ProjectOperation"]["ProjectedColumns"]


class TestApplyModification:
    """Tests for apply_modification."""

    def test_calls_update_dataset_with_correct_params(self, modifier, mock_qs_client, sample_dataset_def):
        """Should call qs_client.update_dataset with extracted parameters."""
        mock_qs_client.update_dataset.return_value = {"Status": 200, "Arn": "arn:aws:quicksight:..."}

        result = modifier.apply_modification("dataset-123", sample_dataset_def)

        mock_qs_client.update_dataset.assert_called_once()
        call_args = mock_qs_client.update_dataset.call_args
        assert call_args[0][0] == "dataset-123"  # dataset_id
        params = call_args[0][1]
        assert params["Name"] == "Sales Data"
        assert params["ImportMode"] == "SPICE"
        assert "PhysicalTableMap" in params
        assert "LogicalTableMap" in params

    def test_returns_api_response(self, modifier, mock_qs_client, sample_dataset_def):
        """Should return the UpdateDataSet API response."""
        expected_response = {"Status": 200, "Arn": "arn:aws:quicksight:us-east-1:123:dataset/ds-1"}
        mock_qs_client.update_dataset.return_value = expected_response

        result = modifier.apply_modification("dataset-123", sample_dataset_def)

        assert result == expected_response

    def test_raises_modification_error_on_failure(self, modifier, mock_qs_client, sample_dataset_def):
        """Should raise ModificationError when UpdateDataSet fails."""
        mock_qs_client.update_dataset.side_effect = QSClientError(
            "Access denied", error_code="AccessDeniedException", api_name="update_data_set"
        )

        with pytest.raises(ModificationError) as exc_info:
            modifier.apply_modification("dataset-123", sample_dataset_def)

        assert "dataset-123" in str(exc_info.value)
        assert exc_info.value.dataset_id == "dataset-123"

    def test_modification_error_preserves_snapshot_key(self, modifier, mock_qs_client, sample_dataset_def):
        """ModificationError should support snapshot_key for rollback reference (Req 7.5)."""
        mock_qs_client.update_dataset.side_effect = QSClientError(
            "Internal error", error_code="InternalFailure", api_name="update_data_set"
        )

        with pytest.raises(ModificationError) as exc_info:
            modifier.apply_modification("dataset-456", sample_dataset_def)

        # snapshot_key defaults to None when not set by caller; modifier doesn't set it
        # but the attribute must exist so orchestrator can assign it post-catch
        assert hasattr(exc_info.value, "snapshot_key")
        assert exc_info.value.dataset_id == "dataset-456"

    def test_modification_error_wraps_original_exception(self, modifier, mock_qs_client, sample_dataset_def):
        """ModificationError should chain the original QSClientError for diagnostics (Req 7.5)."""
        original_error = QSClientError(
            "Conflict detected", error_code="ConflictException", api_name="update_data_set"
        )
        mock_qs_client.update_dataset.side_effect = original_error

        with pytest.raises(ModificationError) as exc_info:
            modifier.apply_modification("dataset-789", sample_dataset_def)

        # Should chain the original exception via __cause__
        assert exc_info.value.__cause__ is original_error
        assert "dataset-789" in str(exc_info.value)


class TestTriggerIngestion:
    """Tests for trigger_ingestion."""

    def test_calls_create_ingestion_with_unique_id(self, modifier, mock_qs_client):
        """Should call create_ingestion with a unique ingestion_id."""
        mock_qs_client.create_ingestion.return_value = {
            "Status": 201,
            "IngestionId": "optimiser-abc123",
        }

        result = modifier.trigger_ingestion("dataset-123")

        mock_qs_client.create_ingestion.assert_called_once()
        call_args = mock_qs_client.create_ingestion.call_args
        assert call_args[0][0] == "dataset-123"
        # Ingestion ID should start with "optimiser-"
        assert call_args[0][1].startswith("optimiser-")

    def test_returns_api_response(self, modifier, mock_qs_client):
        """Should return the CreateIngestion API response."""
        expected = {"Status": 201, "IngestionId": "optimiser-abc123"}
        mock_qs_client.create_ingestion.return_value = expected

        result = modifier.trigger_ingestion("dataset-123")

        assert result == expected

    def test_raises_modification_error_on_failure(self, modifier, mock_qs_client):
        """Should raise ModificationError when CreateIngestion fails."""
        mock_qs_client.create_ingestion.side_effect = QSClientError(
            "Throttled", error_code="ThrottlingException", api_name="create_ingestion"
        )

        with pytest.raises(ModificationError) as exc_info:
            modifier.trigger_ingestion("dataset-123")

        assert exc_info.value.dataset_id == "dataset-123"

    def test_generates_unique_ingestion_ids(self, modifier, mock_qs_client):
        """Each call should generate a distinct ingestion ID (Req 7.4)."""
        mock_qs_client.create_ingestion.return_value = {"Status": 201}

        modifier.trigger_ingestion("dataset-1")
        modifier.trigger_ingestion("dataset-1")

        calls = mock_qs_client.create_ingestion.call_args_list
        id_1 = calls[0][0][1]
        id_2 = calls[1][0][1]
        assert id_1 != id_2
        assert id_1.startswith("optimiser-")
        assert id_2.startswith("optimiser-")


class TestValidateModification:
    """Tests for validate_modification."""

    def test_returns_true_when_retained_is_superset(self, modifier):
        """Should return True when retained columns cover all consumer references."""
        retained = {"col_a", "col_b", "col_c", "col_d"}
        consumer_refs = {"col_a", "col_b"}

        result = modifier.validate_modification(retained, consumer_refs)

        assert result is True

    def test_returns_true_when_sets_are_equal(self, modifier):
        """Should return True when retained columns exactly equal consumer references."""
        retained = {"col_a", "col_b"}
        consumer_refs = {"col_a", "col_b"}

        result = modifier.validate_modification(retained, consumer_refs)

        assert result is True

    def test_returns_true_when_consumer_refs_empty(self, modifier):
        """Should return True when there are no consumer references."""
        retained = {"col_a", "col_b"}
        consumer_refs: set[str] = set()

        result = modifier.validate_modification(retained, consumer_refs)

        assert result is True

    def test_raises_validation_error_when_missing_columns(self, modifier):
        """Should raise ValidationError when retained doesn't cover consumer refs."""
        retained = {"col_a", "col_b"}
        consumer_refs = {"col_a", "col_b", "col_c", "col_d"}

        with pytest.raises(ValidationError) as exc_info:
            modifier.validate_modification(retained, consumer_refs)

        assert exc_info.value.missing_columns == {"col_c", "col_d"}
        assert "col_c" in str(exc_info.value) or "col_d" in str(exc_info.value)


class TestInsertProjectOperation:
    """Tests for _insert_project_operation."""

    def test_prepends_to_existing_transforms(self, modifier):
        """Should append ProjectOperation after existing non-project transforms."""
        logical_table = {
            "Alias": "Table",
            "Source": {"PhysicalTableId": "pt-1"},
            "DataTransforms": [
                {"RenameColumnOperation": {"ColumnName": "a", "NewColumnName": "b"}},
                {"CastColumnTypeOperation": {"ColumnName": "c", "NewColumnType": "INTEGER"}},
            ],
        }

        result = modifier._insert_project_operation(logical_table, ["col_a", "col_b"])

        assert len(result["DataTransforms"]) == 3
        assert "RenameColumnOperation" in result["DataTransforms"][0]
        assert "CastColumnTypeOperation" in result["DataTransforms"][1]
        assert "ProjectOperation" in result["DataTransforms"][-1]
        assert result["DataTransforms"][-1]["ProjectOperation"]["ProjectedColumns"] == ["col_a", "col_b"]

    def test_replaces_existing_project_at_position_0(self, modifier):
        """Should replace existing ProjectOperation at the last position."""
        logical_table = {
            "Alias": "Table",
            "Source": {"PhysicalTableId": "pt-1"},
            "DataTransforms": [
                {"RenameColumnOperation": {"ColumnName": "a", "NewColumnName": "b"}},
                {"ProjectOperation": {"ProjectedColumns": ["old_col_1", "old_col_2"]}},
            ],
        }

        result = modifier._insert_project_operation(logical_table, ["new_col_1"])

        assert len(result["DataTransforms"]) == 2
        assert "RenameColumnOperation" in result["DataTransforms"][0]
        assert result["DataTransforms"][-1]["ProjectOperation"]["ProjectedColumns"] == ["new_col_1"]

    def test_creates_data_transforms_when_empty(self, modifier):
        """Should create DataTransforms list when it's empty."""
        logical_table = {
            "Alias": "Table",
            "Source": {"PhysicalTableId": "pt-1"},
            "DataTransforms": [],
        }

        result = modifier._insert_project_operation(logical_table, ["col_x"])

        assert len(result["DataTransforms"]) == 1
        assert result["DataTransforms"][0]["ProjectOperation"]["ProjectedColumns"] == ["col_x"]

    def test_creates_data_transforms_when_missing(self, modifier):
        """Should create DataTransforms key when it doesn't exist."""
        logical_table = {
            "Alias": "Table",
            "Source": {"PhysicalTableId": "pt-1"},
        }

        result = modifier._insert_project_operation(logical_table, ["col_y"])

        assert "DataTransforms" in result
        assert len(result["DataTransforms"]) == 1
        assert result["DataTransforms"][0]["ProjectOperation"]["ProjectedColumns"] == ["col_y"]
