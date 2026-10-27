# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Integration tests for the full SPICE Dataset Optimiser pipeline.

Tests the handler function and Orchestrator with mocked AWS clients to verify:
1. Dry run flow (no write APIs called)
2. Apply mode flow (snapshot, modification, ingestion)
3. Partial failure handling (one dataset fails, others continue)
4. Safety threshold halt behaviour

Requirements: 5.1, 5.4, 8.1, 11.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, ANY

import pytest

from lambda_source.handler import handler, Orchestrator
from lambda_source.models import OptimiserConfig


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

ACCOUNT_ID = "123456789012"
REGION = "us-east-1"
DATASET_ID_1 = "dataset-aaa-111"
DATASET_ID_2 = "dataset-bbb-222"
DATASET_ID_3 = "dataset-ccc-333"


def _make_event(
    target_datasets: list[str],
    execution_mode: str = "dry_run",
    safety_threshold_pct: float = 0.7,
    sns_topic_arn: str | None = None,
) -> dict:
    """Build an inline event payload for the Lambda handler."""
    event = {
        "aws_account_id": ACCOUNT_ID,
        "aws_region": REGION,
        "execution_mode": execution_mode,
        "target_datasets": target_datasets,
        "safety_threshold_pct": safety_threshold_pct,
        "s3_bucket": "test-bucket",
        "s3_prefix": "optimiser/",
    }
    if sns_topic_arn:
        event["sns_topic_arn"] = sns_topic_arn
    return event


def _dataset_definition(dataset_id: str, columns: list[dict]) -> dict:
    """Build a mock DescribeDataSet response structure."""
    return {
        "DataSetId": dataset_id,
        "Name": f"Dataset {dataset_id}",
        "ImportMode": "SPICE",
        "OutputColumns": columns,
        "PhysicalTableMap": {
            "pt-1": {
                "RelationalTable": {
                    "DataSourceArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:datasource/ds-1",
                    "Schema": "public",
                    "Name": "table_1",
                    "InputColumns": columns,
                }
            }
        },
        "LogicalTableMap": {
            "lt-1": {
                "Alias": "Main",
                "Source": {"PhysicalTableId": "pt-1"},
                "DataTransforms": [],
            }
        },
    }


def _analysis_definition(dataset_id: str, used_columns: list[str]) -> dict:
    """Build a mock DescribeAnalysisDefinition response."""
    # Build visual field wells referencing used_columns
    field_wells = []
    for col in used_columns:
        field_wells.append({
            "CategoricalDimensionField": {
                "FieldId": f"field-{col}",
                "Column": {"DataSetIdentifier": f"ds-ident-{dataset_id}", "ColumnName": col},
            }
        })

    return {
        "DataSetIdentifierDeclarations": [
            {
                "Identifier": f"ds-ident-{dataset_id}",
                "DataSetArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{dataset_id}",
            }
        ],
        "Sheets": [
            {
                "SheetId": "sheet-1",
                "Name": "Sheet 1",
                "Visuals": [
                    {
                        "BarChartVisual": {
                            "VisualId": "visual-1",
                            "ChartConfiguration": {
                                "FieldWells": {
                                    "BarChartAggregatedFieldWells": {
                                        "Category": field_wells,
                                        "Values": [],
                                    }
                                }
                            },
                        }
                    }
                ],
            }
        ],
        "CalculatedFields": [],
        "FilterGroups": [],
        "ParameterDeclarations": [],
    }


def _analysis_summary(analysis_id: str, status: str = "CREATION_SUCCESSFUL") -> dict:
    """Build a mock analysis summary from ListAnalyses."""
    return {
        "AnalysisId": analysis_id,
        "Arn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:analysis/{analysis_id}",
        "Name": f"Analysis {analysis_id}",
        "Status": status,
    }


# ---------------------------------------------------------------------------
# Test 1: Dry run end-to-end flow
# ---------------------------------------------------------------------------


class TestDryRunFlow:
    """Test end-to-end dry run flow with mocked boto3 clients.

    Verifies that in dry_run mode:
    - DescribeDataSet is called to get column info
    - ListAnalyses/ListDashboards are called for discovery
    - Consumer definitions are scanned
    - Report is generated and stored to S3
    - SNS notification is sent
    - NO write APIs (UpdateDataSet, CreateIngestion) are called

    Requirements: 5.1, 5.4
    """

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_dry_run_two_datasets_no_write_apis(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """Dry run with 2 datasets, one with unused columns: no write APIs called, SNS sent."""
        # Dataset 1: has unused columns
        columns_1 = [
            {"Name": "col_used_1", "Type": "STRING"},
            {"Name": "col_used_2", "Type": "INTEGER"},
            {"Name": "col_unused_1", "Type": "STRING"},
            {"Name": "col_unused_2", "Type": "DECIMAL"},
        ]
        dataset_def_1 = _dataset_definition(DATASET_ID_1, columns_1)
        analysis_def_1 = _analysis_definition(DATASET_ID_1, ["col_used_1", "col_used_2"])

        # Dataset 2: all columns used (no optimization opportunity)
        columns_2 = [
            {"Name": "col_a", "Type": "STRING"},
            {"Name": "col_b", "Type": "INTEGER"},
        ]
        dataset_def_2 = _dataset_definition(DATASET_ID_2, columns_2)
        analysis_def_2 = _analysis_definition(DATASET_ID_2, ["col_a", "col_b"])

        # Mock QuickSight client
        mock_qs = MagicMock()

        def describe_side_effect(**kwargs):
            ds_id = kwargs.get("DataSetId", "")
            if ds_id == DATASET_ID_1:
                return {"DataSet": dataset_def_1}
            elif ds_id == DATASET_ID_2:
                return {"DataSet": dataset_def_2}
            return {"DataSet": {}}

        mock_qs.describe_data_set.side_effect = describe_side_effect
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}

        def describe_analysis_side_effect(**kwargs):
            # The analysis references both datasets
            analysis_id = kwargs.get("AnalysisId", "")
            # We'll build a combined definition that references both datasets
            return {
                "Definition": {
                    "DataSetIdentifierDeclarations": [
                        {
                            "Identifier": f"ds-ident-{DATASET_ID_1}",
                            "DataSetArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{DATASET_ID_1}",
                        },
                        {
                            "Identifier": f"ds-ident-{DATASET_ID_2}",
                            "DataSetArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{DATASET_ID_2}",
                        },
                    ],
                    "Sheets": [
                        {
                            "SheetId": "sheet-1",
                            "Name": "Sheet 1",
                            "Visuals": [
                                {
                                    "BarChartVisual": {
                                        "VisualId": "visual-1",
                                        "ChartConfiguration": {
                                            "FieldWells": {
                                                "BarChartAggregatedFieldWells": {
                                                    "Category": [
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f1",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_1}",
                                                                    "ColumnName": "col_used_1",
                                                                },
                                                            }
                                                        },
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f2",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_1}",
                                                                    "ColumnName": "col_used_2",
                                                                },
                                                            }
                                                        },
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f3",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_2}",
                                                                    "ColumnName": "col_a",
                                                                },
                                                            }
                                                        },
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f4",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_2}",
                                                                    "ColumnName": "col_b",
                                                                },
                                                            }
                                                        },
                                                    ],
                                                    "Values": [],
                                                }
                                            }
                                        },
                                    }
                                }
                            ],
                        }
                    ],
                    "CalculatedFields": [],
                    "FilterGroups": [],
                    "ParameterDeclarations": [],
                }
            }

        mock_qs.describe_analysis_definition.side_effect = describe_analysis_side_effect
        mock_qs_boto3.client.return_value = mock_qs

        # Mock S3 clients (for reporter and snapshot)
        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        # Mock CloudWatch client
        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        # Mock SNS
        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1, DATASET_ID_2],
            execution_mode="dry_run",
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
        )

        result = handler(event, None)

        # Should succeed
        assert result["status"] == "success"
        assert result["datasets_processed"] == 2
        assert result["datasets_failed"] == 0

        # Verify NO write APIs were called
        mock_qs.update_data_set.assert_not_called()
        mock_qs.create_ingestion.assert_not_called()

        # Verify read APIs WERE called
        assert mock_qs.describe_data_set.call_count == 2
        mock_qs.list_analyses.assert_called()
        mock_qs.describe_analysis_definition.assert_called()

        # Verify reports were stored to S3
        assert mock_s3.put_object.call_count >= 2  # at least one report per dataset

        # Verify SNS notification was sent (summary for the dataset with unused cols)
        mock_sns.publish.assert_called()

        # Verify both dataset results have report keys
        for ds_result in result["dataset_results"]:
            assert ds_result["report_s3_key"] is not None

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_dry_run_report_stored_to_s3(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """In dry_run mode, an optimisation report should be stored to S3."""
        columns = [
            {"Name": "col_used", "Type": "STRING"},
            {"Name": "col_unused", "Type": "INTEGER"},
        ]

        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        analysis_def = _analysis_definition(DATASET_ID_1, ["col_used"])

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="dry_run",
        )

        result = handler(event, None)

        assert result["status"] == "success"
        # S3 put_object should have been called for the report
        mock_s3.put_object.assert_called()

        # The report key should be in the dataset results
        assert len(result["dataset_results"]) == 1
        ds_result = result["dataset_results"][0]
        assert ds_result["report_s3_key"] is not None
        assert "reports/" in ds_result["report_s3_key"]


# ---------------------------------------------------------------------------
# Test 2: Apply mode flow
# ---------------------------------------------------------------------------


class TestApplyModeFlow:
    """Test end-to-end apply mode flow with mocked boto3 clients.

    Verifies that in apply mode:
    - Snapshot is captured before modification
    - Dataset definition is modified (UpdateDataSet called)
    - Ingestion is triggered (CreateIngestion called)
    - Report includes snapshot key and API response

    Requirements: 5.1, 5.4
    """

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    # Added when the per-dataset lock shipped in c496dd6. Apply mode holds an S3 lock
    # across the whole snapshot -> validate -> update -> ingest sequence, so without
    # this the lock cannot be acquired and the dataset fails before _execute_apply.
    # It went unnoticed because this suite hung and never reported.
    @patch("lambda_source.dataset_lock.boto3")
    def test_apply_mode_full_flow(
        self,
        mock_lock_boto3,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """Apply mode should snapshot, modify, ingest, and report with snapshot key."""
        columns = [
            {"Name": "col_used_1", "Type": "STRING"},
            {"Name": "col_used_2", "Type": "INTEGER"},
            {"Name": "col_unused_1", "Type": "STRING"},
        ]

        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        analysis_def = _analysis_definition(DATASET_ID_1, ["col_used_1", "col_used_2"])

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs.update_data_set.return_value = {"Status": 200, "Arn": "arn:updated"}
        mock_qs.create_ingestion.return_value = {"Status": 201, "IngestionId": "ing-1"}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="apply",
        )

        result = handler(event, None)

        assert result["status"] == "success"
        assert result["datasets_processed"] == 1
        assert result["total_columns_removed"] == 1

        # Verify write APIs WERE called
        mock_qs.update_data_set.assert_called_once()
        mock_qs.create_ingestion.assert_called_once()

        # Verify snapshot was captured (S3 put for snapshot + report)
        assert mock_s3.put_object.call_count >= 2  # snapshot + report

        # Check the snapshot_s3_key is in results
        ds_result = result["dataset_results"][0]
        assert ds_result["snapshot_s3_key"] is not None
        assert "snapshots/" in ds_result["snapshot_s3_key"]

        # Check report key is present
        assert ds_result["report_s3_key"] is not None
        assert "reports/" in ds_result["report_s3_key"]

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    @patch("lambda_source.dataset_lock.boto3")
    def test_apply_mode_snapshot_before_modification(
        self,
        mock_lock_boto3,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """Snapshot S3 put should happen before UpdateDataSet is called."""
        columns = [
            {"Name": "col_a", "Type": "STRING"},
            {"Name": "col_unused", "Type": "STRING"},
        ]

        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        analysis_def = _analysis_definition(DATASET_ID_1, ["col_a"])

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs.update_data_set.return_value = {"Status": 200}
        mock_qs.create_ingestion.return_value = {"Status": 201}
        mock_qs_boto3.client.return_value = mock_qs

        # Track call order
        call_order = []

        mock_s3 = MagicMock()

        def track_s3_put(**kwargs):
            call_order.append(("s3_put", kwargs.get("Key", "")))
            return {}

        mock_s3.put_object.side_effect = track_s3_put
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        def track_update(**kwargs):
            call_order.append(("update_dataset",))
            return {"Status": 200}

        mock_qs.update_data_set.side_effect = track_update

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="apply",
        )

        handler(event, None)

        # Find the positions of snapshot and update_dataset in call_order
        snapshot_indices = [
            i for i, c in enumerate(call_order)
            if c[0] == "s3_put" and "snapshots/" in c[1]
        ]
        update_indices = [
            i for i, c in enumerate(call_order)
            if c[0] == "update_dataset"
        ]

        assert len(snapshot_indices) >= 1, "Snapshot should have been stored"
        assert len(update_indices) >= 1, "UpdateDataSet should have been called"
        # Snapshot should be before modification
        assert snapshot_indices[0] < update_indices[0], (
            "Snapshot must be captured before UpdateDataSet"
        )


# ---------------------------------------------------------------------------
# Test 3: Partial failure handling
# ---------------------------------------------------------------------------


class TestPartialFailureHandling:
    """Test that one dataset failure does not block others.

    Configures three target datasets where one fails (DescribeDataSet throws)
    and verifies the other two still process successfully.

    Requirements: 8.1
    """

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_three_datasets_one_fails_others_succeed(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """With 3 datasets where 1 fails, the other 2 should process successfully."""
        from botocore.exceptions import ClientError

        # Dataset 1 columns (will succeed - has unused columns)
        columns_1 = [
            {"Name": "col_used", "Type": "STRING"},
            {"Name": "col_unused", "Type": "INTEGER"},
        ]
        dataset_def_1 = _dataset_definition(DATASET_ID_1, columns_1)
        analysis_def_1 = _analysis_definition(DATASET_ID_1, ["col_used"])

        # Dataset 3 columns (will succeed - has unused columns)
        columns_3 = [
            {"Name": "col_x", "Type": "STRING"},
            {"Name": "col_y_unused", "Type": "DECIMAL"},
        ]
        dataset_def_3 = _dataset_definition(DATASET_ID_3, columns_3)
        analysis_def_3 = _analysis_definition(DATASET_ID_3, ["col_x"])

        # Dataset 2 will fail on DescribeDataSet
        mock_qs = MagicMock()

        def describe_side_effect(**kwargs):
            dataset_id = kwargs.get("DataSetId", "")
            if dataset_id == DATASET_ID_2:
                raise ClientError(
                    {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}},
                    "DescribeDataSet",
                )
            elif dataset_id == DATASET_ID_1:
                return {"DataSet": dataset_def_1}
            elif dataset_id == DATASET_ID_3:
                return {"DataSet": dataset_def_3}
            return {"DataSet": {}}

        mock_qs.describe_data_set.side_effect = describe_side_effect
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}

        def describe_analysis_side_effect(**kwargs):
            # Build a combined definition that references all datasets
            return {
                "Definition": {
                    "DataSetIdentifierDeclarations": [
                        {
                            "Identifier": f"ds-ident-{DATASET_ID_1}",
                            "DataSetArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{DATASET_ID_1}",
                        },
                        {
                            "Identifier": f"ds-ident-{DATASET_ID_3}",
                            "DataSetArn": f"arn:aws:quicksight:{REGION}:{ACCOUNT_ID}:dataset/{DATASET_ID_3}",
                        },
                    ],
                    "Sheets": [
                        {
                            "SheetId": "sheet-1",
                            "Name": "Sheet 1",
                            "Visuals": [
                                {
                                    "BarChartVisual": {
                                        "VisualId": "visual-1",
                                        "ChartConfiguration": {
                                            "FieldWells": {
                                                "BarChartAggregatedFieldWells": {
                                                    "Category": [
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f1",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_1}",
                                                                    "ColumnName": "col_used",
                                                                },
                                                            }
                                                        },
                                                        {
                                                            "CategoricalDimensionField": {
                                                                "FieldId": "f2",
                                                                "Column": {
                                                                    "DataSetIdentifier": f"ds-ident-{DATASET_ID_3}",
                                                                    "ColumnName": "col_x",
                                                                },
                                                            }
                                                        },
                                                    ],
                                                    "Values": [],
                                                }
                                            }
                                        },
                                    }
                                }
                            ],
                        }
                    ],
                    "CalculatedFields": [],
                    "FilterGroups": [],
                    "ParameterDeclarations": [],
                }
            }

        mock_qs.describe_analysis_definition.side_effect = describe_analysis_side_effect
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1, DATASET_ID_2, DATASET_ID_3],
            execution_mode="dry_run",
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:test-topic",
        )

        result = handler(event, None)

        # Should have partial failure status
        assert result["status"] == "partial_failure"
        assert result["datasets_processed"] == 2
        assert result["datasets_failed"] == 1

        # Check all 3 individual results are present
        assert len(result["dataset_results"]) == 3

        # Find the failed and succeeded results
        failed_result = next(
            r for r in result["dataset_results"] if r["dataset_id"] == DATASET_ID_2
        )
        success_result_1 = next(
            r for r in result["dataset_results"] if r["dataset_id"] == DATASET_ID_1
        )
        success_result_3 = next(
            r for r in result["dataset_results"] if r["dataset_id"] == DATASET_ID_3
        )

        assert failed_result["status"] == "failed"
        assert "error_message" in failed_result
        # "scanned", not "optimized". This is a dry run: it described three datasets and
        # modified none of them. Reporting "optimized" here is what let a read-only scan
        # come back claiming columns had been removed and capacity freed.
        assert success_result_1["status"] == "scanned"
        assert success_result_3["status"] == "scanned"

        # Error notification should have been sent for the failed dataset
        mock_sns.publish.assert_called()

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_failure_sends_error_notification(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """Failed datasets should trigger error notification via SNS."""
        from botocore.exceptions import ClientError

        mock_qs = MagicMock()
        mock_qs.describe_data_set.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "DescribeDataSet",
        )
        mock_qs.list_analyses.return_value = {"AnalysisSummaryList": []}
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(target_datasets=[DATASET_ID_1])
        event["sns_topic_arn"] = "arn:aws:sns:us-east-1:123:test-topic"

        result = handler(event, None)

        assert result["datasets_failed"] == 1
        # SNS publish should be called for error notification
        mock_sns.publish.assert_called()


# ---------------------------------------------------------------------------
# Test 4: Safety threshold halt behaviour
# ---------------------------------------------------------------------------


class TestSafetyThresholdHalt:
    """Test that optimization halts when safety threshold is breached.

    Configures a dataset where 90% of columns are unused and threshold is 0.7,
    verifying that optimization is halted and threshold_breached is reported.

    Requirements: 11.3
    """

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_threshold_breach_halts_optimization(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """When 90% unused columns exceed 0.7 threshold, status is threshold_breached."""
        # 10 columns, only 1 used → 90% unused, threshold set to 0.7 (70%)
        columns = [{"Name": f"col_{i}", "Type": "STRING"} for i in range(10)]
        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        # Only col_0 is used
        analysis_def = _analysis_definition(DATASET_ID_1, ["col_0"])

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="apply",
            safety_threshold_pct=0.7,  # 70% threshold, 90% unused → breached
        )

        result = handler(event, None)

        assert result["status"] == "success"
        assert result["datasets_processed"] == 1

        ds_result = result["dataset_results"][0]
        assert ds_result["status"] == "threshold_breached"

        # No write APIs should be called when threshold is breached
        mock_qs.update_data_set.assert_not_called()
        mock_qs.create_ingestion.assert_not_called()

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    @patch("lambda_source.dataset_lock.boto3")
    def test_threshold_not_breached_allows_optimization(
        self,
        mock_lock_boto3,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """When unused columns are within threshold, optimization proceeds."""
        # 10 columns, 8 used → 20% unused, threshold 0.5 (50%)
        columns = [{"Name": f"col_{i}", "Type": "STRING"} for i in range(10)]
        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        used_cols = [f"col_{i}" for i in range(8)]
        analysis_def = _analysis_definition(DATASET_ID_1, used_cols)

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs.update_data_set.return_value = {"Status": 200}
        mock_qs.create_ingestion.return_value = {"Status": 201}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="apply",
            safety_threshold_pct=0.5,
        )

        result = handler(event, None)

        assert result["status"] == "success"
        ds_result = result["dataset_results"][0]
        assert ds_result["status"] == "optimized"

        # Write APIs should be called
        mock_qs.update_data_set.assert_called_once()
        mock_qs.create_ingestion.assert_called_once()

    @patch("lambda_source.handler.boto3")
    @patch("lambda_source.snapshot.boto3")
    @patch("lambda_source.reporter.boto3")
    @patch("lambda_source.notifications.boto3")
    @patch("lambda_source.qs_client.boto3")
    def test_threshold_breach_still_generates_report(
        self,
        mock_qs_boto3,
        mock_notif_boto3,
        mock_reporter_boto3,
        mock_snapshot_boto3,
        mock_handler_boto3,
    ):
        """Even when threshold is breached, a report should be stored to S3."""
        columns = [{"Name": f"col_{i}", "Type": "STRING"} for i in range(10)]
        dataset_def = _dataset_definition(DATASET_ID_1, columns)
        analysis_def = _analysis_definition(DATASET_ID_1, ["col_0"])

        mock_qs = MagicMock()
        mock_qs.describe_data_set.return_value = {"DataSet": dataset_def}
        mock_qs.list_analyses.return_value = {
            "AnalysisSummaryList": [_analysis_summary("analysis-1")],
        }
        mock_qs.list_dashboards.return_value = {"DashboardSummaryList": []}
        mock_qs.describe_analysis_definition.return_value = {"Definition": analysis_def}
        mock_qs_boto3.client.return_value = mock_qs

        mock_s3 = MagicMock()
        mock_s3.put_object.return_value = {}
        mock_reporter_boto3.client.return_value = mock_s3
        mock_snapshot_boto3.client.return_value = mock_s3

        mock_cw = MagicMock()
        mock_handler_boto3.client.return_value = mock_cw

        mock_sns = MagicMock()
        mock_notif_boto3.client.return_value = mock_sns

        event = _make_event(
            target_datasets=[DATASET_ID_1],
            execution_mode="dry_run",
            safety_threshold_pct=0.2,
        )

        result = handler(event, None)

        ds_result = result["dataset_results"][0]
        assert ds_result["status"] == "threshold_breached"
        assert ds_result["report_s3_key"] is not None
        # Verify S3 was called to store the report
        mock_s3.put_object.assert_called()
