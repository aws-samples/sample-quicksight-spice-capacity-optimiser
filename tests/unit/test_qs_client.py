# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for QSClient wrapper with retry logic.

Tests validate:
- Exponential backoff with jitter for throttling errors (max 5 retries)
- Non-retryable error handling (Access Denied, Resource Not Found)
- Proper error classification and logging
- All API methods delegate correctly to _call_with_retry
- Pagination handling in list_analyses and list_dashboards

Requirements: 14.1, 14.2
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from lambda_source.qs_client import (
    AccessDeniedError,
    NetworkTimeoutError,
    QSClient,
    QSClientError,
    ResourceNotFoundError,
    ServerError,
    ThrottlingError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client_error(code: str, message: str = "error", http_status: int = 400) -> ClientError:
    """Create a botocore ClientError with the given error code and HTTP status."""
    return ClientError(
        error_response={
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": http_status},
        },
        operation_name="TestOperation",
    )


@pytest.fixture
def mock_boto_client():
    """Create a mock boto3 QuickSight client."""
    return MagicMock()


@pytest.fixture
def qs_client(mock_boto_client):
    """Create a QSClient with injected mock boto3 client."""
    return QSClient(account_id="123456789012", region="us-east-1", client=mock_boto_client)


# ---------------------------------------------------------------------------
# Retry Logic Tests
# ---------------------------------------------------------------------------


class TestRetryLogicThrottling:
    """Test exponential backoff with jitter for throttling errors (Req 14.1)."""

    @patch("lambda_source.qs_client.time.sleep")
    def test_throttling_retries_then_succeeds(self, mock_sleep, qs_client, mock_boto_client):
        """Throttled call succeeds after retries."""
        throttle_error = _make_client_error("ThrottlingException")
        mock_boto_client.describe_data_set.side_effect = [
            throttle_error,
            throttle_error,
            {"DataSet": {"DataSetId": "ds-1"}},
        ]

        result = qs_client.describe_dataset("ds-1")

        assert result == {"DataSetId": "ds-1"}
        assert mock_boto_client.describe_data_set.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("lambda_source.qs_client.time.sleep")
    def test_throttling_exhausts_max_retries(self, mock_sleep, qs_client, mock_boto_client):
        """ThrottlingError raised after MAX_RETRIES (5) throttle failures."""
        throttle_error = _make_client_error("ThrottlingException")
        # Need 5 throttle errors: attempts 1-4 sleep and retry, attempt 5 raises
        mock_boto_client.describe_data_set.side_effect = [throttle_error] * 5

        with pytest.raises(ThrottlingError) as exc_info:
            qs_client.describe_dataset("ds-1")

        assert "retries exhausted" in str(exc_info.value).lower()
        assert exc_info.value.error_code == "ThrottlingException"
        # 4 sleeps (attempts 1-4 sleep, attempt 5 immediately raises)
        assert mock_sleep.call_count == 4

    @patch("lambda_source.qs_client.time.sleep")
    def test_various_throttle_codes_are_retried(self, mock_sleep, qs_client, mock_boto_client):
        """All throttle error codes trigger retry behaviour."""
        throttle_codes = [
            "Throttling",
            "ThrottlingException",
            "TooManyRequestsException",
            "RequestThrottled",
            "LimitExceededException",
        ]

        for code in throttle_codes:
            mock_boto_client.describe_data_set.reset_mock()
            mock_sleep.reset_mock()
            qs_client.clear_cache()  # Clear cache so each iteration hits the mock
            mock_boto_client.describe_data_set.side_effect = [
                _make_client_error(code),
                {"DataSet": {"DataSetId": "ds-1"}},
            ]

            result = qs_client.describe_dataset("ds-1")
            assert result == {"DataSetId": "ds-1"}, f"Failed for throttle code: {code}"
            assert mock_sleep.call_count == 1

    @patch("lambda_source.qs_client.random.uniform")
    @patch("lambda_source.qs_client.time.sleep")
    def test_exponential_backoff_with_jitter(self, mock_sleep, mock_uniform, qs_client, mock_boto_client):
        """Backoff delay uses exponential formula with full jitter."""
        # Make uniform return max of range to test the formula
        mock_uniform.side_effect = lambda low, high: high
        throttle_error = _make_client_error("ThrottlingException")
        mock_boto_client.describe_data_set.side_effect = [
            throttle_error,
            throttle_error,
            throttle_error,
            {"DataSet": {"DataSetId": "ds-1"}},
        ]

        qs_client.describe_dataset("ds-1")

        # Expected max delays: 2^1=2, 2^2=4, 2^3=8
        assert mock_sleep.call_args_list[0][0][0] == 2.0
        assert mock_sleep.call_args_list[1][0][0] == 4.0
        assert mock_sleep.call_args_list[2][0][0] == 8.0


class TestRetryLogicNonRetryable:
    """Test non-retryable error handling (Req 14.2)."""

    def test_access_denied_fails_immediately(self, qs_client, mock_boto_client):
        """Access denied errors are not retried."""
        mock_boto_client.describe_data_set.side_effect = _make_client_error("AccessDeniedException")

        with pytest.raises(AccessDeniedError) as exc_info:
            qs_client.describe_dataset("ds-1")

        assert mock_boto_client.describe_data_set.call_count == 1
        assert exc_info.value.error_code == "AccessDeniedException"
        assert exc_info.value.api_name is not None

    def test_resource_not_found_fails_immediately(self, qs_client, mock_boto_client):
        """Resource not found errors are not retried."""
        mock_boto_client.describe_data_set.side_effect = _make_client_error("ResourceNotFoundException")

        with pytest.raises(ResourceNotFoundError) as exc_info:
            qs_client.describe_dataset("ds-1")

        assert mock_boto_client.describe_data_set.call_count == 1
        assert exc_info.value.error_code == "ResourceNotFoundException"

    def test_unknown_error_fails_immediately(self, qs_client, mock_boto_client):
        """Unclassified errors are not retried."""
        mock_boto_client.describe_data_set.side_effect = _make_client_error(
            "ValidationException", http_status=400
        )

        with pytest.raises(QSClientError) as exc_info:
            qs_client.describe_dataset("ds-1")

        assert mock_boto_client.describe_data_set.call_count == 1
        assert exc_info.value.error_code == "ValidationException"


class TestRetryLogicServerErrors:
    """Test 5xx server error retry handling."""

    @patch("lambda_source.qs_client.time.sleep")
    def test_server_error_retries_then_succeeds(self, mock_sleep, qs_client, mock_boto_client):
        """5xx errors are retried up to MAX_SERVER_ERROR_RETRIES."""
        server_error = _make_client_error("InternalFailure", http_status=500)
        mock_boto_client.describe_data_set.side_effect = [
            server_error,
            {"DataSet": {"DataSetId": "ds-1"}},
        ]

        result = qs_client.describe_dataset("ds-1")
        assert result == {"DataSetId": "ds-1"}
        assert mock_sleep.call_count == 1

    @patch("lambda_source.qs_client.time.sleep")
    def test_server_error_exhausts_retries(self, mock_sleep, qs_client, mock_boto_client):
        """ServerError raised after MAX_SERVER_ERROR_RETRIES (3) failures."""
        server_error = _make_client_error("InternalFailure", http_status=500)
        # 3 failures: attempts 1-2 sleep, attempt 3 raises
        mock_boto_client.describe_data_set.side_effect = [server_error] * 3

        with pytest.raises(ServerError):
            qs_client.describe_dataset("ds-1")

        assert mock_sleep.call_count == 2


class TestRetryLogicNetworkTimeouts:
    """Test network timeout retry handling."""

    @patch("lambda_source.qs_client.time.sleep")
    def test_connect_timeout_retries_then_succeeds(self, mock_sleep, qs_client, mock_boto_client):
        """ConnectTimeoutError is retried."""
        mock_boto_client.describe_data_set.side_effect = [
            ConnectTimeoutError(endpoint_url="https://quicksight.us-east-1.amazonaws.com"),
            {"DataSet": {"DataSetId": "ds-1"}},
        ]

        result = qs_client.describe_dataset("ds-1")
        assert result == {"DataSetId": "ds-1"}
        assert mock_sleep.call_count == 1

    @patch("lambda_source.qs_client.time.sleep")
    def test_read_timeout_exhausts_retries(self, mock_sleep, qs_client, mock_boto_client):
        """NetworkTimeoutError raised after MAX_TIMEOUT_RETRIES (3) failures."""
        mock_boto_client.describe_data_set.side_effect = [
            ReadTimeoutError(endpoint_url="https://quicksight.us-east-1.amazonaws.com")
        ] * 3

        with pytest.raises(NetworkTimeoutError):
            qs_client.describe_dataset("ds-1")

        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# API Method Tests
# ---------------------------------------------------------------------------


class TestDescribeDataset:
    """Test describe_dataset method."""

    def test_returns_dataset_object(self, qs_client, mock_boto_client):
        """Returns the DataSet object from response."""
        mock_boto_client.describe_data_set.return_value = {
            "DataSet": {"DataSetId": "ds-1", "Name": "Test Dataset"},
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        result = qs_client.describe_dataset("ds-1")

        assert result == {"DataSetId": "ds-1", "Name": "Test Dataset"}
        mock_boto_client.describe_data_set.assert_called_once_with(
            AwsAccountId="123456789012",
            DataSetId="ds-1",
        )


class TestUpdateDataset:
    """Test update_dataset method."""

    def test_passes_definition_with_account_and_id(self, qs_client, mock_boto_client):
        """Merges account ID and dataset ID into definition params."""
        mock_boto_client.update_data_set.return_value = {"Status": 200}
        definition = {"Name": "Updated", "PhysicalTableMap": {}}

        qs_client.update_dataset("ds-1", definition)

        mock_boto_client.update_data_set.assert_called_once_with(
            Name="Updated",
            PhysicalTableMap={},
            AwsAccountId="123456789012",
            DataSetId="ds-1",
        )


class TestCreateIngestion:
    """Test create_ingestion method."""

    def test_creates_ingestion_with_correct_params(self, qs_client, mock_boto_client):
        """Calls CreateIngestion with correct parameters."""
        mock_boto_client.create_ingestion.return_value = {
            "IngestionId": "ing-1",
            "IngestionStatus": "INITIALIZED",
        }

        result = qs_client.create_ingestion("ds-1", "ing-1")

        assert result["IngestionId"] == "ing-1"
        mock_boto_client.create_ingestion.assert_called_once_with(
            AwsAccountId="123456789012",
            DataSetId="ds-1",
            IngestionId="ing-1",
        )


class TestListAnalyses:
    """Test list_analyses method with pagination."""

    def test_single_page(self, qs_client, mock_boto_client):
        """Returns analyses from a single page."""
        mock_boto_client.list_analyses.return_value = {
            "AnalysisSummaryList": [{"AnalysisId": "a-1"}, {"AnalysisId": "a-2"}],
        }

        result = qs_client.list_analyses()

        assert len(result) == 2
        assert result[0]["AnalysisId"] == "a-1"

    def test_multiple_pages(self, qs_client, mock_boto_client):
        """Handles pagination with NextToken."""
        mock_boto_client.list_analyses.side_effect = [
            {
                "AnalysisSummaryList": [{"AnalysisId": "a-1"}],
                "NextToken": "token-1",
            },
            {
                "AnalysisSummaryList": [{"AnalysisId": "a-2"}],
            },
        ]

        result = qs_client.list_analyses()

        assert len(result) == 2
        assert mock_boto_client.list_analyses.call_count == 2


class TestListDashboards:
    """Test list_dashboards method with pagination."""

    def test_single_page(self, qs_client, mock_boto_client):
        """Returns dashboards from a single page."""
        mock_boto_client.list_dashboards.return_value = {
            "DashboardSummaryList": [{"DashboardId": "d-1"}],
        }

        result = qs_client.list_dashboards()

        assert len(result) == 1
        assert result[0]["DashboardId"] == "d-1"

    def test_multiple_pages(self, qs_client, mock_boto_client):
        """Handles pagination with NextToken."""
        mock_boto_client.list_dashboards.side_effect = [
            {
                "DashboardSummaryList": [{"DashboardId": "d-1"}],
                "NextToken": "token-1",
            },
            {
                "DashboardSummaryList": [{"DashboardId": "d-2"}, {"DashboardId": "d-3"}],
            },
        ]

        result = qs_client.list_dashboards()

        assert len(result) == 3


class TestListDatasets:
    """Test list_datasets method with pagination."""

    def test_single_page(self, qs_client, mock_boto_client):
        """Returns datasets from a single page."""
        mock_boto_client.list_data_sets.return_value = {
            "DataSetSummaries": [
                {"DataSetId": "ds-1", "Name": "Sales", "ImportMode": "SPICE"},
                {"DataSetId": "ds-2", "Name": "Orders", "ImportMode": "DIRECT_QUERY"},
            ],
        }

        result = qs_client.list_datasets()

        assert len(result) == 2
        assert result[0]["DataSetId"] == "ds-1"
        assert result[1]["DataSetId"] == "ds-2"

    def test_multiple_pages(self, qs_client, mock_boto_client):
        """Handles pagination with NextToken."""
        mock_boto_client.list_data_sets.side_effect = [
            {
                "DataSetSummaries": [{"DataSetId": "ds-1"}],
                "NextToken": "page-2-token",
            },
            {
                "DataSetSummaries": [{"DataSetId": "ds-2"}, {"DataSetId": "ds-3"}],
            },
        ]

        result = qs_client.list_datasets()

        assert len(result) == 3
        assert mock_boto_client.list_data_sets.call_count == 2
        # Second call should include NextToken
        second_call_kwargs = mock_boto_client.list_data_sets.call_args_list[1][1]
        assert second_call_kwargs["NextToken"] == "page-2-token"

    def test_empty_account(self, qs_client, mock_boto_client):
        """Returns empty list when no datasets exist."""
        mock_boto_client.list_data_sets.return_value = {
            "DataSetSummaries": [],
        }

        result = qs_client.list_datasets()

        assert result == []

    @patch("lambda_source.qs_client.time.sleep")
    def test_throttling_during_pagination(self, mock_sleep, qs_client, mock_boto_client):
        """Retries on throttling during pagination."""
        throttle_error = _make_client_error("ThrottlingException")
        mock_boto_client.list_data_sets.side_effect = [
            {
                "DataSetSummaries": [{"DataSetId": "ds-1"}],
                "NextToken": "token-1",
            },
            throttle_error,
            {
                "DataSetSummaries": [{"DataSetId": "ds-2"}],
            },
        ]

        result = qs_client.list_datasets()

        assert len(result) == 2
        assert mock_boto_client.list_data_sets.call_count == 3
        assert mock_sleep.call_count == 1

    def test_passes_account_id(self, qs_client, mock_boto_client):
        """Passes AwsAccountId to the API call."""
        mock_boto_client.list_data_sets.return_value = {
            "DataSetSummaries": [],
        }

        qs_client.list_datasets()

        mock_boto_client.list_data_sets.assert_called_once_with(
            AwsAccountId="123456789012",
        )


class TestDescribeAnalysisDefinition:
    """Test describe_analysis_definition method."""

    def test_returns_definition_object(self, qs_client, mock_boto_client):
        """Returns the Definition object from response."""
        mock_boto_client.describe_analysis_definition.return_value = {
            "Definition": {"Sheets": []},
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        result = qs_client.describe_analysis_definition("a-1")

        assert result == {"Sheets": []}
        mock_boto_client.describe_analysis_definition.assert_called_once_with(
            AwsAccountId="123456789012",
            AnalysisId="a-1",
        )


class TestDescribeDashboardDefinition:
    """Test describe_dashboard_definition method."""

    def test_returns_definition_object(self, qs_client, mock_boto_client):
        """Returns the Definition object from response."""
        mock_boto_client.describe_dashboard_definition.return_value = {
            "Definition": {"Sheets": [], "CalculatedFields": []},
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        result = qs_client.describe_dashboard_definition("d-1")

        assert result == {"Sheets": [], "CalculatedFields": []}
        mock_boto_client.describe_dashboard_definition.assert_called_once_with(
            AwsAccountId="123456789012",
            DashboardId="d-1",
        )


# ---------------------------------------------------------------------------
# Compute Delay Tests
# ---------------------------------------------------------------------------


class TestComputeDelay:
    """Test the exponential backoff delay calculation."""

    def test_delay_is_non_negative(self):
        """Delay should always be >= 0."""
        for attempt in range(1, 10):
            delay = QSClient._compute_delay(attempt)
            assert delay >= 0

    def test_delay_bounded_by_max(self):
        """Delay should not exceed base * 2^attempt."""
        for attempt in range(1, 6):
            max_expected = QSClient.BASE_DELAY * (2 ** attempt)
            delay = QSClient._compute_delay(attempt)
            assert delay <= max_expected

    def test_delay_increases_with_attempt(self):
        """Maximum possible delay increases with each attempt."""
        prev_max = 0
        for attempt in range(1, 6):
            max_delay = QSClient.BASE_DELAY * (2 ** attempt)
            assert max_delay > prev_max
            prev_max = max_delay
