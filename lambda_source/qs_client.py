# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""QuickSight API wrapper with exponential backoff retry logic.

Provides a high-level interface to QuickSight APIs used by the SPICE Dataset
Optimiser. Implements retry strategies differentiated by error type:
- Throttling (429 / ThrottlingException): exponential backoff, max 5 retries
- Internal Server Error (5xx): exponential backoff, max 3 retries
- Network timeout: 3 retries with increasing timeout
- Access Denied: fail fast
- Resource Not Found: fail fast (caller decides to skip)
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

import boto3
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

logger = logging.getLogger(__name__)

# Below this, the SDK does not recognise the FileSource variant of PhysicalTableMap.
# It does not fail: DescribeDataSet returns a hollow but well-formed response for
# file-upload and new-experience datasets — one empty PhysicalTableMap entry, no
# DataPrepConfiguration — so a healthy dataset reads as "empty" and the pre-flight guard
# looks like it is misfiring. Cost real debugging time on 2026-09-11 via an unpatched CLI.
MINIMUM_BOTO3_VERSION = (1, 43, 64)


def _parse_version(raw: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in (raw or "").split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_boto3_version() -> str:
    """Warn loudly if the bundled SDK is too old to read every dataset shape.

    A warning rather than a hard failure: the optimiser still works correctly on legacy
    relational datasets with an older SDK, so refusing to start would be worse than the
    problem. The point is that the failure mode is otherwise silent and misleading, and a
    log line naming the version turns a confusing empty-dataset result into an obvious
    cause.

    Returns:
        Empty string when the version is acceptable, otherwise the warning message.
    """
    actual = _parse_version(getattr(boto3, "__version__", ""))
    if actual and actual >= MINIMUM_BOTO3_VERSION:
        return ""

    required = ".".join(str(p) for p in MINIMUM_BOTO3_VERSION)
    message = (
        f"boto3 {getattr(boto3, '__version__', 'unknown')} is below the required "
        f"{required}. File-upload and new-experience datasets will be read as empty "
        "rather than failing, because this SDK does not recognise the FileSource "
        "PhysicalTableMap variant. Results for those datasets cannot be trusted."
    )
    logger.warning(message, extra={"operation": "boto3_version_check"})
    return message


# Runs at import so it appears once per cold start, before any dataset is read.
check_boto3_version()


class QSClientError(Exception):
    """Base exception for QSClient errors."""

    def __init__(self, message: str, error_code: str | None = None, api_name: str | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.api_name = api_name


class AccessDeniedError(QSClientError):
    """Raised when an API call returns Access Denied."""

    pass


class ResourceNotFoundError(QSClientError):
    """Raised when a requested resource does not exist."""

    pass


class ThrottlingError(QSClientError):
    """Raised when retries are exhausted for a throttled API call."""

    pass


class ServerError(QSClientError):
    """Raised when retries are exhausted for 5xx server errors."""

    pass


class NetworkTimeoutError(QSClientError):
    """Raised when retries are exhausted for network timeouts."""

    pass


# Error codes that indicate throttling
_THROTTLE_CODES = frozenset({
    "Throttling",
    "ThrottlingException",
    "TooManyRequestsException",
    "RequestThrottled",
    "LimitExceededException",
})

# Error codes that indicate a non-retryable access issue
_ACCESS_DENIED_CODES = frozenset({
    "AccessDeniedException",
    "UnauthorizedOperation",
    "AccessDenied",
})

# Error codes that indicate the resource was not found
_NOT_FOUND_CODES = frozenset({
    "ResourceNotFoundException",
    "ResourceNotFound",
})


class PaginationError(QSClientError):
    """Raised when a paginated listing cannot be completed safely.

    Deliberately an error rather than a truncated result. A short
    ``list_datasets`` makes child discovery miss children, which makes their parent's
    columns look unreferenced, which puts them in a removal list. Silent truncation is
    therefore a data-loss path, and failing loudly is the only safe response.
    """

    pass


class QSClient:
    """QuickSight API wrapper with exponential backoff retry."""

    MAX_RETRIES = 5
    BASE_DELAY = 1.0  # seconds
    MAX_SERVER_ERROR_RETRIES = 3
    MAX_TIMEOUT_RETRIES = 3

    # Pages, not items. QuickSight list APIs return up to 100 per page, so this allows
    # ~20,000 resources in one account — far beyond the 500-dataset accounts this tool
    # targets, while still bounding the loop.
    MAX_PAGES = 200

    def __init__(self, account_id: str, region: str, client: Any | None = None):
        """Initialize QSClient.

        Args:
            account_id: AWS account ID for QuickSight API calls.
            region: AWS region for QuickSight API calls.
            client: Optional pre-configured boto3 QuickSight client (for testing).
        """
        self._account_id = account_id
        self._region = region
        self._client = client or boto3.client("quicksight", region_name=region)
        self._dataset_cache: dict[str, dict] = {}

    @property
    def account_id(self) -> str:
        """AWS account ID used for API calls."""
        return self._account_id

    def _paginate(
        self, api_func: Callable, result_key: str, **kwargs: Any
    ) -> list[dict]:
        """Collect every page of a paginated list API.

        Replaces three hand-rolled ``while True`` loops that each trusted
        ``response["NextToken"]`` unconditionally, with no iteration cap and no check
        that the token was even a string. Any response that kept yielding a truthy
        token looped until the Lambda's 900s timeout killed it — and a job killed that
        way never wrote a callback, so it presented to the caller as a job stuck on
        ``running``.

        That is also, precisely, why ``tests/integration`` never completed. The suite
        predates child discovery, so it never mocked ``list_data_sets``; the resulting
        bare ``MagicMock`` returned a truthy token forever. The guard below fixes both
        the test symptom and the production failure mode, because they are the same
        defect.

        Three bounds, in order of how likely they are to matter:

        1. The token must be a non-empty ``str``. The API contract says it is one, so
           ``None``, ``""``, a number, or a mock all mean "no more pages".
        2. A token already seen means the service is not advancing. Stop rather than
           cycle.
        3. A hard page cap, as a backstop against a token stream that never repeats.

        Args:
            api_func: The bound client method to call.
            result_key: Response key holding the page's items.
            **kwargs: Initial request arguments; ``NextToken`` is managed here.

        Returns:
            Every item across all pages.

        Raises:
            PaginationError: If the page cap is reached, since returning a partial
                listing would be worse than failing.
        """
        items: list[dict] = []
        seen_tokens: set[str] = set()
        request = dict(kwargs)
        api_name = getattr(api_func, "__name__", str(api_func))

        for page in range(1, self.MAX_PAGES + 1):
            response = self._call_with_retry(api_func, **request)

            page_items = response.get(result_key)
            if isinstance(page_items, list):
                items.extend(page_items)

            token = response.get("NextToken")
            if not isinstance(token, str) or not token:
                return items

            if token in seen_tokens:
                logger.warning(
                    "Pagination token repeated, stopping to avoid a cycle",
                    extra={
                        "api_name": api_name,
                        "pages_read": page,
                        "items_collected": len(items),
                    },
                )
                return items

            seen_tokens.add(token)
            request["NextToken"] = token

        raise PaginationError(
            f"{api_name} exceeded {self.MAX_PAGES} pages "
            f"({len(items)} items collected). Refusing to return a partial listing, "
            "because a short listing silently becomes a removal candidate downstream.",
            error_code="PaginationLimitExceeded",
            api_name=api_name,
        )

    def clear_cache(self) -> None:
        """Clear the dataset definition cache."""
        self._dataset_cache.clear()

    def describe_dataset(self, dataset_id: str) -> dict:
        """Retrieve the full definition of a dataset.

        Results are cached for the lifetime of this client instance to avoid
        redundant API calls during recursive traversal.

        Args:
            dataset_id: QuickSight dataset identifier.

        Returns:
            The DataSet object from the API response.
        """
        if dataset_id in self._dataset_cache:
            return self._dataset_cache[dataset_id]

        response = self._call_with_retry(
            self._client.describe_data_set,
            AwsAccountId=self._account_id,
            DataSetId=dataset_id,
        )
        result = response.get("DataSet", response)
        self._dataset_cache[dataset_id] = result
        return result

    def update_dataset(self, dataset_id: str, definition: dict) -> dict:
        """Update a dataset definition.

        Args:
            dataset_id: QuickSight dataset identifier.
            definition: Complete dataset definition kwargs for UpdateDataSet.

        Returns:
            API response dict.
        """
        # Merge account ID into the definition params
        params = {**definition}
        params["AwsAccountId"] = self._account_id
        params["DataSetId"] = dataset_id

        return self._call_with_retry(self._client.update_data_set, **params)

    def create_ingestion(self, dataset_id: str, ingestion_id: str) -> dict:
        """Trigger a SPICE ingestion for a dataset.

        Args:
            dataset_id: QuickSight dataset identifier.
            ingestion_id: Unique ID for the new ingestion.

        Returns:
            API response dict.
        """
        return self._call_with_retry(
            self._client.create_ingestion,
            AwsAccountId=self._account_id,
            DataSetId=dataset_id,
            IngestionId=ingestion_id,
        )

    def list_analyses(self) -> list[dict]:
        """Return all analyses with pagination handling.

        Returns:
            List of analysis summary dicts.
        """
        return self._paginate(
            self._client.list_analyses,
            "AnalysisSummaryList",
            AwsAccountId=self._account_id,
        )

    def list_dashboards(self) -> list[dict]:
        """Return all dashboards with pagination handling.

        Returns:
            List of dashboard summary dicts.
        """
        return self._paginate(
            self._client.list_dashboards,
            "DashboardSummaryList",
            AwsAccountId=self._account_id,
        )

    def list_datasets(self) -> list[dict]:
        """Return all dataset summaries with pagination handling.

        Returns:
            List of dataset summary dicts from the QuickSight API.
            Each dict contains DataSetId, Name, ImportMode, and other metadata.
        """
        return self._paginate(
            self._client.list_data_sets,
            "DataSetSummaries",
            AwsAccountId=self._account_id,
        )

    def describe_analysis_definition(self, analysis_id: str) -> dict:
        """Retrieve the full definition of an analysis.

        Args:
            analysis_id: QuickSight analysis identifier.

        Returns:
            The Definition object from the API response.
        """
        response = self._call_with_retry(
            self._client.describe_analysis_definition,
            AwsAccountId=self._account_id,
            AnalysisId=analysis_id,
        )
        return response.get("Definition", response)

    def describe_dashboard_definition(self, dashboard_id: str) -> dict:
        """Retrieve the full definition of a dashboard.

        Args:
            dashboard_id: QuickSight dashboard identifier.

        Returns:
            The Definition object from the API response.
        """
        response = self._call_with_retry(
            self._client.describe_dashboard_definition,
            AwsAccountId=self._account_id,
            DashboardId=dashboard_id,
        )
        return response.get("Definition", response)

    def _call_with_retry(self, api_func: Callable, **kwargs: Any) -> dict:
        """Execute an API call with exponential backoff on throttling and server errors.

        Retry strategy:
        - Throttling errors: up to MAX_RETRIES (5) with exponential backoff + jitter
        - 5xx server errors: up to MAX_SERVER_ERROR_RETRIES (3) with same backoff
        - Network timeouts: up to MAX_TIMEOUT_RETRIES (3) with increasing timeout

        Non-retryable errors (Access Denied, Resource Not Found) are raised immediately.

        Args:
            api_func: The boto3 API method to call.
            **kwargs: Arguments to pass to the API method.

        Returns:
            The API response dict.

        Raises:
            AccessDeniedError: On access denied responses.
            ResourceNotFoundError: On resource not found responses.
            ThrottlingError: When throttling retries are exhausted.
            ServerError: When server error retries are exhausted.
            NetworkTimeoutError: When timeout retries are exhausted.
        """
        api_name = getattr(api_func, "__name__", None)
        if not isinstance(api_name, str):
            api_name = str(api_func)
        throttle_attempts = 0
        server_error_attempts = 0
        timeout_attempts = 0

        while True:
            try:
                response = api_func(**kwargs)
                return response

            except (ConnectTimeoutError, ReadTimeoutError) as exc:
                timeout_attempts += 1
                if timeout_attempts >= self.MAX_TIMEOUT_RETRIES:
                    logger.error(
                        "Network timeout exhausted retries",
                        extra={
                            "api_name": api_name,
                            "attempts": timeout_attempts,
                            "error": str(exc),
                        },
                    )
                    raise NetworkTimeoutError(
                        f"Network timeout after {timeout_attempts} retries calling {api_name}: {exc}",
                        error_code="NetworkTimeout",
                        api_name=api_name,
                    ) from exc

                delay = self._compute_delay(timeout_attempts)
                logger.warning(
                    "Network timeout, retrying",
                    extra={
                        "api_name": api_name,
                        "attempt": timeout_attempts,
                        "delay_seconds": delay,
                    },
                )
                time.sleep(delay)

            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code", "Unknown")
                error_message = exc.response.get("Error", {}).get("Message", str(exc))
                http_status = exc.response.get("ResponseMetadata", {}).get(
                    "HTTPStatusCode", 0
                )

                # Access Denied — fail fast
                if error_code in _ACCESS_DENIED_CODES:
                    logger.error(
                        "Access denied",
                        extra={
                            "api_name": api_name,
                            "error_code": error_code,
                            "error_message": error_message,
                        },
                    )
                    raise AccessDeniedError(
                        f"Access denied calling {api_name}: {error_message}",
                        error_code=error_code,
                        api_name=api_name,
                    ) from exc

                # Resource Not Found — fail fast
                if error_code in _NOT_FOUND_CODES:
                    logger.warning(
                        "Resource not found",
                        extra={
                            "api_name": api_name,
                            "error_code": error_code,
                            "error_message": error_message,
                        },
                    )
                    raise ResourceNotFoundError(
                        f"Resource not found calling {api_name}: {error_message}",
                        error_code=error_code,
                        api_name=api_name,
                    ) from exc

                # Throttling — retry with exponential backoff
                if error_code in _THROTTLE_CODES:
                    throttle_attempts += 1
                    if throttle_attempts >= self.MAX_RETRIES:
                        logger.error(
                            "Throttling retries exhausted",
                            extra={
                                "api_name": api_name,
                                "attempts": throttle_attempts,
                            },
                        )
                        raise ThrottlingError(
                            f"Throttling retries exhausted after {throttle_attempts} attempts calling {api_name}",
                            error_code=error_code,
                            api_name=api_name,
                        ) from exc

                    delay = self._compute_delay(throttle_attempts)
                    logger.warning(
                        "Throttled, retrying",
                        extra={
                            "api_name": api_name,
                            "attempt": throttle_attempts,
                            "delay_seconds": delay,
                        },
                    )
                    time.sleep(delay)
                    continue

                # 5xx Server Error — retry with backoff
                if 500 <= http_status < 600:
                    server_error_attempts += 1
                    if server_error_attempts >= self.MAX_SERVER_ERROR_RETRIES:
                        logger.error(
                            "Server error retries exhausted",
                            extra={
                                "api_name": api_name,
                                "attempts": server_error_attempts,
                                "http_status": http_status,
                            },
                        )
                        raise ServerError(
                            f"Server error retries exhausted after {server_error_attempts} attempts calling {api_name}: {error_message}",
                            error_code=error_code,
                            api_name=api_name,
                        ) from exc

                    delay = self._compute_delay(server_error_attempts)
                    logger.warning(
                        "Server error, retrying",
                        extra={
                            "api_name": api_name,
                            "attempt": server_error_attempts,
                            "http_status": http_status,
                            "delay_seconds": delay,
                        },
                    )
                    time.sleep(delay)
                    continue

                # Unknown / unclassified error — fail fast
                logger.error(
                    "Unhandled API error",
                    extra={
                        "api_name": api_name,
                        "error_code": error_code,
                        "http_status": http_status,
                        "error_message": error_message,
                    },
                )
                raise QSClientError(
                    f"Unhandled error calling {api_name}: [{error_code}] {error_message}",
                    error_code=error_code,
                    api_name=api_name,
                ) from exc

    @staticmethod
    def _compute_delay(attempt: int) -> float:
        """Compute exponential backoff delay with full jitter.

        delay = random(0, base * 2^attempt)

        Args:
            attempt: The retry attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        max_delay = QSClient.BASE_DELAY * (2 ** attempt)
        return random.uniform(0, max_delay)
