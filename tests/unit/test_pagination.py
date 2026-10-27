# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for QSClient._paginate.

Three hand-rolled ``while True`` loops each trusted ``response["NextToken"]``
unconditionally, with no iteration cap and no check that the token was even a string.
Any response that kept yielding a truthy token looped until the Lambda's 900s timeout
killed it — and a job killed that way never wrote a callback, so it reached the caller
as a job stuck on ``running``.

It is also why ``tests/integration`` never completed once, for weeks. That suite
predates child discovery so it never mocked ``list_data_sets``; the bare ``MagicMock``
returned a truthy token forever. Test symptom and production failure mode were the
same defect.

The bound that matters most is the type check: the API contract says ``NextToken`` is a
string, so anything else means "no more pages". That alone fixes both.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lambda_source.qs_client import PaginationError, QSClient


def make_client(api_responses: list[dict]) -> tuple[QSClient, MagicMock]:
    """A QSClient whose list_data_sets returns the given pages in order."""
    boto_client = MagicMock()
    boto_client.list_data_sets.side_effect = api_responses
    return QSClient(account_id="9" * 12, region="us-east-1", client=boto_client), boto_client


class TestTokenTypeGuard:
    def test_a_bare_magicmock_terminates_instead_of_looping(self):
        """The exact shape that hung the integration suite: an unconfigured mock whose
        .get() returns a truthy MagicMock for every key."""
        boto_client = MagicMock()  # list_data_sets not configured at all
        client = QSClient(account_id="9" * 12, region="us-east-1", client=boto_client)

        result = client.list_datasets()

        assert result == []
        assert boto_client.list_data_sets.call_count == 1

    @pytest.mark.parametrize("token", [None, "", 0, 123, [], {}, True, MagicMock()])
    def test_a_non_string_token_means_no_more_pages(self, token):
        client, boto_client = make_client(
            [{"DataSetSummaries": [{"DataSetId": "a"}], "NextToken": token}]
        )

        result = client.list_datasets()

        assert [d["DataSetId"] for d in result] == ["a"]
        assert boto_client.list_data_sets.call_count == 1

    def test_a_missing_token_key_means_no_more_pages(self):
        client, _ = make_client([{"DataSetSummaries": [{"DataSetId": "a"}]}])
        assert len(client.list_datasets()) == 1


class TestNormalPagination:
    def test_follows_a_multi_page_listing(self):
        client, boto_client = make_client(
            [
                {"DataSetSummaries": [{"DataSetId": "a"}], "NextToken": "t1"},
                {"DataSetSummaries": [{"DataSetId": "b"}], "NextToken": "t2"},
                {"DataSetSummaries": [{"DataSetId": "c"}]},
            ]
        )

        result = client.list_datasets()

        assert [d["DataSetId"] for d in result] == ["a", "b", "c"]
        assert boto_client.list_data_sets.call_count == 3

    def test_forwards_the_token_on_the_next_request(self):
        client, boto_client = make_client(
            [
                {"DataSetSummaries": [], "NextToken": "t1"},
                {"DataSetSummaries": []},
            ]
        )
        client.list_datasets()

        second_call = boto_client.list_data_sets.call_args_list[1]
        assert second_call[1]["NextToken"] == "t1"

    def test_a_non_list_result_key_is_skipped_not_fatal(self):
        """A malformed page should not take down the whole run."""
        client, _ = make_client(
            [
                {"DataSetSummaries": "not-a-list", "NextToken": "t1"},
                {"DataSetSummaries": [{"DataSetId": "b"}]},
            ]
        )

        assert [d["DataSetId"] for d in client.list_datasets()] == ["b"]

    def test_an_empty_listing_returns_empty(self):
        client, _ = make_client([{"DataSetSummaries": []}])
        assert client.list_datasets() == []


class TestCycleGuard:
    def test_a_repeated_token_stops_rather_than_cycling(self):
        """A service that stops advancing should not spin the caller forever."""
        client, boto_client = make_client(
            [
                {"DataSetSummaries": [{"DataSetId": "a"}], "NextToken": "same"},
                {"DataSetSummaries": [{"DataSetId": "b"}], "NextToken": "same"},
            ]
        )

        result = client.list_datasets()

        assert [d["DataSetId"] for d in result] == ["a", "b"]
        assert boto_client.list_data_sets.call_count == 2

    def test_a_two_token_cycle_is_caught(self):
        client, _ = make_client(
            [
                {"DataSetSummaries": [], "NextToken": "t1"},
                {"DataSetSummaries": [], "NextToken": "t2"},
                {"DataSetSummaries": [], "NextToken": "t1"},
            ]
        )

        # Terminates rather than alternating indefinitely.
        assert client.list_datasets() == []


class TestPageCap:
    def test_an_endless_unique_token_stream_raises(self):
        """The backstop. Reached only if every token is new, which no real service
        does, but an unbounded loop is what the 900s timeout used to absorb."""
        boto_client = MagicMock()
        counter = {"n": 0}

        def endless(**kwargs):
            counter["n"] += 1
            return {"DataSetSummaries": [], "NextToken": f"token-{counter['n']}"}

        boto_client.list_data_sets.side_effect = endless
        client = QSClient(account_id="9" * 12, region="us-east-1", client=boto_client)

        with pytest.raises(PaginationError) as exc:
            client.list_datasets()

        assert "exceeded" in str(exc.value)
        assert counter["n"] == QSClient.MAX_PAGES

    def test_the_cap_raises_rather_than_truncating(self):
        """Returning a short listing is the dangerous option: child discovery would
        miss children, so their parent's columns would look unreferenced and land in a
        removal list. Failing loudly is the only safe response."""
        boto_client = MagicMock()
        counter = {"n": 0}

        def endless_with_items(**kwargs):
            counter["n"] += 1
            return {
                "DataSetSummaries": [{"DataSetId": f"ds-{counter['n']}"}],
                "NextToken": f"unique-{counter['n']}",
            }

        boto_client.list_data_sets.side_effect = endless_with_items
        client = QSClient(account_id="9" * 12, region="us-east-1", client=boto_client)

        with pytest.raises(PaginationError) as exc:
            client.list_datasets()

        # The message reports what was collected, so an operator can see the scale of
        # what would have been silently dropped.
        assert str(QSClient.MAX_PAGES) in str(exc.value)

    def test_pagination_error_is_a_qsclient_error(self):
        """So existing `except QSClientError` handlers catch it."""
        from lambda_source.qs_client import QSClientError

        assert issubclass(PaginationError, QSClientError)


class TestAllListingsUseTheGuard:
    @pytest.mark.parametrize(
        "method,api_attr,result_key",
        [
            ("list_datasets", "list_data_sets", "DataSetSummaries"),
            ("list_analyses", "list_analyses", "AnalysisSummaryList"),
            ("list_dashboards", "list_dashboards", "DashboardSummaryList"),
        ],
    )
    def test_every_paginated_listing_terminates_on_a_mock(
        self, method, api_attr, result_key
    ):
        """All three loops had the defect, so all three need the guard."""
        boto_client = MagicMock()  # nothing configured
        client = QSClient(account_id="9" * 12, region="us-east-1", client=boto_client)

        assert getattr(client, method)() == []
        assert getattr(boto_client, api_attr).call_count == 1

    @pytest.mark.parametrize(
        "method,api_attr,result_key",
        [
            ("list_datasets", "list_data_sets", "DataSetSummaries"),
            ("list_analyses", "list_analyses", "AnalysisSummaryList"),
            ("list_dashboards", "list_dashboards", "DashboardSummaryList"),
        ],
    )
    def test_every_paginated_listing_follows_multiple_pages(
        self, method, api_attr, result_key
    ):
        boto_client = MagicMock()
        getattr(boto_client, api_attr).side_effect = [
            {result_key: [{"Id": "a"}], "NextToken": "t1"},
            {result_key: [{"Id": "b"}]},
        ]
        client = QSClient(account_id="9" * 12, region="us-east-1", client=boto_client)

        assert len(getattr(client, method)()) == 2
