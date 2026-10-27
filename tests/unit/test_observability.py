# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for correlation IDs and the boto3 version guard.

Both close out items that had been sitting in the backlog's Bugs section without being
bugs. Correlation IDs were the last unshipped piece of the deleted `mcp_server` suite's
intent — input validation, structured errors and job state transitions all arrived under
other names. The version guard turns a silent, misleading failure into a stated one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from lambda_source import mcp_dispatcher
from lambda_source.mcp_dispatcher import correlation_id_from, dispatch
from lambda_source.qs_client import (
    MINIMUM_BOTO3_VERSION,
    QSClient,
    _parse_version,
    check_boto3_version,
)


def gateway_context(**custom) -> SimpleNamespace:
    return SimpleNamespace(client_context=SimpleNamespace(custom=custom))


class TestCorrelationId:
    def test_prefers_the_gateway_request_id(self):
        """So one id spans the gateway and the Lambda rather than stopping at the
        function boundary."""
        ctx = gateway_context(bedrockAgentCoreAwsRequestId="gw-req-1")

        assert correlation_id_from(ctx) == "gw-req-1"

    def test_falls_back_to_the_lambda_request_id(self):
        ctx = SimpleNamespace(client_context=None, aws_request_id="lambda-req-1")

        assert correlation_id_from(ctx) == "lambda-req-1"

    def test_gateway_id_wins_over_lambda_id(self):
        ctx = gateway_context(bedrockAgentCoreAwsRequestId="gw-req-1")
        ctx.aws_request_id = "lambda-req-1"

        assert correlation_id_from(ctx) == "gw-req-1"

    def test_generates_one_for_a_direct_invocation(self):
        """Direct invokes pass None for context, which is normal, not an error."""
        generated = correlation_id_from(None)

        assert generated
        assert len(generated) == 36  # UUID4

    def test_generated_ids_are_unique(self):
        assert correlation_id_from(None) != correlation_id_from(None)

    @pytest.mark.parametrize(
        "ctx",
        [
            None,
            SimpleNamespace(),
            SimpleNamespace(client_context=SimpleNamespace(custom=None)),
            gateway_context(),
            gateway_context(bedrockAgentCoreAwsRequestId=""),
            gateway_context(bedrockAgentCoreAwsRequestId=123),
        ],
    )
    def test_never_raises_on_an_odd_context(self, ctx):
        assert correlation_id_from(ctx)


class TestCorrelationIdOnResponses:
    def test_an_error_response_carries_the_correlation_id(self):
        """The point: an operator reporting "it failed" can hand over something
        greppable instead of a screenshot."""
        ctx = gateway_context(bedrockAgentCoreAwsRequestId="gw-req-42")

        with patch.object(mcp_dispatcher, "boto3"):
            response = dispatch({"unroutable": "shape"}, ctx)

        assert response["status"] == "error"
        assert response["correlation_id"] == "gw-req-42"

    def test_a_validation_error_carries_it_too(self):
        ctx = gateway_context(
            bedrockAgentCoreAwsRequestId="gw-req-7",
            bedrockAgentCoreToolName="target___list_spice_datasets",
        )

        with patch.object(mcp_dispatcher, "boto3"):
            response = dispatch({"aws_account_id": "nope"}, ctx)

        assert response["status"] == "error"
        assert response["correlation_id"] == "gw-req-7"

    def test_a_successful_response_does_not_carry_one(self):
        """On success the caller already has what it asked for; the id would be noise."""
        ctx = gateway_context(
            bedrockAgentCoreAwsRequestId="gw-req-9",
            bedrockAgentCoreToolName="target___list_spice_datasets",
        )
        qs = create_autospec(QSClient, instance=True)
        qs.list_datasets.return_value = []

        with patch.object(mcp_dispatcher, "QSClient", return_value=qs):
            response = dispatch({"aws_account_id": "9" * 12}, ctx)

        assert response["status"] == "success"
        assert "correlation_id" not in response

    def test_the_id_appears_in_the_dispatch_log(self, caplog):
        import logging

        ctx = gateway_context(bedrockAgentCoreAwsRequestId="gw-req-log")

        with caplog.at_level(logging.INFO), patch.object(mcp_dispatcher, "boto3"):
            dispatch({"unroutable": "shape"}, ctx)

        ids = [
            getattr(r, "correlation_id", None)
            for r in caplog.records
            if hasattr(r, "correlation_id")
        ]
        assert "gw-req-log" in ids


class TestBoto3VersionGuard:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.43.93", (1, 43, 93)),
            ("1.43.64", (1, 43, 64)),
            ("1.42.38", (1, 42, 38)),
            ("2.0", (2, 0)),
            ("", ()),
            ("garbage", ()),
        ],
    )
    def test_version_parsing(self, raw, expected):
        assert _parse_version(raw) == expected

    def test_an_acceptable_version_is_silent(self):
        with patch("lambda_source.qs_client.boto3") as mock_boto3:
            mock_boto3.__version__ = "1.43.93"
            assert check_boto3_version() == ""

    def test_the_exact_minimum_is_acceptable(self):
        with patch("lambda_source.qs_client.boto3") as mock_boto3:
            mock_boto3.__version__ = ".".join(str(p) for p in MINIMUM_BOTO3_VERSION)
            assert check_boto3_version() == ""

    @pytest.mark.parametrize("version", ["1.42.38", "1.28.0", "1.0.0"])
    def test_an_old_version_warns_and_says_why(self, version):
        """A warning, not a hard failure: the tool still works on legacy relational
        datasets, so refusing to start would be worse than the problem. The failure it
        describes is otherwise completely silent."""
        with patch("lambda_source.qs_client.boto3") as mock_boto3:
            mock_boto3.__version__ = version
            message = check_boto3_version()

        assert version in message
        assert "FileSource" in message
        assert "read as empty" in message

    def test_an_unparseable_version_warns_rather_than_assuming_it_is_fine(self):
        with patch("lambda_source.qs_client.boto3") as mock_boto3:
            mock_boto3.__version__ = "not-a-version"
            assert check_boto3_version() != ""

    def test_the_floor_matches_requirements_txt(self):
        """Three places declared a boto3 floor and setup.py said 1.28.0, so
        `pip install -e .` produced an environment that mis-reads real datasets."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        required = ".".join(str(p) for p in MINIMUM_BOTO3_VERSION)

        assert f"boto3>={required}" in (root / "requirements.txt").read_text()
        assert f'"boto3>={required}"' in (root / "setup.py").read_text()
