# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for event routing and the job terminal-state guarantee.

Routing used to be inferred from which keys happened to be present, and the tool
signatures overlap with the pipeline event shape. A pipeline event missing one key was
therefore reinterpreted as a different tool call, its callback was never written, and
the job sat on ``status: "running"`` until its 7-day TTL expired — indistinguishable
from a job that was genuinely still working.

Two guarantees are pinned here:

1. Routing is decided by declaration, not by guessing (``TestRouteEvent``).
2. Any event carrying ``callback_job_id`` reaches a terminal state, on every path
   including the ones it should never have taken (``TestTerminalStateGuarantee``).

The gateway does pass the tool name — on the context as
``bedrockAgentCoreToolName``, formatted ``{target}___{tool}``. The old code asserted
in a docstring that it did not, and built shape inference on that false premise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lambda_source.handler import _is_scheduled_event, handler, route_event
from lambda_source.mcp_dispatcher import KNOWN_TOOLS, tool_name_from_context


def gateway_context(tool_name: str) -> SimpleNamespace:
    """A Lambda context shaped the way AgentCore Gateway delivers it."""
    return SimpleNamespace(
        client_context=SimpleNamespace(
            custom={"bedrockAgentCoreToolName": tool_name}
        )
    )


PIPELINE_EVENT = {
    "aws_account_id": "123456789012",
    "aws_region": "us-east-1",
    "execution_mode": "dry_run",
    "target_datasets": ["ds-1"],
    "s3_bucket": "bucket",
    "s3_prefix": "prefix/",
}

SCHEDULED_EVENT = {
    "version": "0",
    "id": "abc",
    "detail-type": "Scheduled Event",
    "source": "aws.events",
    "account": "123456789012",
    "time": "2026-09-11T00:00:00Z",
    "region": "us-east-1",
    "resources": ["arn:aws:events:us-east-1:123456789012:rule/spice-optimiser-schedule"],
    "detail": {},
}


class TestToolNameFromContext:
    def test_strips_the_target_prefix(self):
        ctx = gateway_context("spice-optimiser-lambda___scan_unused_columns")
        assert tool_name_from_context(ctx) == "scan_unused_columns"

    def test_accepts_an_unprefixed_name(self):
        ctx = gateway_context("scan_unused_columns")
        assert tool_name_from_context(ctx) == "scan_unused_columns"

    def test_every_known_tool_survives_the_round_trip(self):
        for tool in KNOWN_TOOLS:
            ctx = gateway_context(f"spice-optimiser-lambda___{tool}")
            assert tool_name_from_context(ctx) == tool

    @pytest.mark.parametrize(
        "ctx",
        [
            None,
            SimpleNamespace(),
            SimpleNamespace(client_context=None),
            SimpleNamespace(client_context=SimpleNamespace(custom=None)),
            SimpleNamespace(client_context=SimpleNamespace(custom={})),
            gateway_context(""),
        ],
    )
    def test_absent_name_is_a_normal_condition_not_an_error(self, ctx):
        """Direct invocation passes None for context. That means "not a gateway
        call", so it must return empty rather than raise."""
        assert tool_name_from_context(ctx) == ""


class TestRouteEvent:
    def test_explicit_invocation_source_wins(self):
        assert route_event({"invocation_source": "mcp_job"}, None) == "pipeline"
        assert route_event({"invocation_source": "tool"}, None) == "tool"

    def test_gateway_tool_name_routes_to_the_tool_surface(self):
        ctx = gateway_context("spice-optimiser-lambda___list_spice_datasets")
        assert route_event({"aws_account_id": "9" * 12}, ctx) == "tool"

    def test_callback_job_id_alone_routes_to_the_pipeline(self):
        """Only the pipeline is ever invoked with a callback. Routing such an event
        anywhere else is precisely what orphaned jobs for seven days."""
        assert route_event({"callback_job_id": "j", "callback_table": "t"}, None) == (
            "pipeline"
        )

    def test_pipeline_event_routes_to_the_pipeline(self):
        assert route_event(PIPELINE_EVENT, None) == "pipeline"

    def test_pipeline_event_without_s3_bucket_still_routes_to_the_pipeline(self):
        """The regression. s3_bucket used to be part of the routing decision, so an
        otherwise valid pipeline event missing it was silently reinterpreted as a
        tool call. Config validation rejects a missing bucket with a legible error;
        routing has no business deciding it."""
        event = {k: v for k, v in PIPELINE_EVENT.items() if k != "s3_bucket"}
        assert route_event(event, None) == "pipeline"

    def test_scheduled_event_routes_to_the_pipeline(self):
        """It used to fall through to the dispatcher and return "cannot identify
        tool" — a 3ms no-op every 7 days that nothing surfaced."""
        assert route_event(SCHEDULED_EVENT, None) == "pipeline"

    @pytest.mark.parametrize(
        "event",
        [
            {"job_id": "j"},
            {"dataset_ids": ["a"], "aws_account_id": "9" * 12},
            {"aws_account_id": "9" * 12},
        ],
    )
    def test_tool_shaped_events_route_to_the_tool_surface(self, event):
        assert route_event(event, None) == "tool"

    def test_empty_event_routes_to_the_tool_surface(self):
        assert route_event({}, None) == "tool"

    def test_gateway_context_does_not_hijack_a_pipeline_callback(self):
        """A self-invoked pipeline job passes invocation_source, which must outrank a
        stale tool name on a reused execution context."""
        ctx = gateway_context("spice-optimiser-lambda___scan_unused_columns")
        event = dict(PIPELINE_EVENT, invocation_source="mcp_job", callback_job_id="j")
        assert route_event(event, ctx) == "pipeline"


class TestScheduledEventDetection:
    @pytest.mark.parametrize(
        "event,expected",
        [
            (SCHEDULED_EVENT, True),
            ({"source": "aws.events"}, True),
            ({"detail-type": "Scheduled Event"}, True),
            ({}, False),
            (PIPELINE_EVENT, False),
            ({"source": "aws.s3"}, False),
        ],
    )
    def test_detection(self, event, expected):
        assert _is_scheduled_event(event) is expected

    @patch("lambda_source.handler._run_pipeline", return_value={"status": "success"})
    def test_schedule_supplies_its_own_config_location(self, mock_run):
        """The EventBridge rule is created with no Input, so the function has to
        supply the config location from its own environment or the schedule can
        never work."""
        env = {"S3_BUCKET": "cfg-bucket", "CONFIG_S3_KEY": "prefix/config.json"}
        with patch.dict("os.environ", env, clear=False):
            handler(SCHEDULED_EVENT, None)

        forwarded = mock_run.call_args[0][0]
        assert forwarded["config_s3_bucket"] == "cfg-bucket"
        assert forwarded["config_s3_key"] == "prefix/config.json"
        assert forwarded["invocation_source"] == "schedule"


class TestTerminalStateGuarantee:
    """A job must always reach a terminal state. Never left on "running"."""

    @patch("lambda_source.handler._write_job_callback")
    @patch("lambda_source.handler.load_config", side_effect=RuntimeError("boom"))
    def test_exception_before_config_still_writes_a_callback(
        self, mock_load, mock_callback
    ):
        result = handler({"callback_job_id": "j", "callback_table": "t"}, None)

        assert result["status"] == "error"
        mock_callback.assert_called_once()

    @patch("lambda_source.handler._write_job_callback")
    def test_tool_route_carrying_a_callback_still_writes_one(self, mock_callback):
        """A tool call should not carry a callback. If one arrives anyway the job
        still has to be closed out rather than left running."""
        with patch(
            "lambda_source.mcp_dispatcher.dispatch",
            return_value={"status": "success", "datasets": []},
        ):
            result = handler(
                {
                    "job_id": "some-job",
                    "callback_job_id": "j",
                    "callback_table": "t",
                },
                gateway_context("spice-optimiser-lambda___get_job_result"),
            )

        assert result["status"] == "success"
        mock_callback.assert_called_once()

    @patch("lambda_source.handler._write_job_callback")
    def test_no_callback_fields_means_no_callback(self, mock_callback):
        with patch(
            "lambda_source.mcp_dispatcher.dispatch", return_value={"status": "success"}
        ):
            handler({"job_id": "j"}, None)

        mock_callback.assert_not_called()

    def test_every_response_carries_a_status(self):
        """The old unknown-tool path returned a bare {"error": ...} with no status,
        so callers doing result["status"] raised KeyError instead of seeing a
        failure."""
        with patch("lambda_source.mcp_dispatcher.boto3"):
            response = handler({"unrecognised": "shape"}, None)

        assert "status" in response
        assert response["status"] == "error"

    def test_unknown_tool_names_the_known_tools(self):
        with patch("lambda_source.mcp_dispatcher.boto3"):
            response = handler({"unrecognised": "shape"}, None)

        assert sorted(KNOWN_TOOLS) == response["known_tools"]

    @patch("lambda_source.handler._write_job_callback")
    def test_none_event_does_not_crash(self, mock_callback):
        with patch("lambda_source.mcp_dispatcher.boto3"):
            response = handler(None, None)

        assert response["status"] == "error"
