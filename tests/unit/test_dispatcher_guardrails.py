# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the tool surface's guardrail passthrough and stale-job detection.

Two defects covered here.

``safety_threshold_pct`` and ``quarantine_period_days`` were advertised on the tool
schema and never read by ``_start_job``, so the pipeline fell back to its own defaults.
Because the pipeline gates quarantine on ``quarantine_period_days is not None``,
quarantine never ran at all through the agent path. Both guardrails were documented,
offered to the agent, and inert.

Separately, a job whose worker died never reached a terminal state. It stayed
``running`` until the 7-day TTL, and a caller could not tell it from a live one. A
Lambda cannot outlive its own timeout, so past that point the job is provably dead.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from lambda_source import mcp_dispatcher
from lambda_source.mcp_dispatcher import (
    STALE_JOB_AFTER_SECONDS,
    _get_job_result,
    _is_stale,
    _parse_guardrails,
    _start_job,
)


class TestGuardrailParsing:
    def test_absent_guardrails_yield_an_empty_fragment(self):
        fragment, error = _parse_guardrails({})
        assert fragment == {}
        assert error == ""

    def test_valid_values_pass_through(self):
        fragment, error = _parse_guardrails(
            {"safety_threshold_pct": 0.5, "quarantine_period_days": 14}
        )
        assert fragment == {"safety_threshold_pct": 0.5, "quarantine_period_days": 14}
        assert error == ""

    @pytest.mark.parametrize("value", [0.0, 1.0, 0.35])
    def test_threshold_boundaries_are_accepted(self, value):
        fragment, error = _parse_guardrails({"safety_threshold_pct": value})
        assert error == ""
        assert fragment["safety_threshold_pct"] == value

    @pytest.mark.parametrize("value", [-0.1, 1.1, 42])
    def test_out_of_range_threshold_is_rejected(self, value):
        """Silently clamping or defaulting when an operator asked for a stricter
        threshold is the wrong failure mode for a safety control."""
        fragment, error = _parse_guardrails({"safety_threshold_pct": value})
        assert fragment == {}
        assert "between 0.0 and 1.0" in error

    def test_non_numeric_threshold_is_rejected(self):
        fragment, error = _parse_guardrails({"safety_threshold_pct": "loose"})
        assert fragment == {}
        assert "must be a number" in error

    def test_negative_quarantine_is_rejected(self):
        fragment, error = _parse_guardrails({"quarantine_period_days": -1})
        assert fragment == {}
        assert "non-negative" in error

    def test_non_integer_quarantine_is_rejected(self):
        fragment, error = _parse_guardrails({"quarantine_period_days": "soon"})
        assert fragment == {}
        assert "must be an integer" in error

    def test_zero_quarantine_is_valid_and_distinct_from_absent(self):
        """Zero days means "no waiting period", which is a real choice. Absent means
        "no quarantine gate at all". The pipeline treats them differently."""
        fragment, _ = _parse_guardrails({"quarantine_period_days": 0})
        assert fragment == {"quarantine_period_days": 0}

    @pytest.mark.parametrize(
        "event",
        [
            {"safety_threshold_pct": None},
            {"quarantine_period_days": None},
        ],
    )
    def test_explicit_null_is_treated_as_absent(self, event):
        fragment, error = _parse_guardrails(event)
        assert fragment == {}
        assert error == ""


class TestGuardrailsReachThePipeline:
    def _start(self, event: dict):
        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job("dry_run", event)
        return response, mock_lambda

    def test_values_appear_in_the_async_payload(self):
        """The assertion the deleted mcp_server tests used to make, restored against
        the layer that actually ships.

        quarantine_period_days requires a configured table, so this patches one in.
        Without it the call is now refused rather than silently stripped — see
        TestQuarantineAvailability.
        """
        import json

        with patch.object(mcp_dispatcher, "QUARANTINE_TABLE", "quarantine-table"):
            _, mock_lambda = self._start(
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "safety_threshold_pct": 0.42,
                    "quarantine_period_days": 21,
                }
            )

        payload = json.loads(mock_lambda.invoke.call_args[1]["Payload"].decode())
        assert payload["safety_threshold_pct"] == 0.42
        assert payload["quarantine_period_days"] == 21
        assert payload["dynamodb_table"] == "quarantine-table"

    def test_payload_declares_its_route(self):
        """So the pipeline branch is chosen by declaration rather than inferred from
        which keys happen to be present."""
        import json

        _, mock_lambda = self._start(
            {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12}
        )

        payload = json.loads(mock_lambda.invoke.call_args[1]["Payload"].decode())
        assert payload["invocation_source"] == "mcp_job"

    def test_effective_guardrails_are_echoed_back(self):
        """An operator should be able to confirm which guardrails were in force, not
        assume the values they passed were honoured."""
        response, _ = self._start(
            {
                "dataset_ids": ["ds-1"],
                "aws_account_id": "9" * 12,
                "safety_threshold_pct": 0.3,
            }
        )

        assert response["effective_guardrails"]["safety_threshold_pct"] == 0.3
        assert response["effective_guardrails"]["quarantine_period_days"] is None

    def test_invalid_guardrail_rejects_before_creating_a_job(self):
        """A rejected call must not leave a job record behind."""
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            response = _start_job(
                "dry_run",
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "safety_threshold_pct": 5.0,
                },
            )

        assert response["status"] == "error"
        mock_boto3.resource.assert_not_called()

    def test_missing_required_fields_are_reported_with_a_status(self):
        response = _start_job("dry_run", {})
        assert response["status"] == "error"
        assert "dataset_ids" in response["error"]


class TestDestructiveConfirmation:
    """apply mode requires an explicit action="remove".

    This check used to be implicit: routing was by payload shape and
    ``action == "remove"`` was the only thing separating a remove from a scan, so
    omitting it silently downgraded the call to a dry run. Routing by tool name
    removed that side effect, which would have left the confirmation unenforced —
    the schema declares it required, but a safeguard on a destructive operation
    should not depend on the gateway choosing to validate it.
    """

    def test_apply_without_confirmation_is_refused(self):
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            response = _start_job(
                "apply", {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12}
            )

        assert response["status"] == "error"
        assert 'action="remove"' in response["error"]
        # And nothing was created.
        mock_boto3.resource.assert_not_called()

    @pytest.mark.parametrize("value", ["Remove", "REMOVE", "delete", "", None, True])
    def test_only_the_exact_literal_counts_as_confirmation(self, value):
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            response = _start_job(
                "apply",
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "action": value,
                },
            )

        assert response["status"] == "error"
        mock_boto3.resource.assert_not_called()

    def test_apply_with_confirmation_proceeds(self):
        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job(
                "apply",
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "action": "remove",
                },
            )

        assert response["status"] == "running"
        assert "job_id" in response

    def test_dry_run_never_needs_confirmation(self):
        """Read-only analysis must not be gated behind a destructive-intent flag."""
        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job(
                "dry_run", {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12}
            )

        assert response["status"] == "running"

    def test_confirmation_is_checked_on_the_gateway_route_too(self):
        """The path that actually matters: a named tool call, not shape inference."""
        from types import SimpleNamespace

        from lambda_source.handler import handler

        ctx = SimpleNamespace(
            client_context=SimpleNamespace(
                custom={
                    "bedrockAgentCoreToolName": (
                        "spice-optimiser-lambda___remove_unused_columns"
                    )
                }
            )
        )
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            response = handler(
                {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12}, ctx
            )

        assert response["status"] == "error"
        assert 'action="remove"' in response["error"]
        mock_boto3.resource.assert_not_called()


class TestStaleJobDetection:
    def test_recent_running_job_is_not_stale(self):
        assert _is_stale({"created_at": int(time.time())}) is False

    def test_job_older_than_any_possible_worker_is_stale(self):
        created = int(time.time()) - STALE_JOB_AFTER_SECONDS - 1
        assert _is_stale({"created_at": created}) is True

    @pytest.mark.parametrize("value", [None, "", "not-a-number", {}])
    def test_unreadable_created_at_is_not_treated_as_stale(self, value):
        """Refusing to declare a job dead on the basis of data we cannot read is the
        safer default."""
        assert _is_stale({"created_at": value}) is False

    def _get(self, item: dict) -> dict:
        table = MagicMock()
        table.get_item.return_value = {"Item": item}
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = table
            return _get_job_result({"job_id": "j"})

    def test_live_job_reports_running_and_asks_for_another_poll(self):
        result = self._get(
            {"job_id": "j", "status": "running", "created_at": int(time.time())}
        )
        assert result["status"] == "running"
        assert "Poll again" in result["message"]

    def test_dead_job_reports_failed_instead_of_running_forever(self):
        """Previously this returned "Still running. Poll again in 10 seconds." for
        seven days, so the agent polled a corpse."""
        result = self._get(
            {
                "job_id": "j",
                "status": "running",
                "created_at": int(time.time()) - STALE_JOB_AFTER_SECONDS - 60,
            }
        )
        assert result["status"] == "failed"
        assert result["inferred_from_staleness"] is True
        assert "timed out" in result["error_detail"]

    def test_completed_job_is_untouched_by_the_staleness_check(self):
        result = self._get(
            {
                "job_id": "j",
                "status": "completed",
                "created_at": 0,
                "result": {"datasets_processed": 1},
            }
        )
        assert result["status"] == "completed"
        assert "inferred_from_staleness" not in result

    def test_missing_job_reports_an_error_with_a_status(self):
        table = MagicMock()
        table.get_item.return_value = {}
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = table
            result = _get_job_result({"job_id": "absent"})

        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_missing_job_id_reports_an_error_with_a_status(self):
        assert _get_job_result({})["status"] == "error"


class TestQuarantineAvailability:
    """Quarantine is optional, and asking for it when it is absent is refused.

    The feature was documented in the README at 14 days, defaulted to None in code, and
    disabled on the deployed stack — three different answers. Worse, requesting it was
    accepted and then ignored: `_start_job` never forwarded the value, and even once it
    did, `_parse_config` read `dynamodb_table` only from the payload while
    CloudFormation set it in the environment. So `QuarantineManager` was disabled, the
    handler's compound gate fell through, and columns were removed immediately by a run
    that had explicitly asked for a waiting period.

    Optional is fine. Silently optional is not.
    """

    def _start(self, event: dict, table: str):
        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "QUARANTINE_TABLE", table), patch.object(
            mcp_dispatcher, "boto3"
        ) as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job("dry_run", event)
        return response, mock_lambda, mock_boto3

    def test_requesting_quarantine_without_a_table_is_refused(self):
        response, _, mock_boto3 = self._start(
            {
                "dataset_ids": ["ds-1"],
                "aws_account_id": "9" * 12,
                "quarantine_period_days": 14,
            },
            table="",
        )

        assert response["status"] == "error"
        assert "quarantine is not enabled" in response["error"]
        # Refused before any job record exists.
        mock_boto3.resource.assert_not_called()

    def test_the_refusal_explains_both_ways_out(self):
        response, _, _ = self._start(
            {
                "dataset_ids": ["ds-1"],
                "aws_account_id": "9" * 12,
                "quarantine_period_days": 14,
            },
            table="",
        )

        assert "QuarantineEnabled=true" in response["error"]
        assert "omit quarantine_period_days" in response["error"]

    def test_omitting_quarantine_works_without_a_table(self):
        """The supported way to run without a waiting period. Quarantine stays optional
        and this is the default path."""
        response, _, _ = self._start(
            {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12}, table=""
        )

        assert response["status"] == "running"

    def test_requesting_quarantine_with_a_table_proceeds(self):
        response, _, _ = self._start(
            {
                "dataset_ids": ["ds-1"],
                "aws_account_id": "9" * 12,
                "quarantine_period_days": 14,
            },
            table="quarantine-table",
        )

        assert response["status"] == "running"
        assert response["effective_guardrails"]["quarantine_period_days"] == 14

    def test_zero_days_still_requires_a_table(self):
        """Zero means "no waiting period but do track", which still needs somewhere to
        record state. Distinct from omitting the parameter entirely."""
        response, _, _ = self._start(
            {
                "dataset_ids": ["ds-1"],
                "aws_account_id": "9" * 12,
                "quarantine_period_days": 0,
            },
            table="",
        )

        assert response["status"] == "error"

    def test_table_is_forwarded_so_the_run_is_self_describing(self):
        import json

        _, mock_lambda, _ = self._start(
            {"dataset_ids": ["ds-1"], "aws_account_id": "9" * 12},
            table="quarantine-table",
        )

        payload = json.loads(mock_lambda.invoke.call_args[1]["Payload"].decode())
        assert payload["dynamodb_table"] == "quarantine-table"


class TestStaleVerdictPersistence:
    """The inferred terminal state is written back, conditionally.

    Previously the verdict was read-only: `get_job_result` reported `failed` while the
    record still said `running`, so anything reading DynamoDB directly saw the wrong
    answer and the inference was recomputed on every poll.
    """

    def _stale_item(self) -> dict:
        return {
            "job_id": "j",
            "status": "running",
            "created_at": int(time.time()) - STALE_JOB_AFTER_SECONDS - 60,
        }

    def _get(self, item: dict, update_side_effect=None):
        table = MagicMock()
        table.get_item.return_value = {"Item": item}
        if update_side_effect:
            table.update_item.side_effect = update_side_effect
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = table
            result = _get_job_result({"job_id": "j"})
        return result, table

    def test_a_stale_job_is_written_back_as_failed(self):
        _, table = self._get(self._stale_item())

        table.update_item.assert_called_once()
        values = table.update_item.call_args[1]["ExpressionAttributeValues"]
        assert values[":failed"] == "failed"
        assert values[":inferred"] is True

    def test_the_write_is_conditional_on_still_running(self):
        """So two concurrent polls cannot both write, and a real callback landing at the
        same moment wins rather than being overwritten by an inference."""
        _, table = self._get(self._stale_item())

        kwargs = table.update_item.call_args[1]
        assert kwargs["ConditionExpression"] == "#s = :running"
        assert kwargs["ExpressionAttributeValues"][":running"] == "running"

    def test_the_inference_is_recorded_as_an_inference(self):
        """"The worker reported failure" and "we concluded it failed" are different
        facts and a later reader should be able to tell them apart."""
        _, table = self._get(self._stale_item())

        expr = table.update_item.call_args[1]["UpdateExpression"]
        assert "inferred_from_staleness" in expr

    def test_a_live_job_is_never_written_to(self):
        _, table = self._get(
            {"job_id": "j", "status": "running", "created_at": int(time.time())}
        )

        table.update_item.assert_not_called()

    def test_a_completed_job_is_never_written_to(self):
        _, table = self._get(
            {"job_id": "j", "status": "completed", "created_at": 0, "result": {}}
        )

        table.update_item.assert_not_called()

    def test_a_conditional_check_failure_is_not_fatal(self):
        """Means something else reached a terminal state first, which is the outcome we
        wanted anyway."""
        exc = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem"
        )
        result, _ = self._get(self._stale_item(), update_side_effect=exc)

        assert result["status"] == "failed"

    def test_a_throttled_write_still_returns_the_verdict(self):
        """Bookkeeping must not fail a status poll."""
        exc = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException"}}, "UpdateItem"
        )
        result, _ = self._get(self._stale_item(), update_side_effect=exc)

        assert result["status"] == "failed"
        assert result["inferred_from_staleness"] is True

    def test_an_unexpected_write_error_still_returns_the_verdict(self):
        result, _ = self._get(
            self._stale_item(), update_side_effect=RuntimeError("boom")
        )

        assert result["status"] == "failed"


class TestCascadeChildRefreshGuardrail:
    """cascade_child_refresh is an opt-in on remove_unused_columns.

    It fans CreateIngestion out across the whole dependency tree after the parent is
    optimised, so it is off unless the caller sets it. Parsed and forwarded the same way
    the other guardrails are, so a job record shows whether a cascade was requested.
    """

    def test_absent_cascade_is_not_in_the_fragment(self):
        fragment, error = _parse_guardrails({"dataset_ids": ["ds-1"]})
        assert "cascade_child_refresh" not in fragment
        assert error == ""

    @pytest.mark.parametrize(
        "value,expected",
        [(True, True), (False, False), ("true", True), ("false", False)],
    )
    def test_boolean_and_string_boolean_are_accepted(self, value, expected):
        fragment, error = _parse_guardrails({"cascade_child_refresh": value})
        assert error == ""
        assert fragment["cascade_child_refresh"] is expected

    @pytest.mark.parametrize("value", ["yes", "1", 1, 0, "maybe"])
    def test_non_boolean_is_rejected(self, value):
        fragment, error = _parse_guardrails({"cascade_child_refresh": value})
        assert fragment == {}
        assert "must be a boolean" in error

    def test_explicit_null_is_treated_as_absent(self):
        fragment, error = _parse_guardrails({"cascade_child_refresh": None})
        assert fragment == {}
        assert error == ""

    def test_it_reaches_the_async_payload_on_a_remove(self):
        import json

        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job(
                "apply",
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "action": "remove",
                    "cascade_child_refresh": True,
                },
            )

        assert response["status"] == "running"
        payload = json.loads(mock_lambda.invoke.call_args[1]["Payload"].decode())
        assert payload["cascade_child_refresh"] is True
        assert response["effective_guardrails"]["cascade_child_refresh"] is True

    def test_it_defaults_to_false_in_the_echoed_guardrails(self):
        mock_lambda = MagicMock()
        with patch.object(mcp_dispatcher, "boto3") as mock_boto3:
            mock_boto3.resource.return_value.Table.return_value = MagicMock()
            mock_boto3.client.return_value = mock_lambda
            response = _start_job(
                "apply",
                {
                    "dataset_ids": ["ds-1"],
                    "aws_account_id": "9" * 12,
                    "action": "remove",
                },
            )

        assert response["effective_guardrails"]["cascade_child_refresh"] is False
