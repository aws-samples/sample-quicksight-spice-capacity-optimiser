# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""A dry run must not describe work it did not do.

`_execute_dry_run` reported ``status="optimized"`` and populated a field named
``columns_removed`` — documented in the model as "apply mode only". That leaked all the
way out, so a read-only scan of a 25-column dataset returned:

    status: "optimized"
    total_columns_removed: 8
    total_spice_freed_bytes: 154634566

None of which happened; the dataset was untouched. The only thing separating that from a
real removal was the job record's separate top-level ``mode`` field, which nobody has
reason to consult when the body already says, in the past tense, that the work is done.

Two ways that costs something. A user who believes the removal already happened never
runs apply and never gets the saving. A user who knows the scan was read-only concludes
the tool modified their data anyway — which is worse, because the next thing they do is
go looking for damage. The chat agent relays this body verbatim, so it inherits whichever
reading it lands on.

These assertions are on the wording of the response rather than on internals, because the
wording is the defect. A caller cannot see ``DatasetResult``.
"""

from __future__ import annotations


import pytest

from lambda_source.handler import _build_response
from lambda_source.models import DatasetResult, ExecutionMode, ExecutionResult

# Vocabulary that asserts a completed mutation. Any of these appearing in a dry-run
# response is the bug, whatever else the response also says.
PAST_TENSE_KEYS = (
    "total_columns_removed",
    "columns_removed_count",
    "total_spice_freed_bytes",
)


def _result(mode: str, *, columns: int = 8, savings: int = 154_634_566) -> ExecutionResult:
    return ExecutionResult(
        execution_id="exec-1",
        timestamp="2026-09-15T12:00:00Z",
        execution_mode=mode,
        datasets_processed=1,
        total_columns_affected=columns,
        total_spice_freed_bytes=savings,
        dataset_results=[
            DatasetResult(
                dataset_id="ds-1",
                dataset_name="sales_orders_SO.json",
                status="scanned" if mode == ExecutionMode.DRY_RUN.value else "optimized",
                total_columns=25,
                unused_columns_count=columns,
                columns_affected=[f"col_{i}" for i in range(columns)],
                estimated_spice_savings_bytes=savings,
                report_s3_key="p/reports/ds-1.json",
            )
        ],
    )


class TestDryRunResponse:
    def test_no_past_tense_key_appears_anywhere_in_a_dry_run_response(self):
        response = _build_response(_result(ExecutionMode.DRY_RUN.value), 1234.0)
        flat = {*response, *response["dataset_results"][0]}

        offenders = sorted(set(PAST_TENSE_KEYS) & flat)
        assert not offenders, (
            f"a dry-run response claims completed work via {offenders}. Nothing was "
            "modified."
        )

    def test_the_counts_are_still_reported_under_honest_names(self):
        """Renaming must not lose the numbers — a scan that reports nothing is useless."""
        response = _build_response(_result(ExecutionMode.DRY_RUN.value), 1234.0)

        assert response["total_columns_identified"] == 8
        assert response["projected_spice_savings_bytes"] == 154_634_566
        assert response["dataset_results"][0]["columns_identified_count"] == 8

    def test_the_status_does_not_say_optimized(self):
        response = _build_response(_result(ExecutionMode.DRY_RUN.value), 1234.0)
        assert response["dataset_results"][0]["status"] == "scanned"

    def test_the_mode_is_in_the_body_not_only_on_the_job_record(self):
        """So a consumer holding just this dict can tell which vocabulary it has."""
        response = _build_response(_result(ExecutionMode.DRY_RUN.value), 1234.0)
        assert response["execution_mode"] == ExecutionMode.DRY_RUN.value

    def test_the_message_states_plainly_that_nothing_changed(self):
        """The agent reads this sentence out. It has to be unambiguous on its own."""
        message = _build_response(_result(ExecutionMode.DRY_RUN.value), 1234.0)["message"]

        assert "no dataset was modified" in message
        assert "would free" in message
        assert "action='remove'" in message, "should say how to actually apply it"

        # Asserted against specific claim-of-completion phrasings rather than banning the
        # word "removed" outright. The conditional "would free ... if removed" is both
        # accurate and the clearest way to say it, so a blanket word ban would have forced
        # the message to be worse in order to pass. A regex is not a judge of tense.
        for claim in ("columns removed", "optimisation complete", "spice freed"):
            assert claim not in message.lower(), (
                f"dry-run message asserts completed work via {claim!r}: {message!r}"
            )


class TestApplyResponse:
    """The corrected dry run must not have flattened the distinction the other way.

    Apply mode really did remove columns and free capacity, and has to keep saying so —
    otherwise the fix trades one inaccurate report for another.
    """

    def test_apply_still_reports_removal_in_the_past_tense(self):
        response = _build_response(_result(ExecutionMode.APPLY.value), 1234.0)

        assert response["total_columns_removed"] == 8
        assert response["total_spice_freed_bytes"] == 154_634_566
        assert response["dataset_results"][0]["columns_removed_count"] == 8
        assert response["dataset_results"][0]["status"] == "optimized"

    def test_apply_carries_no_dry_run_hedging(self):
        response = _build_response(_result(ExecutionMode.APPLY.value), 1234.0)

        assert "projected_spice_savings_bytes" not in response
        assert "total_columns_identified" not in response
        assert "message" not in response, (
            "the dry-run explanatory message must not appear on a real removal"
        )


class TestJobCallback:
    """What the chat agent actually reads, via the DynamoDB job record.

    The callback used to ask the response for apply-mode keys unconditionally. Left
    unfixed alongside the rest, it would have defaulted them to 0 and reported a scan
    that found 8 columns as having found none — a quieter version of the same lie.
    """

    @pytest.mark.parametrize(
        "mode,expected_key,absent_key",
        [
            (
                ExecutionMode.DRY_RUN.value,
                "total_columns_identified",
                "total_columns_removed",
            ),
            (
                ExecutionMode.APPLY.value,
                "total_columns_removed",
                "total_columns_identified",
            ),
        ],
        ids=["dry_run", "apply"],
    )
    def test_the_callback_copies_the_vocabulary_of_the_run(
        self, mode, expected_key, absent_key, monkeypatch
    ):
        from lambda_source import handler

        captured: dict = {}

        class _Table:
            def update_item(self, **kwargs):
                captured.update(kwargs)

        class _Resource:
            def Table(self, name):  # noqa: N802 - boto3's own casing
                return _Table()

        monkeypatch.setattr(handler.boto3, "resource", lambda *a, **k: _Resource())

        response = _build_response(_result(mode), 1234.0)
        handler._write_job_callback("job-1", "jobs", response)

        stored = captured["ExpressionAttributeValues"][":result"]
        assert stored[expected_key] == 8
        assert absent_key not in stored
        assert stored["execution_mode"] == mode
