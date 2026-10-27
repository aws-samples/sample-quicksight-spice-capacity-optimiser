"""A dataset refused by the safety threshold must not be reported as a saving.

Found while hunting for a dataset to verify the refresh path on. A scan of three
datasets returned:

    total_columns_identified: 4
    projected_spice_savings_bytes: 5432918
    message: "Identified 4 unused column(s) across 3 dataset(s), which would free
              about 5432918 bytes if removed. Re-run with action='remove' to apply."

One of the three was ``threshold_breached``. It contributed 0 columns — correctly, since
nothing will be removed — but its ``estimated_spice_savings_bytes`` was still added to the
aggregate. So part of that 5432918 belonged to a dataset the tool refuses to touch, and
which no amount of re-running will realise.

A single-dataset scan made it starker still: ``columns_identified_count: 0`` beside
``projected_spice_savings_bytes: 20093055``. Zero columns freeing 19 MB, in one response.

The per-dataset report is not wrong to carry that number — it sits next to
``threshold_breached: true``, which gives it meaning: *this is what you would free if you
raised the threshold*. It is the aggregate, which mentions no threshold, that turns it
into a false promise.

The invitation was worse than the arithmetic. "Re-run with action='remove' to apply"
cannot work: the threshold is a property of the dataset, not of the mode requested, so
apply refuses on exactly the same check.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lambda_source.handler import _build_response
from lambda_source.models import DatasetResult, ExecutionMode, ExecutionResult
from lambda_source.notifications import NotificationPublisher


_counter = itertools.count()


def _dataset(status: str, *, columns: int, savings: int) -> DatasetResult:
    # Unique id per call. Deriving it from the status alone collided whenever a case used
    # two datasets of the same status, and run() then processed one of them twice -
    # which the real aggregation exposed immediately and the reimplemented one hid.
    ident = f"ds-{status}-{next(_counter)}"
    return DatasetResult(
        dataset_id=ident,
        dataset_name=f"{ident}.csv",
        status=status,
        total_columns=19,
        unused_columns_count=columns,
        # A blocked dataset carries no columns_affected: nothing is going to be removed.
        # The savings figure survives because its own report gives it context.
        columns_affected=[f"col_{i}" for i in range(columns)],
        estimated_spice_savings_bytes=savings,
    )


def _result(mode: str, datasets: list[DatasetResult]) -> ExecutionResult:
    """Aggregate by running the real ``Orchestrator.run()`` with process_dataset stubbed.

    The first version of this helper reimplemented the aggregation loop inside the test.
    Every assertion passed — and kept passing when the production branch was deliberately
    broken, because the test was exercising its own copy of the logic rather than the
    code that ships. That is the same mistake this suite has caught twice before: a test
    that mirrors the implementation verifies nothing about it.

    Driving the real method means the blocked branch, the counters and the metric choice
    are all the ones that run in Lambda.
    """
    from lambda_source.handler import Orchestrator

    orchestrator = object.__new__(Orchestrator)
    orchestrator._config = SimpleNamespace(
        execution_mode=mode,
        target_datasets=[d.dataset_id for d in datasets],
        safety_threshold_pct=0.7,
    )
    orchestrator._notifier = MagicMock(spec=NotificationPublisher)
    orchestrator.emit_metric = MagicMock(return_value=True)

    by_id = {d.dataset_id: d for d in datasets}
    orchestrator.process_dataset = lambda dataset_id: by_id[dataset_id]

    return orchestrator.run()


class TestBlockedDatasetsDoNotInflateSavings:
    def test_a_blocked_dataset_contributes_nothing_to_the_total(self):
        """The live case: two scanned plus one blocked."""
        result = _result(
            ExecutionMode.DRY_RUN.value,
            [
                _dataset("scanned", columns=1, savings=1_000),
                _dataset("scanned", columns=3, savings=3_000),
                _dataset("threshold_breached", columns=0, savings=5_000_000),
            ],
        )
        response = _build_response(result, 1.0)

        assert response["total_columns_identified"] == 4
        assert response["projected_spice_savings_bytes"] == 4_000, (
            "the blocked dataset's would-be saving leaked into the aggregate"
        )

    def test_the_blocked_count_is_surfaced(self):
        result = _result(
            ExecutionMode.DRY_RUN.value,
            [
                _dataset("scanned", columns=1, savings=1_000),
                _dataset("threshold_breached", columns=0, savings=5_000_000),
            ],
        )
        response = _build_response(result, 1.0)

        assert response["datasets_blocked"] == 1

    def test_a_clean_run_is_not_cluttered_with_a_zero(self):
        result = _result(
            ExecutionMode.DRY_RUN.value, [_dataset("scanned", columns=2, savings=2_000)]
        )
        response = _build_response(result, 1.0)

        assert "datasets_blocked" not in response

    def test_the_message_counts_only_datasets_it_can_act_on(self):
        result = _result(
            ExecutionMode.DRY_RUN.value,
            [
                _dataset("scanned", columns=4, savings=4_000),
                _dataset("threshold_breached", columns=0, savings=5_000_000),
            ],
        )
        message = _build_response(result, 1.0)["message"]

        assert "across 1 dataset(s)" in message, (
            "counted the blocked dataset among those it found columns in"
        )
        assert "A further 1 dataset(s) were refused" in message


class TestEverythingBlocked:
    """The single-dataset case, where the old message was pure contradiction."""

    def _message(self) -> str:
        result = _result(
            ExecutionMode.DRY_RUN.value,
            [_dataset("threshold_breached", columns=0, savings=20_093_055)],
        )
        return _build_response(result, 1.0)["message"]

    def test_it_does_not_invite_a_rerun_that_will_refuse(self):
        message = self._message()

        assert "would refuse again" in message
        assert "Re-run with action='remove' to apply" not in message, (
            "told the caller to apply, which fails the identical threshold check"
        )

    def test_it_reports_no_phantom_saving(self):
        result = _result(
            ExecutionMode.DRY_RUN.value,
            [_dataset("threshold_breached", columns=0, savings=20_093_055)],
        )
        response = _build_response(result, 1.0)

        assert response["projected_spice_savings_bytes"] == 0
        assert response["total_columns_identified"] == 0

    def test_it_says_what_the_operator_could_actually_do(self):
        """Two real options, and the likelier diagnosis named.

        A dataset whose consumers reference none of its columns is usually a scan that
        could not resolve the references, not a genuinely dead dataset. Saying so costs
        one sentence and is the difference between a dead end and a next step.
        """
        message = self._message()

        assert "safety_threshold_pct" in message
        assert "could not resolve" in message


class TestApplyModeIsUnaffected:
    def test_apply_still_reports_what_it_removed(self):
        result = _result(
            ExecutionMode.APPLY.value, [_dataset("optimized", columns=3, savings=57_786)]
        )
        response = _build_response(result, 1.0)

        assert response["total_columns_removed"] == 3
        assert response["total_spice_freed_bytes"] == 57_786
        assert "message" not in response

    def test_a_blocked_dataset_in_apply_mode_is_also_excluded(self):
        """Apply hits the same threshold, so the same distortion applied there."""
        result = _result(
            ExecutionMode.APPLY.value,
            [
                _dataset("optimized", columns=3, savings=57_786),
                _dataset("threshold_breached", columns=0, savings=5_000_000),
            ],
        )
        response = _build_response(result, 1.0)

        assert response["total_spice_freed_bytes"] == 57_786
        assert response["datasets_blocked"] == 1


class TestTheAgentActuallySeesTheBlockedCount:
    """The job record is what the chat agent reads, and it is a chosen subset.

    _write_job_callback copies named fields rather than forwarding the response, so a
    field added to _build_response and not added there is simply absent. Caught on the
    first live check: the response carried datasets_blocked and the job record returned
    None. The same omission happened with the dry-run vocabulary fields, which is why it
    is now asserted rather than assumed.
    """

    def _stored(self, datasets, monkeypatch):
        from lambda_source import handler

        captured: dict = {}

        class _Table:
            def update_item(self, **kwargs):
                captured.update(kwargs)

        class _Resource:
            def Table(self, name):  # noqa: N802 - boto3's own casing
                return _Table()

        monkeypatch.setattr(handler.boto3, "resource", lambda *a, **k: _Resource())
        response = _build_response(_result(ExecutionMode.DRY_RUN.value, datasets), 1.0)
        handler._write_job_callback("job-1", "jobs", response)
        return captured["ExpressionAttributeValues"][":result"]

    def test_the_blocked_count_reaches_the_job_record(self, monkeypatch):
        stored = self._stored(
            [
                _dataset("scanned", columns=2, savings=2_000),
                _dataset("threshold_breached", columns=0, savings=5_000_000),
            ],
            monkeypatch,
        )

        assert stored["datasets_blocked"] == 1
        assert stored["projected_spice_savings_bytes"] == 2_000

    def test_a_clean_run_does_not_carry_the_key(self, monkeypatch):
        stored = self._stored([_dataset("scanned", columns=2, savings=2_000)], monkeypatch)

        assert "datasets_blocked" not in stored

    def test_the_explanatory_message_reaches_the_job_record_too(self, monkeypatch):
        """It is the sentence the agent reads out when everything was refused."""
        stored = self._stored(
            [_dataset("threshold_breached", columns=0, savings=5_000_000)], monkeypatch
        )

        assert "would refuse again" in stored["message"]


class TestBlockedIsNeitherSuccessNorFailure:
    @pytest.mark.parametrize("mode", [ExecutionMode.DRY_RUN.value, ExecutionMode.APPLY.value])
    def test_a_blocked_dataset_is_not_counted_as_failed(self, mode):
        """Nothing went wrong. The dataset was examined and deliberately left alone."""
        result = _result(
            mode, [_dataset("threshold_breached", columns=0, savings=5_000_000)]
        )
        response = _build_response(result, 1.0)

        assert response["datasets_failed"] == 0
        assert response["status"] == "success"
        assert response["dataset_results"][0]["status"] == "threshold_breached"
