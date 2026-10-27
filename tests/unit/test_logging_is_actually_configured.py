# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""``_configure_logging`` must work when a handler already exists.

That is the only case that occurs in production, and it was the one case the function
did not handle. It was guarded by ``if not root_logger.handlers:``, and the Python Lambda
runtime installs a root handler before any of our code runs — so in Lambda, and only in
Lambda, the entire body was skipped.

Nothing failed. Logs kept appearing, in the runtime's default format, and every
``extra={...}`` payload was dropped on the way out because the default formatter ignores
it. The fields were there, populated, and discarded. Concretely, during one debugging
session:

- ``Unhandled exception before or during routing`` arrived with no traceback
- ``Cannot release dataset lock`` arrived without the ``error`` field attached
  specifically to explain why, so the cause had to be established by IAM simulation
- ``Dataset lock acquired`` / ``released`` never appeared at all, being INFO

Twenty-eight tests in ``test_observability.py`` covered the correlation id and the
formatter, passed throughout, and none of them called ``_configure_logging``. So the
suite was green around a function that did nothing — the same shape as a mock accepting a
call signature that production rejects. A formatter that is never installed formats
nothing.

These assertions exercise installation, not formatting. The distinction is the bug.
"""

from __future__ import annotations

import json
import logging

import pytest

from lambda_source.handler import _configure_logging, _JsonFormatter


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Leave the root logger exactly as found.

    This module is unusual in mutating global logging state, so it cleans up rather than
    leaking a JSON formatter into every test that runs after it — pytest's caplog and
    other suites depend on the root logger.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_formatters = [h.formatter for h in saved_handlers]
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for handler, formatter in zip(saved_handlers, saved_formatters):
            handler.setFormatter(formatter)


def _lambda_like_root(root: logging.Logger) -> logging.Handler:
    """Reproduce what the Lambda runtime leaves behind: one pre-installed handler."""
    root.handlers[:] = []
    preinstalled = logging.StreamHandler()
    preinstalled.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    root.addHandler(preinstalled)
    root.setLevel(logging.WARNING)
    return preinstalled


class TestHandlerAlreadyExists:
    """The production case, and the one the old guard skipped entirely."""

    def test_the_existing_handler_gets_the_json_formatter(self, _restore_root_logger):
        root = _restore_root_logger
        preinstalled = _lambda_like_root(root)

        _configure_logging()

        assert isinstance(preinstalled.formatter, _JsonFormatter), (
            "the pre-installed Lambda handler kept its default formatter, so every "
            "extra={...} payload in the codebase is still being discarded"
        )

    def test_the_level_is_raised_so_info_is_not_dropped(self, _restore_root_logger):
        root = _restore_root_logger
        _lambda_like_root(root)
        assert root.level == logging.WARNING

        _configure_logging()

        assert root.level == logging.INFO

    def test_no_second_handler_is_added(self, _restore_root_logger):
        """Adding one would duplicate every line in CloudWatch."""
        root = _restore_root_logger
        _lambda_like_root(root)

        _configure_logging()

        assert len(root.handlers) == 1

    def test_calling_it_twice_is_harmless(self, _restore_root_logger):
        """It runs per invocation, and a warm container keeps the previous state."""
        root = _restore_root_logger
        _lambda_like_root(root)

        _configure_logging()
        _configure_logging()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, _JsonFormatter)


class TestNoHandlerExists:
    """Local runs and tests, where the old guard did work. Must keep working."""

    def test_a_handler_is_installed_with_the_json_formatter(self, _restore_root_logger):
        root = _restore_root_logger
        root.handlers[:] = []

        _configure_logging()

        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, _JsonFormatter)
        assert root.level == logging.INFO


class TestLevelOverride:
    def test_an_explicitly_configured_level_wins(self, _restore_root_logger, monkeypatch):
        """Lambda advanced logging controls set this. Overriding it would be rude."""
        root = _restore_root_logger
        _lambda_like_root(root)
        monkeypatch.setenv("AWS_LAMBDA_LOG_LEVEL", "DEBUG")

        _configure_logging()

        assert root.level == logging.DEBUG

    def test_a_nonsense_level_falls_back_to_info_rather_than_silencing_logs(
        self, _restore_root_logger, monkeypatch
    ):
        """``getLevelName`` returns the string "Level BANANA" for junk input.

        Passed to ``setLevel`` that raises, and a typo in an env var must not be able to
        take the logging out — which is the failure this whole module is about.
        """
        root = _restore_root_logger
        _lambda_like_root(root)
        monkeypatch.setenv("AWS_LAMBDA_LOG_LEVEL", "BANANA")

        _configure_logging()

        assert root.level == logging.INFO


class TestTheDetailActuallyReachesTheOutput:
    """End to end through the installed formatter, on the two lines that failed.

    Asserting that a formatter is attached is necessary but not sufficient; what matters
    is that the diagnostic detail survives the trip.
    """

    def _emit(self, root: logging.Logger, capsys, **kwargs) -> dict:
        root.handlers[:] = []
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(stream_handler)

        _configure_logging()
        logging.getLogger("lambda_source.test").warning("Cannot release dataset lock", **kwargs)

        captured = capsys.readouterr()
        return json.loads((captured.err or captured.out).strip().splitlines()[-1])

    def test_extra_fields_survive(self, _restore_root_logger, capsys):
        entry = self._emit(
            _restore_root_logger,
            capsys,
            extra={"dataset_id": "ds-1", "error": "AccessDenied on DeleteObject"},
        )

        assert entry["details"]["dataset_id"] == "ds-1"
        assert entry["details"]["error"] == "AccessDenied on DeleteObject"

    def test_a_traceback_survives(self, _restore_root_logger, capsys):
        """The missing piece behind "Unhandled exception before or during routing"."""
        try:
            raise TypeError("create_ingestion() missing 1 required positional argument")
        except TypeError:
            entry = self._emit(_restore_root_logger, capsys, exc_info=True)

        assert "TypeError" in entry["exception"]
        assert "create_ingestion()" in entry["exception"]

    def test_an_info_line_is_emitted_at_all(self, _restore_root_logger, capsys):
        """"Dataset lock acquired" and "released" are INFO, and never appeared."""
        root = _restore_root_logger
        root.handlers[:] = []
        root.addHandler(logging.StreamHandler())
        root.setLevel(logging.WARNING)

        _configure_logging()
        logging.getLogger("lambda_source.test").info(
            "Dataset lock released", extra={"dataset_id": "ds-1"}
        )

        captured = capsys.readouterr()
        output = (captured.err or captured.out).strip()
        assert output, "an INFO line was still dropped"
        assert json.loads(output.splitlines()[-1])["message"] == "Dataset lock released"


class TestThirdPartyLoggersStayQuiet:
    """Raising the root level to INFO also unmutes the AWS SDKs.

    Observed on the first live deploy of this fix: botocore logged "Found credentials in
    environment variables" and attached a ``tenant_id`` attribute to every record.
    CloudWatch Logs bills on ingested volume, so that is money spent to bury the lines
    that were the point of the change.
    """

    @pytest.mark.parametrize("name", ["botocore", "boto3", "urllib3", "s3transfer"])
    def test_the_sdk_loggers_are_held_at_warning(self, _restore_root_logger, name):
        root = _restore_root_logger
        _lambda_like_root(root)
        logging.getLogger(name).setLevel(logging.NOTSET)

        _configure_logging()

        assert logging.getLogger(name).level == logging.WARNING

    def test_our_own_loggers_are_not_muted_by_that(self, _restore_root_logger):
        """The whole point was to make our INFO lines visible. Do not undo it."""
        root = _restore_root_logger
        _lambda_like_root(root)

        _configure_logging()

        assert logging.getLogger("lambda_source.dataset_lock").getEffectiveLevel() == (
            logging.INFO
        )

    def test_sdk_warnings_still_get_through(self, _restore_root_logger, capsys):
        """A throttle or a retry storm is exactly what we would want to see."""
        root = _restore_root_logger
        root.handlers[:] = []
        root.addHandler(logging.StreamHandler())

        _configure_logging()
        logging.getLogger("botocore.retries").warning("throttled, backing off")

        captured = capsys.readouterr()
        output = (captured.err or captured.out).strip()
        assert output, "an SDK warning was suppressed"
        assert json.loads(output.splitlines()[-1])["message"] == "throttled, backing off"


class TestRequestIdSurvivesReplacingTheRuntimeFormatter:
    """We overwrite the formatter that normally surfaces the request id, so carry it."""

    def test_the_request_id_is_promoted_to_the_top_level(self, _restore_root_logger, capsys):
        root = _restore_root_logger
        root.handlers[:] = []
        root.addHandler(logging.StreamHandler())

        _configure_logging()
        logging.getLogger("lambda_source.test").warning(
            "something happened",
            extra={"aws_request_id": "45aa695e-74d0-4970-b0ee-80620545eea3"},
        )

        captured = capsys.readouterr()
        entry = json.loads((captured.err or captured.out).strip().splitlines()[-1])

        assert entry["request_id"] == "45aa695e-74d0-4970-b0ee-80620545eea3"
        assert "aws_request_id" not in entry.get("details", {}), "should not be duplicated"

    def test_runtime_injected_noise_is_not_echoed_into_details(
        self, _restore_root_logger, capsys
    ):
        """``tenant_id`` comes from the Lambda runtime, is almost always None, and is on
        every record. Observed on the first live deploy of this fix. It is billable
        volume carrying no information, and it sits between a reader and the fields that
        do."""
        root = _restore_root_logger
        root.handlers[:] = []
        root.addHandler(logging.StreamHandler())

        _configure_logging()
        logging.getLogger("lambda_source.test").warning(
            "Dataset lock released", extra={"tenant_id": None, "dataset_id": "ds-1"}
        )

        captured = capsys.readouterr()
        entry = json.loads((captured.err or captured.out).strip().splitlines()[-1])

        assert "tenant_id" not in entry.get("details", {})
        assert entry["details"]["dataset_id"] == "ds-1", "must not drop our own fields"

    def test_its_absence_is_not_an_error(self, _restore_root_logger, capsys):
        """Direct invocations and local runs have no request id."""
        root = _restore_root_logger
        root.handlers[:] = []
        root.addHandler(logging.StreamHandler())

        _configure_logging()
        logging.getLogger("lambda_source.test").warning("no request id here")

        captured = capsys.readouterr()
        entry = json.loads((captured.err or captured.out).strip().splitlines()[-1])

        assert "request_id" not in entry
        assert entry["message"] == "no request id here"
