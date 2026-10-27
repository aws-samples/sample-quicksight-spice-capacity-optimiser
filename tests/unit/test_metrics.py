# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for CloudWatch metric publishing.

Every metric the optimiser emitted was being silently denied. Three namespaces were in
play and only one was permitted:

    IAM policy allowed        ${StackName}/SpiceOptimiser
    handler.emit_metric used  SPICEDatasetOptimiser
    the dispatcher used       SPICEOptimiser

Invisible because both call sites swallow publishing failures — correctly, since
telemetry must never change what a caller is told, but that also means a
misconfiguration produces no signal beyond a warning nobody reads.

The mismatch was latent rather than new: the namespace condition was added to
`template.yaml` on 2026-08-04 but the stack was not updated until 2026-09-14, so the
deployed role had no condition and metrics flowed. Deploying six weeks later activated
the restriction and turned them off.

The load-bearing test is
`TestNamespaceAgreement::test_the_iam_condition_matches_the_code`, which is what stops
the two drifting apart again.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lambda_source.metrics import METRIC_NAMESPACE, emit_metric

TEMPLATE = Path(__file__).resolve().parents[2] / "deployment" / "template.yaml"


class TestNamespaceAgreement:
    def test_the_iam_condition_matches_the_code(self):
        """The check that would have caught the original mismatch."""
        template = TEMPLATE.read_text()
        match = re.search(r"cloudwatch:namespace:\s*'([^']+)'", template)

        assert match, "CloudWatchMetricsPolicy has no cloudwatch:namespace condition"
        assert match.group(1) == METRIC_NAMESPACE, (
            f"IAM permits namespace {match.group(1)!r} but the code publishes to "
            f"{METRIC_NAMESPACE!r}. Every metric would be denied."
        )

    def test_the_alarm_watches_the_same_namespace(self):
        """An alarm on a namespace nothing publishes to can never fire."""
        template = TEMPLATE.read_text()
        alarm = template.split("UnrecognisedToolAlarm:")[1].split("ThrottleAlarm:")[0]
        match = re.search(r"Namespace:\s*(\S+)", alarm)

        assert match
        assert match.group(1) == METRIC_NAMESPACE

    def test_only_one_namespace_appears_in_the_source(self):
        """Two hardcoded namespaces in two modules is how this happened."""
        source_dir = Path(__file__).resolve().parents[2] / "lambda_source"
        found = set()
        for path in source_dir.glob("*.py"):
            for m in re.finditer(r'Namespace\s*=\s*["\']([^"\']+)["\']', path.read_text()):
                found.add(m.group(1))

        assert not found - {METRIC_NAMESPACE}, (
            f"namespaces hardcoded outside metrics.py: {found - {METRIC_NAMESPACE}}"
        )


class TestEmitMetric:
    def test_publishes_to_the_shared_namespace(self):
        client = MagicMock()

        assert emit_metric("SomeMetric", 1, "Count", client=client) is True

        kwargs = client.put_metric_data.call_args[1]
        assert kwargs["Namespace"] == METRIC_NAMESPACE
        assert kwargs["MetricData"][0]["MetricName"] == "SomeMetric"
        assert kwargs["MetricData"][0]["Value"] == 1

    def test_omits_dimensions_when_not_given(self):
        """An alarm with no dimensions does not match a datapoint published with them,
        so a dimensionless default is the safe one for anything alarmed."""
        client = MagicMock()
        emit_metric("SomeMetric", 1, client=client)

        assert "Dimensions" not in client.put_metric_data.call_args[1]["MetricData"][0]

    def test_includes_dimensions_when_given(self):
        client = MagicMock()
        emit_metric("SomeMetric", 1, dimensions={"Foo": "bar"}, client=client)

        dims = client.put_metric_data.call_args[1]["MetricData"][0]["Dimensions"]
        assert dims == [{"Name": "Foo", "Value": "bar"}]

    def test_a_failure_is_reported_but_never_raised(self):
        """Telemetry must not be able to fail a real request."""
        client = MagicMock()
        client.put_metric_data.side_effect = RuntimeError("denied")

        assert emit_metric("SomeMetric", 1, client=client) is False

    def test_a_failure_is_logged_with_the_namespace(self, caplog):
        """So the next namespace mismatch names itself instead of being invisible."""
        import logging

        client = MagicMock()
        client.put_metric_data.side_effect = RuntimeError("denied")

        with caplog.at_level(logging.WARNING):
            emit_metric("SomeMetric", 1, client=client)

        namespaces = [
            getattr(r, "namespace", None)
            for r in caplog.records
            if hasattr(r, "namespace")
        ]
        assert METRIC_NAMESPACE in namespaces

    def test_dimension_values_are_truncated(self):
        client = MagicMock()
        emit_metric("M", 1, dimensions={"Long": "x" * 400}, client=client)

        value = client.put_metric_data.call_args[1]["MetricData"][0]["Dimensions"][0][
            "Value"
        ]
        assert len(value) == 255


class TestCallSitesUseTheSharedHelper:
    def test_the_orchestrator_delegates(self):
        """It used to hardcode its own namespace."""
        from unittest.mock import patch

        from lambda_source.handler import Orchestrator
        from lambda_source.models import OptimiserConfig

        cloudwatch = MagicMock()
        with patch("lambda_source.handler.boto3") as mock_boto3, patch(
            "lambda_source.reporter.boto3"
        ), patch("lambda_source.snapshot.boto3"), patch(
            "lambda_source.notifications.boto3"
        ), patch(
            "lambda_source.qs_client.boto3"
        ):
            mock_boto3.client.return_value = cloudwatch
            orch = Orchestrator(
                OptimiserConfig(target_datasets=[], s3_bucket="b", s3_prefix="p/")
            )
            orch.emit_metric("DatasetsProcessed", 3, "Count")

        assert cloudwatch.put_metric_data.call_args[1]["Namespace"] == METRIC_NAMESPACE

    def test_the_unrecognised_dispatch_metric_is_dimensionless(self):
        """It is alarmed on, and the alarm has no dimensions."""
        from unittest.mock import patch

        from lambda_source import mcp_dispatcher

        with patch.object(mcp_dispatcher, "emit_metric") as mock_emit:
            mcp_dispatcher._emit_unrecognised_dispatch_metric("unknown", "payload_shape")

        args, kwargs = mock_emit.call_args
        assert args[0] == "UnrecognisedToolDispatch"
        assert not kwargs.get("dimensions")
