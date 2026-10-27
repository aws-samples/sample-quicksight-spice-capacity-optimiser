# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CloudWatch metric publishing, with one namespace shared by every caller.

There were three namespaces in play and only one of them was permitted:

    IAM policy allowed        ${StackName}/SpiceOptimiser
    handler.emit_metric used  SPICEDatasetOptimiser
    the dispatcher used       SPICEOptimiser

So every metric the optimiser emitted was denied. It was invisible because both call
sites swallow publishing failures — correctly, since telemetry must never change what a
caller is told, but that also means a misconfiguration produces no signal anywhere except
a warning line nobody reads.

The mismatch was latent rather than new. The namespace condition was added to
`template.yaml` in `15b3d39` on 2026-08-04, but the stack was last updated 2026-08-03 and
then not again until 2026-09-14 — so the deployed role had no condition and metrics
flowed. Deploying the template six weeks later activated the restriction and silently
turned them off. A latent template defect and a deploy are what it took to surface it.

One module now owns the namespace so the constant cannot drift from the IAM policy again,
and a test asserts the two agree.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)

# Must match the `cloudwatch:namespace` condition on CloudWatchMetricsPolicy in
# deployment/template.yaml. tests/unit/test_metrics.py asserts they agree.
#
# Kept as the historical name rather than switching to a stack-scoped one, so the
# existing metrics and any dashboards built on them stay continuous.
METRIC_NAMESPACE = "SPICEDatasetOptimiser"


def emit_metric(
    metric_name: str,
    value: float,
    unit: str = "Count",
    dimensions: dict[str, str] | None = None,
    client: Any | None = None,
) -> bool:
    """Publish one metric. Never raises.

    Args:
        metric_name: Metric name.
        value: Metric value.
        unit: CloudWatch unit.
        dimensions: Optional dimensions. Omit for anything an alarm watches — an alarm
            with no dimensions does not match a datapoint published with them, which is
            a quiet way to build an alarm that can never fire.
        client: Optional pre-built CloudWatch client, for tests and for reuse.

    Returns:
        True when the publish succeeded. Returned rather than raised so a caller can
        assert on it in a test without telemetry ever being able to fail a real request.
    """
    datum: dict[str, Any] = {
        "MetricName": metric_name,
        "Value": value,
        "Unit": unit,
    }
    if dimensions:
        datum["Dimensions"] = [
            {"Name": k, "Value": str(v)[:255]} for k, v in dimensions.items()
        ]

    try:
        cloudwatch = client or boto3.client("cloudwatch")
        cloudwatch.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=[datum])
        return True
    except Exception as exc:  # noqa: BLE001 - telemetry is never load-bearing
        logger.warning(
            "Failed to emit CloudWatch metric",
            extra={
                "metric_name": metric_name,
                "namespace": METRIC_NAMESPACE,
                "error": str(exc),
                "operation": "emit_metric",
            },
        )
        return False
