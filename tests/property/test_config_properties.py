# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for configuration parsing round-trip.

Feature: spice-dataset-optimiser
Property 16: Configuration parsing round-trip

Validates: Requirements 11.1
"""

from __future__ import annotations

from hypothesis import given, settings

from lambda_source.config import _parse_config
from lambda_source.models import OptimiserConfig
from tests.property.strategies import st_config


@given(config=st_config())
@settings(max_examples=100)
def test_config_round_trip_preserves_all_fields(config: OptimiserConfig) -> None:
    """Property 16: Configuration parsing round-trip.

    **Validates: Requirements 11.1**

    For any valid configuration conforming to the schema, parsing it into
    the OptimiserConfig dataclass and then serializing back to JSON SHALL
    produce a semantically equivalent document (all field values preserved).
    """
    # Step 1: Serialize the config to a dictionary (JSON representation)
    config_dict = {
        "target_datasets": config.target_datasets,
        "execution_mode": config.execution_mode,
        "safety_threshold_pct": config.safety_threshold_pct,
        "quarantine_period_days": config.quarantine_period_days,
        "s3_bucket": config.s3_bucket,
        "s3_prefix": config.s3_prefix,
        "sns_topic_arn": config.sns_topic_arn,
        "dynamodb_table": config.dynamodb_table,
        "aws_account_id": config.aws_account_id,
        "aws_region": config.aws_region,
    }

    # Step 2: Parse it back using _parse_config
    parsed = _parse_config(config_dict)

    # Step 3: Assert all fields are preserved
    assert parsed.target_datasets == config.target_datasets
    assert parsed.execution_mode == config.execution_mode
    assert parsed.safety_threshold_pct == config.safety_threshold_pct
    assert parsed.quarantine_period_days == config.quarantine_period_days
    assert parsed.s3_bucket == config.s3_bucket
    assert parsed.s3_prefix == config.s3_prefix
    assert parsed.sns_topic_arn == config.sns_topic_arn
    assert parsed.dynamodb_table == config.dynamodb_table
    assert parsed.aws_account_id == config.aws_account_id
    assert parsed.aws_region == config.aws_region


def test_cascade_child_refresh_round_trips() -> None:
    """The opt-in cascade flag survives parsing, defaults off, and is coerced to bool.

    It gates a side-effecting fan-out of CreateIngestion across the dependency tree, so a
    value silently lost in parsing would mean a caller believing children were refreshed
    when they were not — or the reverse.
    """
    base = {"target_datasets": ["ds-1"], "aws_account_id": "9" * 12, "aws_region": "us-east-1"}

    assert _parse_config(base).cascade_child_refresh is False, "defaults off"
    assert _parse_config({**base, "cascade_child_refresh": True}).cascade_child_refresh is True
    assert _parse_config({**base, "cascade_child_refresh": False}).cascade_child_refresh is False
    # Coerced to bool, like skip_unreachable_consumers.
    assert _parse_config({**base, "cascade_child_refresh": 1}).cascade_child_refresh is True
