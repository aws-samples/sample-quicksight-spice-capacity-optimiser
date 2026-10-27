# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for the configuration loader module.

Tests load_config, validate_config, and internal parsing helpers.
Validates Requirements 11.1, 11.2, 11.4, 12.2, 12.3.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lambda_source.config import load_config, validate_config
from lambda_source.models import ExecutionMode, OptimiserConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_inline_event() -> dict:
    """Return a minimal valid inline event payload."""
    return {
        "aws_account_id": "123456789012",
        "aws_region": "us-east-1",
        "execution_mode": "dry_run",
        "target_datasets": {"mode": "explicit", "dataset_ids": ["ds-1", "ds-2"]},
        "safety_threshold_pct": 0.7,
        "quarantine_period_days": 14,
        "s3_bucket": "my-bucket",
        "s3_prefix": "spice-optimiser/",
        "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:alerts",
        "dynamodb_table": "quarantine-table",
    }


# ---------------------------------------------------------------------------
# Tests: load_config — inline event payload (Req 12.2)
# ---------------------------------------------------------------------------


class TestLoadConfigInline:
    """Tests for loading config from inline event payload."""

    def test_loads_from_top_level_event_fields(self):
        """Config fields at event top level are recognised and parsed."""
        event = _valid_inline_event()
        config = load_config(event)

        assert config.aws_account_id == "123456789012"
        assert config.aws_region == "us-east-1"
        assert config.execution_mode == "dry_run"
        assert config.target_datasets == ["ds-1", "ds-2"]
        assert config.safety_threshold_pct == 0.7
        assert config.quarantine_period_days == 14
        assert config.s3_bucket == "my-bucket"
        assert config.s3_prefix == "spice-optimiser/"
        assert config.sns_topic_arn == "arn:aws:sns:us-east-1:123456789012:alerts"
        assert config.dynamodb_table == "quarantine-table"

    def test_loads_from_nested_config_key(self):
        """Config under a 'config' key in the event is recognised."""
        event = {"config": _valid_inline_event()}
        config = load_config(event)

        assert config.aws_account_id == "123456789012"
        assert config.execution_mode == "dry_run"

    def test_target_datasets_as_simple_list(self):
        """Target datasets can be provided as a plain list of strings."""
        event = _valid_inline_event()
        event["target_datasets"] = ["dataset-a", "dataset-b", "dataset-c"]
        config = load_config(event)

        assert config.target_datasets == ["dataset-a", "dataset-b", "dataset-c"]

    def test_target_datasets_as_structured_dict(self):
        """Target datasets in structured format with mode and dataset_ids."""
        event = _valid_inline_event()
        event["target_datasets"] = {
            "mode": "explicit",
            "dataset_ids": ["id-1", "id-2"],
        }
        config = load_config(event)

        assert config.target_datasets == ["id-1", "id-2"]

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None when not provided."""
        event = {
            "aws_account_id": "111111111111",
            "aws_region": "eu-west-1",
            "target_datasets": ["ds-1"],
            "s3_bucket": "bucket",
        }
        config = load_config(event)

        assert config.quarantine_period_days is None
        assert config.sns_topic_arn is None
        assert config.dynamodb_table is None

    def test_defaults_execution_mode_to_dry_run(self):
        """Execution mode defaults to dry_run when not specified (Req 5.1)."""
        event = {
            "aws_account_id": "111111111111",
            "aws_region": "eu-west-1",
            "target_datasets": ["ds-1"],
            "s3_bucket": "bucket",
        }
        config = load_config(event)

        assert config.execution_mode == ExecutionMode.DRY_RUN.value

    def test_defaults_safety_threshold_to_0_7(self):
        """Safety threshold defaults to 0.7 when not specified."""
        event = {
            "aws_account_id": "111111111111",
            "aws_region": "eu-west-1",
            "target_datasets": ["ds-1"],
            "s3_bucket": "bucket",
        }
        config = load_config(event)

        assert config.safety_threshold_pct == 0.7


# ---------------------------------------------------------------------------
# Tests: load_config — S3 source (Req 12.3)
# ---------------------------------------------------------------------------


class TestLoadConfigS3:
    """Tests for loading config from S3."""

    @patch("lambda_source.config.boto3.client")
    def test_loads_from_s3_when_bucket_and_key_provided(self, mock_boto_client):
        """Config is loaded from S3 when config_s3_bucket and config_s3_key are in event."""
        config_json = json.dumps(_valid_inline_event()).encode("utf-8")
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=config_json))
        }
        mock_boto_client.return_value = mock_s3

        event = {
            "config_s3_bucket": "config-bucket",
            "config_s3_key": "configs/optimiser.json",
        }
        config = load_config(event)

        mock_s3.get_object.assert_called_once_with(
            Bucket="config-bucket", Key="configs/optimiser.json"
        )
        assert config.aws_account_id == "123456789012"
        assert config.target_datasets == ["ds-1", "ds-2"]

    @patch("lambda_source.config.boto3.client")
    def test_s3_load_failure_raises_value_error(self, mock_boto_client):
        """ValueError is raised when S3 config cannot be read."""
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("Access Denied")
        mock_boto_client.return_value = mock_s3

        event = {
            "config_s3_bucket": "config-bucket",
            "config_s3_key": "missing.json",
        }
        with pytest.raises(ValueError, match="Failed to load configuration from s3://"):
            load_config(event)

    def test_no_config_source_raises_value_error(self):
        """ValueError is raised when event has no recognisable config source."""
        event = {"some_unrelated_key": "value"}
        with pytest.raises(ValueError, match="Cannot determine configuration source"):
            load_config(event)


# ---------------------------------------------------------------------------
# Tests: validate_config (Req 11.1, 11.2)
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config_returns_no_errors(self):
        """A fully valid config produces an empty error list."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="dry_run",
            safety_threshold_pct=0.7,
            quarantine_period_days=14,
            s3_bucket="my-bucket",
            s3_prefix="prefix/",
            sns_topic_arn="arn:aws:sns:us-east-1:123:topic",
            dynamodb_table="table",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert errors == []

    def test_missing_aws_account_id(self):
        """Missing aws_account_id produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            s3_bucket="bucket",
            aws_account_id="",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("aws_account_id" in e for e in errors)

    def test_missing_aws_region(self):
        """Missing aws_region produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="",
        )
        errors = validate_config(config)
        assert any("aws_region" in e for e in errors)

    def test_missing_s3_bucket(self):
        """Missing s3_bucket produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            s3_bucket="",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("s3_bucket" in e for e in errors)

    def test_empty_target_datasets(self):
        """Empty target_datasets list produces a validation error."""
        config = OptimiserConfig(
            target_datasets=[],
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("target_datasets" in e for e in errors)

    def test_safety_threshold_below_zero(self):
        """safety_threshold_pct below 0.0 produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            safety_threshold_pct=-0.1,
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("safety_threshold_pct" in e for e in errors)

    def test_safety_threshold_above_one(self):
        """safety_threshold_pct above 1.0 produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            safety_threshold_pct=1.01,
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("safety_threshold_pct" in e for e in errors)

    def test_safety_threshold_at_zero_is_valid(self):
        """safety_threshold_pct at 0.0 boundary is valid."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            safety_threshold_pct=0.0,
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert not any("safety_threshold_pct" in e for e in errors)

    def test_safety_threshold_at_one_is_valid(self):
        """safety_threshold_pct at 1.0 boundary is valid."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            safety_threshold_pct=1.0,
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert not any("safety_threshold_pct" in e for e in errors)

    def test_invalid_execution_mode(self):
        """Invalid execution_mode value produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="invalid_mode",
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("execution_mode" in e for e in errors)

    def test_dry_run_execution_mode_is_valid(self):
        """'dry_run' is a valid execution_mode."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="dry_run",
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert not any("execution_mode" in e for e in errors)

    def test_apply_execution_mode_is_valid(self):
        """'apply' is a valid execution_mode."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="apply",
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert not any("execution_mode" in e for e in errors)

    def test_multiple_validation_errors_reported(self):
        """Multiple issues in a single config all produce separate errors."""
        config = OptimiserConfig(
            target_datasets=[],
            execution_mode="bogus",
            safety_threshold_pct=2.0,
            s3_bucket="",
            aws_account_id="",
            aws_region="",
        )
        errors = validate_config(config)
        assert len(errors) >= 5  # At least one for each problem field


    def test_safety_threshold_nan_is_invalid(self):
        """safety_threshold_pct as NaN produces a validation error."""
        config = OptimiserConfig(
            target_datasets=["ds-1"],
            safety_threshold_pct=float("nan"),
            s3_bucket="bucket",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        errors = validate_config(config)
        assert any("safety_threshold_pct" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: load_config — malformed JSON handling (Req 11.1)
# ---------------------------------------------------------------------------


class TestLoadConfigMalformedInput:
    """Tests for handling malformed or invalid config input."""

    @patch("lambda_source.config.boto3.client")
    def test_malformed_json_from_s3_raises_value_error(self, mock_boto_client):
        """Malformed JSON from S3 raises ValueError."""
        malformed_content = b"{ this is not valid json }"
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=malformed_content))
        }
        mock_boto_client.return_value = mock_s3

        event = {
            "config_s3_bucket": "config-bucket",
            "config_s3_key": "configs/broken.json",
        }
        with pytest.raises(ValueError, match="Failed to load configuration from s3://"):
            load_config(event)

    def test_empty_event_raises_value_error(self):
        """An empty event dict raises ValueError (no config source)."""
        with pytest.raises(ValueError, match="Cannot determine configuration source"):
            load_config({})


# ---------------------------------------------------------------------------
# Tests: load_config — inline event overrides S3 (Req 12.2)
# ---------------------------------------------------------------------------


class TestLoadConfigPriority:
    """Tests that inline config takes priority over S3 config reference."""

    @patch("lambda_source.config.boto3.client")
    def test_inline_config_key_takes_priority_over_s3_reference(self, mock_boto_client):
        """When event has both a 'config' key and S3 references, inline wins.

        The inline config under the 'config' key should be used and S3
        should never be accessed.
        """
        inline_config = _valid_inline_event()
        inline_config["aws_account_id"] = "INLINE_ACCOUNT"

        event = {
            "config": inline_config,
            "config_s3_bucket": "config-bucket",
            "config_s3_key": "configs/optimiser.json",
        }

        config = load_config(event)

        # Inline config was used
        assert config.aws_account_id == "INLINE_ACCOUNT"
        # S3 was never called
        mock_boto_client.assert_not_called()

    @patch("lambda_source.config.boto3.client")
    def test_top_level_fields_take_priority_over_s3_reference(self, mock_boto_client):
        """When event has top-level config fields and S3 references, inline fields win."""
        event = {
            "aws_account_id": "TOP_LEVEL_ACCOUNT",
            "aws_region": "us-west-2",
            "target_datasets": ["ds-top"],
            "s3_bucket": "my-bucket",
            "config_s3_bucket": "config-bucket",
            "config_s3_key": "configs/optimiser.json",
        }

        config = load_config(event)

        # Top-level fields were used
        assert config.aws_account_id == "TOP_LEVEL_ACCOUNT"
        assert config.aws_region == "us-west-2"
        # S3 was never called
        mock_boto_client.assert_not_called()
