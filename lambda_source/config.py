# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Configuration loading and validation for the SPICE Dataset Optimiser.

Supports loading configuration from:
- Inline event payload (direct Lambda invocation)
- S3 bucket (EventBridge-triggered execution)

Validates required fields, value ranges, and enumeration constraints.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3

from lambda_source.models import ExecutionMode, OptimiserConfig

logger = logging.getLogger(__name__)


def load_config(event: dict) -> OptimiserConfig:
    """Load configuration from S3 or inline event payload.

    Resolution order:
    1. If event contains a "config" key with a dict value, use it as inline config.
    2. If event itself contains config fields (e.g. "aws_account_id"), treat the
       event as the inline config.
    3. Otherwise, load from S3 using "config_s3_bucket" and "config_s3_key" from
       the event.

    Args:
        event: Lambda event payload.

    Returns:
        Parsed OptimiserConfig dataclass.

    Raises:
        ValueError: If configuration cannot be located or parsed.
    """
    config_data = _resolve_config_source(event)
    return _parse_config(config_data)


def validate_config(config: OptimiserConfig) -> list[str]:
    """Validate configuration values.

    Checks:
    - Required fields are present and non-empty.
    - safety_threshold_pct is within [0.0, 1.0].
    - execution_mode is a valid value ("dry_run" or "apply").

    Args:
        config: OptimiserConfig instance to validate.

    Returns:
        List of validation error strings. Empty list means valid.
    """
    errors: list[str] = []

    # Required fields
    if not config.aws_account_id:
        errors.append("aws_account_id is required")
    if not config.aws_region:
        errors.append("aws_region is required")
    if not config.s3_bucket:
        errors.append("s3_bucket is required")
    if not config.target_datasets:
        errors.append("target_datasets is required and must not be empty")

    # Safety threshold range
    if not (0.0 <= config.safety_threshold_pct <= 1.0):
        errors.append(
            f"safety_threshold_pct must be between 0.0 and 1.0, "
            f"got {config.safety_threshold_pct}"
        )

    # Execution mode validation
    valid_modes = {mode.value for mode in ExecutionMode}
    if config.execution_mode not in valid_modes:
        errors.append(
            f"execution_mode must be one of {sorted(valid_modes)}, "
            f"got '{config.execution_mode}'"
        )

    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = {
    "aws_account_id",
    "aws_region",
    "execution_mode",
    "target_datasets",
    "safety_threshold_pct",
    "quarantine_period_days",
    "s3_bucket",
    "s3_prefix",
    "sns_topic_arn",
    "dynamodb_table",
    "cascade_child_refresh",
}


def _resolve_config_source(event: dict) -> dict:
    """Determine where config lives and return it as a raw dict.

    Args:
        event: Lambda event payload.

    Returns:
        Raw configuration dictionary.

    Raises:
        ValueError: If no config source can be determined.
    """
    # Case 1: Inline config under "config" key
    if "config" in event and isinstance(event["config"], dict):
        logger.info("Loading configuration from inline 'config' key in event payload")
        return event["config"]

    # Case 2: Event itself contains config fields
    if _CONFIG_FIELDS & set(event.keys()):
        logger.info("Loading configuration from inline event payload fields")
        return event

    # Case 3: Load from S3
    s3_bucket = event.get("config_s3_bucket")
    s3_key = event.get("config_s3_key")

    if s3_bucket and s3_key:
        logger.info(
            "Loading configuration from S3: s3://%s/%s", s3_bucket, s3_key
        )
        return _load_from_s3(s3_bucket, s3_key)

    raise ValueError(
        "Cannot determine configuration source. Provide inline config in "
        "the event payload (under 'config' key or as top-level fields), "
        "or provide 'config_s3_bucket' and 'config_s3_key' in the event."
    )


def _load_from_s3(bucket: str, key: str) -> dict:
    """Load and parse JSON configuration from S3.

    Args:
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        ValueError: If the S3 object cannot be read or parsed.
    """
    try:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)
    except Exception as exc:
        raise ValueError(
            f"Failed to load configuration from s3://{bucket}/{key}: {exc}"
        ) from exc


def _env_or(data: dict, key: str, env_var: str) -> str | None:
    """Config value, falling back to the environment when the payload omits it.

    CloudFormation provisions these and exposes them on the function, so the
    environment is the authoritative source for *where* state lives. Requiring every
    caller to echo them back in the payload was a real defect: ``DYNAMODB_TABLE`` was
    set on the function whenever ``QuarantineEnabled=true``, but ``_parse_config`` read
    only the payload, so ``dynamodb_table`` came out ``None``, ``QuarantineManager`` was
    disabled, and quarantine was skipped silently even when explicitly requested.

    Payload still wins when present, so a caller can point a run at a different table.
    """
    value = data.get(key)
    if value:
        return value

    from_env = os.environ.get(env_var, "").strip()
    return from_env or None


def _parse_config(data: dict) -> OptimiserConfig:
    """Parse a raw configuration dictionary into an OptimiserConfig dataclass.

    Handles the target_datasets field which can be:
    - A list of dataset ID strings (simple form)
    - A dict with "mode" and "dataset_ids" keys (structured form from config schema)

    ``dynamodb_table`` and ``sns_topic_arn`` fall back to the environment, because
    CloudFormation owns those names and every invocation path runs inside the function
    it set them on.

    Args:
        data: Raw configuration dictionary.

    Returns:
        OptimiserConfig instance.

    Raises:
        ValueError: If the configuration structure is invalid.
    """
    target_datasets = _parse_target_datasets(data.get("target_datasets"))

    return OptimiserConfig(
        target_datasets=target_datasets,
        execution_mode=data.get("execution_mode", ExecutionMode.DRY_RUN.value),
        safety_threshold_pct=float(data.get("safety_threshold_pct", 0.7)),
        quarantine_period_days=data.get("quarantine_period_days"),
        s3_bucket=data.get("s3_bucket", ""),
        s3_prefix=data.get("s3_prefix", ""),
        sns_topic_arn=_env_or(data, "sns_topic_arn", "SNS_TOPIC_ARN"),
        dynamodb_table=_env_or(data, "dynamodb_table", "DYNAMODB_TABLE"),
        aws_account_id=data.get("aws_account_id", ""),
        aws_region=data.get("aws_region", ""),
        skip_unreachable_consumers=bool(data.get("skip_unreachable_consumers", False)),
        cascade_child_refresh=bool(data.get("cascade_child_refresh", False)),
    )


def _parse_target_datasets(value: Any) -> list[str]:
    """Parse the target_datasets field from various formats.

    Supports:
    - A list of strings: ["dataset-1", "dataset-2"]
    - A dict with "dataset_ids" key: {"mode": "explicit", "dataset_ids": [...]}
    - A comma-separated string: "dataset-1,dataset-2"
    - None or missing: returns empty list

    The string form exists for callers that cannot express a JSON array. The
    EventBridge rule is the driving case: CloudFormation has no map function, so
    quoting each element of a ``CommaDelimitedList`` inside the rule's ``Input`` is not
    expressible. It is also the natural shape for a CLI flag or an environment variable.

    Args:
        value: Raw target_datasets value from config.

    Returns:
        List of dataset identifier strings, with blanks and surrounding whitespace
        removed so a trailing comma or a padded list does not produce empty IDs.
    """
    if value is None:
        return []

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, dict):
        dataset_ids = value.get("dataset_ids", [])
        if isinstance(dataset_ids, str):
            return [item.strip() for item in dataset_ids.split(",") if item.strip()]
        if isinstance(dataset_ids, list):
            return [str(item).strip() for item in dataset_ids if str(item).strip()]
        return []

    return []
