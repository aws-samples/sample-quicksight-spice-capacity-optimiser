# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the quarantine plumbing: config resolution and the pipeline refusal.

Quarantine had four disagreeing sources of truth — README said 14 days,
`configuration/config.json` said 14, the `OptimiserConfig` default was `None`, and the
deployed stack had `QuarantineEnabled=false` with an empty table name. But the real
defect was subtler than the inconsistency: even with everything provisioned correctly,
it could not run.

CloudFormation sets `DYNAMODB_TABLE` on the function whenever
`QuarantineEnabled=true`. `_parse_config` read `dynamodb_table` only from the event
payload, and `_start_job` never put it there. So `dynamodb_table` resolved to `None`,
`QuarantineManager.enabled` was `False`, the handler's compound gate fell through
silently, and a run that had asked for a 14-day waiting period removed columns
immediately and reported success.

Two things are pinned here: the environment fallback that makes it reachable, and the
refusal that makes "requested but unavailable" loud instead of silent. Quarantine stays
optional — omitting the parameter is the supported way to run without it.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from lambda_source.config import load_config
from lambda_source.models import OptimiserConfig
from lambda_source.quarantine import QuarantineManager
from lambda_source.qs_client import QSClient

BASE_PAYLOAD = {
    "target_datasets": ["ds-1"],
    "execution_mode": "dry_run",
    "aws_account_id": "9" * 12,
    "aws_region": "us-east-1",
    "s3_bucket": "bucket",
    "s3_prefix": "prefix/",
}


class TestConfigEnvironmentFallback:
    def test_dynamodb_table_falls_back_to_the_environment(self):
        """The link that was missing. CloudFormation owns this name and sets it on the
        function; requiring every caller to echo it back is what broke the feature."""
        with patch.dict(os.environ, {"DYNAMODB_TABLE": "quarantine-table"}, clear=False):
            config = load_config(dict(BASE_PAYLOAD, quarantine_period_days=14))

        assert config.dynamodb_table == "quarantine-table"

    def test_the_full_chain_now_reaches_an_enabled_manager(self):
        """End to end: env var set -> config populated -> manager enabled -> gate open."""
        with patch.dict(os.environ, {"DYNAMODB_TABLE": "quarantine-table"}, clear=False):
            config = load_config(dict(BASE_PAYLOAD, quarantine_period_days=14))

        with patch("lambda_source.quarantine.boto3"):
            manager = QuarantineManager(dynamodb_table=config.dynamodb_table)

        assert config.quarantine_period_days == 14
        assert manager.enabled is True
        assert (config.quarantine_period_days is not None and manager.enabled) is True

    def test_payload_wins_over_the_environment(self):
        """So a run can be pointed at a different table."""
        with patch.dict(os.environ, {"DYNAMODB_TABLE": "from-env"}, clear=False):
            config = load_config(dict(BASE_PAYLOAD, dynamodb_table="from-payload"))

        assert config.dynamodb_table == "from-payload"

    @pytest.mark.parametrize("env_value", ["", "   "])
    def test_an_empty_environment_value_means_disabled(self, env_value):
        """QuarantineEnabled=false leaves DYNAMODB_TABLE as an empty string, not unset.
        An empty string must not be mistaken for a table name."""
        with patch.dict(os.environ, {"DYNAMODB_TABLE": env_value}, clear=False):
            config = load_config(dict(BASE_PAYLOAD))

        assert config.dynamodb_table is None
        assert QuarantineManager(dynamodb_table=config.dynamodb_table).enabled is False

    def test_absent_environment_variable_means_disabled(self):
        env = {k: v for k, v in os.environ.items() if k != "DYNAMODB_TABLE"}
        with patch.dict(os.environ, env, clear=True):
            config = load_config(dict(BASE_PAYLOAD))

        assert config.dynamodb_table is None

    def test_sns_topic_arn_uses_the_same_fallback(self):
        """Same class of value, same ownership, so it gets the same treatment."""
        with patch.dict(os.environ, {"SNS_TOPIC_ARN": "arn:aws:sns:::t"}, clear=False):
            config = load_config(dict(BASE_PAYLOAD))

        assert config.sns_topic_arn == "arn:aws:sns:::t"


class TestPipelineRefusal:
    """The pipeline refuses to proceed without a guardrail it was asked for."""

    def _process(self, quarantine_days, table):
        from lambda_source.handler import Orchestrator

        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="dry_run",
            s3_bucket="bucket",
            s3_prefix="prefix/",
            aws_account_id="9" * 12,
            aws_region="us-east-1",
            quarantine_period_days=quarantine_days,
            dynamodb_table=table,
        )

        columns = [
            {"Name": "used_col", "Type": "STRING"},
            {"Name": "unused_col", "Type": "STRING"},
        ]
        qs = create_autospec(QSClient, instance=True)
        qs.describe_dataset.return_value = {
            "DataSetId": "ds-1",
            "Name": "orders",
            "ImportMode": "SPICE",
            "OutputColumns": columns,
            "PhysicalTableMap": {"pt": {"RelationalTable": {"InputColumns": columns}}},
            "LogicalTableMap": {},
        }
        qs.list_analyses.return_value = []
        qs.list_dashboards.return_value = []
        qs.list_datasets.return_value = []

        with patch("lambda_source.handler.QSClient", return_value=qs), patch(
            "lambda_source.handler.boto3"
        ), patch("lambda_source.reporter.boto3"), patch(
            "lambda_source.snapshot.boto3"
        ), patch(
            "lambda_source.notifications.boto3"
        ), patch(
            "lambda_source.dataset_lock.boto3"
        ), patch(
            "lambda_source.quarantine.boto3"
        ):
            orch = Orchestrator(config)
            orch._reporter._s3_client = MagicMock()
            orch._reporter.store_report = lambda report, dsid: "key.json"
            return orch.process_dataset("ds-1")

    def test_requested_without_a_table_fails_the_dataset(self):
        """Rather than removing columns the operator believed were protected."""
        result = self._process(quarantine_days=14, table=None)

        assert result.status == "failed"
        assert "quarantine" in result.error_message.lower()
        assert "QuarantineEnabled=true" in result.error_message

    def test_the_error_reports_the_requested_period(self):
        result = self._process(quarantine_days=30, table=None)
        assert "30" in result.error_message

    def test_not_requested_proceeds_normally(self):
        """Quarantine is optional. This is the default and must stay unaffected."""
        result = self._process(quarantine_days=None, table=None)

        assert result.status != "failed"
        assert result.error_message is None

    def test_requested_with_a_table_proceeds(self):
        result = self._process(quarantine_days=14, table="quarantine-table")

        assert result.status != "failed"

    def test_zero_days_without_a_table_still_fails(self):
        """Zero is a real choice — track, but with no waiting period — and still needs
        somewhere to record state. It must not be treated as "not requested"."""
        result = self._process(quarantine_days=0, table=None)

        assert result.status == "failed"
