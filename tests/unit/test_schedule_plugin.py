# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the optional EventBridge schedule.

The schedule used to be created unconditionally, ENABLED, with no ``Input``. It
therefore delivered a bare EventBridge event, and the function had to find its own
configuration from an S3 ``config.json`` that was never written — so the rule fired every
7 days, did nothing, and reported nothing. Confirmed in CloudWatch: 3.5ms invocations
returning "cannot identify tool" going back to at least 2026-08-10.

It is now opt-in the same way quarantine is (``ScheduleEnabled``, default ``'false'``),
with a configurable frequency, mode, dataset list and safety threshold. The rule carries
a complete pipeline event in its ``Input``, so what a scheduled run will do is visible in
the template rather than in a second source of truth that can go missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[2] / "deployment" / "template.yaml"


class _CfnLoader(yaml.SafeLoader):
    """CloudFormation YAML uses short-form intrinsics that SafeLoader rejects."""


def _intrinsic(loader, tag_suffix, node):
    name = f"Fn::{tag_suffix}" if tag_suffix != "Ref" else "Ref"
    if isinstance(node, yaml.ScalarNode):
        return {name: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {name: loader.construct_sequence(node, deep=True)}
    return {name: loader.construct_mapping(node, deep=True)}


for tag in (
    "Ref", "Sub", "GetAtt", "Join", "Select", "Split", "If", "Equals", "Not",
    "And", "Or", "Condition", "FindInMap", "ImportValue", "Base64", "GetAZs",
):
    _CfnLoader.add_constructor(f"!{tag}", lambda l, n, t=tag: _intrinsic(l, t, n))
    yaml.add_multi_constructor("!", _intrinsic, Loader=_CfnLoader)


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.load(TEMPLATE.read_text(), Loader=_CfnLoader)


class TestOptionalByDefault:
    def test_schedule_enabled_defaults_to_false(self):
        """Opt-in, matching QuarantineEnabled. An unattended job should be something
        you ask for, not something you inherit."""
        tpl = yaml.load(TEMPLATE.read_text(), Loader=_CfnLoader)
        param = tpl["Parameters"]["ScheduleEnabled"]

        assert param["Default"] == "false"
        assert set(param["AllowedValues"]) == {"true", "false"}

    def test_the_condition_exists(self, template):
        assert "IsScheduleEnabled" in template["Conditions"]

    @pytest.mark.parametrize("resource", ["ScheduledRule", "SchedulePermission"])
    def test_schedule_resources_are_conditional(self, template, resource):
        """No rule and no Lambda permission unless enabled."""
        assert template["Resources"][resource]["Condition"] == "IsScheduleEnabled"

    def test_the_output_is_conditional_too(self, template):
        """An output referencing a resource that may not exist fails the whole stack."""
        assert template["Outputs"]["ScheduleRuleArn"]["Condition"] == "IsScheduleEnabled"


class TestConfigurability:
    def test_frequency_is_configurable(self, template):
        assert template["Parameters"]["ScheduleExpression"]["Default"] == "rate(7 days)"

    def test_execution_mode_defaults_to_dry_run(self, template):
        """apply on a schedule means unattended column deletion. It must be a
        deliberate choice, never the default."""
        param = template["Parameters"]["ScheduleExecutionMode"]

        assert param["Default"] == "dry_run"
        assert set(param["AllowedValues"]) == {"dry_run", "apply"}

    def test_target_datasets_is_configurable_and_empty_by_default(self, template):
        assert template["Parameters"]["ScheduleTargetDatasets"]["Default"] == ""

    def test_safety_threshold_is_bounded(self, template):
        param = template["Parameters"]["ScheduleSafetyThresholdPct"]

        assert param["Default"] == 0.7
        assert param["MinValue"] == 0.0
        assert param["MaxValue"] == 1.0


class TestRuleCarriesItsOwnConfig:
    def _input_template(self, template) -> str:
        target = template["Resources"]["ScheduledRule"]["Properties"]["Targets"][0]
        sub = target["Input"]["Fn::Sub"]
        # !Sub with a variable map is [template, {vars}]
        return sub[0] if isinstance(sub, list) else sub

    def test_the_rule_has_an_input(self, template):
        """The absence of one is what made the schedule silently useless."""
        target = template["Resources"]["ScheduledRule"]["Properties"]["Targets"][0]
        assert "Input" in target

    def test_the_input_is_valid_json_once_substituted(self, template):
        """Guards against a template that only fails at deploy time."""
        rendered = (
            self._input_template(template)
            .replace("${AWS::AccountId}", "123456789012")
            .replace("${AWS::Region}", "us-east-1")
            .replace("${ScheduleExecutionMode}", "dry_run")
            .replace("${ScheduleTargetDatasets}", "ds-1,ds-2")
            .replace("${ScheduleSafetyThresholdPct}", "0.7")
            .replace("${BucketName}", "bucket")
            .replace("${AWS::StackName}", "spice-optimiser")
        )
        event = json.loads(rendered)

        assert event["invocation_source"] == "schedule"
        assert event["execution_mode"] == "dry_run"
        assert event["target_datasets"] == "ds-1,ds-2"
        assert event["safety_threshold_pct"] == 0.7

    def test_the_rendered_event_routes_to_the_pipeline(self, template):
        """The reason the Input matters: it must be recognised as a pipeline run."""
        from lambda_source.handler import route_event

        rendered = (
            self._input_template(template)
            .replace("${AWS::AccountId}", "123456789012")
            .replace("${AWS::Region}", "us-east-1")
            .replace("${ScheduleExecutionMode}", "dry_run")
            .replace("${ScheduleTargetDatasets}", "ds-1")
            .replace("${ScheduleSafetyThresholdPct}", "0.7")
            .replace("${BucketName}", "bucket")
            .replace("${AWS::StackName}", "spice-optimiser")
        )
        assert route_event(json.loads(rendered), None) == "pipeline"

    def test_the_rendered_event_loads_as_config(self, template):
        """End to end: the rule's Input must survive load_config, including the
        comma-separated dataset list."""
        from lambda_source.config import load_config, validate_config

        rendered = (
            self._input_template(template)
            .replace("${AWS::AccountId}", "123456789012")
            .replace("${AWS::Region}", "us-east-1")
            .replace("${ScheduleExecutionMode}", "dry_run")
            .replace("${ScheduleTargetDatasets}", "ds-1,ds-2,ds-3")
            .replace("${ScheduleSafetyThresholdPct}", "0.7")
            .replace("${BucketName}", "bucket")
            .replace("${AWS::StackName}", "spice-optimiser")
        )
        config = load_config(json.loads(rendered))

        assert config.target_datasets == ["ds-1", "ds-2", "ds-3"]
        assert config.execution_mode == "dry_run"
        assert config.s3_bucket == "bucket"
        assert validate_config(config) == []


class TestCommaSeparatedDatasets:
    """CloudFormation has no map function, so the list arrives as a string."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ds-1", ["ds-1"]),
            ("ds-1,ds-2", ["ds-1", "ds-2"]),
            ("ds-1, ds-2 , ds-3", ["ds-1", "ds-2", "ds-3"]),
            ("ds-1,,ds-2", ["ds-1", "ds-2"]),
            ("ds-1,", ["ds-1"]),
            ("", []),
            ("   ", []),
            (",", []),
        ],
    )
    def test_string_forms(self, raw, expected):
        from lambda_source.config import _parse_target_datasets

        assert _parse_target_datasets(raw) == expected

    def test_existing_list_form_is_unchanged(self):
        from lambda_source.config import _parse_target_datasets

        assert _parse_target_datasets(["a", "b"]) == ["a", "b"]

    def test_existing_dict_form_is_unchanged(self):
        from lambda_source.config import _parse_target_datasets

        assert _parse_target_datasets(
            {"mode": "explicit", "dataset_ids": ["a", "b"]}
        ) == ["a", "b"]

    def test_dict_form_also_accepts_a_string(self):
        from lambda_source.config import _parse_target_datasets

        assert _parse_target_datasets({"dataset_ids": "a,b"}) == ["a", "b"]

    def test_blank_entries_never_become_dataset_ids(self):
        """A trailing comma in a stack parameter must not produce an empty ID that
        then fails a describe call with a confusing error."""
        from lambda_source.config import _parse_target_datasets

        assert "" not in _parse_target_datasets("a,,b,")


class TestBareScheduledEventFallback:
    def test_a_bare_event_still_gets_a_config_location(self):
        """An older rule with no Input, or a manual test, falls back to S3 config."""
        import os
        from unittest.mock import patch

        from lambda_source.handler import _scheduled_event_config

        env = {"S3_BUCKET": "b", "CONFIG_S3_KEY": "k/config.json"}
        with patch.dict(os.environ, env, clear=False):
            enriched = _scheduled_event_config(
                {"source": "aws.events", "detail-type": "Scheduled Event"}
            )

        assert enriched["config_s3_bucket"] == "b"
        assert enriched["config_s3_key"] == "k/config.json"
        assert enriched["invocation_source"] == "schedule"

    def test_a_self_describing_event_is_left_alone(self):
        """The new rule carries everything, so nothing should be injected."""
        from lambda_source.handler import _scheduled_event_config

        event = {
            "source": "aws.events",
            "target_datasets": "ds-1",
            "execution_mode": "dry_run",
            "s3_bucket": "bucket",
        }
        enriched = _scheduled_event_config(event)

        assert "config_s3_bucket" not in enriched
        assert "config_s3_key" not in enriched
        assert enriched["target_datasets"] == "ds-1"
