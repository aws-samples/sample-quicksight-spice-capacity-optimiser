# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Unit tests for scripts/sync_agentcore_config.py pure functions."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

# Load the script module by path (it lives in scripts/, not a package).
_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "sync_agentcore_config.py"
)
_spec = importlib.util.spec_from_file_location("sync_agentcore_config", _SCRIPT)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


@pytest.fixture
def outputs():
    return {
        "McpJobStoreTableName": "spice-optimiser-mcp-jobs",
        "McpJobStoreTableArn": "arn:aws:dynamodb:us-east-1:111122223333:table/spice-optimiser-mcp-jobs",
        "LambdaFunctionArn": "arn:aws:lambda:us-east-1:111122223333:function:spice-optimiser-function",
        "S3BucketName": "spice-optimiser-optimiserbucket-example1234",
    }


class TestUpdateEnvVars:
    def test_injects_env_vars_from_outputs(self, outputs):
        spec = {"runtimes": [{"name": "rt", "envVars": []}]}
        result = sync.update_env_vars(spec, outputs, "spice-optimiser/")
        env = {e["name"]: e["value"] for e in result["runtimes"][0]["envVars"]}
        assert env["JOB_STORE_TABLE_NAME"] == "spice-optimiser-mcp-jobs"
        assert env["LAMBDA_FUNCTION_ARN"] == outputs["LambdaFunctionArn"]
        assert env["S3_BUCKET"] == outputs["S3BucketName"]
        assert env["S3_PREFIX"] == "spice-optimiser/"

    def test_replaces_existing_and_preserves_unrelated(self, outputs):
        spec = {
            "runtimes": [
                {
                    "name": "rt",
                    "envVars": [
                        {"name": "S3_BUCKET", "value": "old-bucket"},
                        {"name": "CUSTOM_FLAG", "value": "keep-me"},
                    ],
                }
            ]
        }
        result = sync.update_env_vars(spec, outputs, "spice-optimiser/")
        env = {e["name"]: e["value"] for e in result["runtimes"][0]["envVars"]}
        assert env["S3_BUCKET"] == outputs["S3BucketName"]  # replaced
        assert env["CUSTOM_FLAG"] == "keep-me"  # preserved

    def test_does_not_mutate_input(self, outputs):
        spec = {"runtimes": [{"name": "rt", "envVars": []}]}
        sync.update_env_vars(spec, outputs, "spice-optimiser/")
        assert spec["runtimes"][0]["envVars"] == []

    def test_omits_env_var_when_output_missing(self):
        spec = {"runtimes": [{"name": "rt", "envVars": []}]}
        result = sync.update_env_vars(spec, {"S3BucketName": "b"}, "p/")
        names = {e["name"] for e in result["runtimes"][0]["envVars"]}
        assert "S3_BUCKET" in names
        assert "LAMBDA_FUNCTION_ARN" not in names  # output absent


class TestUpdatePolicy:
    def test_rewrites_resource_arns_by_action(self, outputs):
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["quicksight:ListDataSets"], "Resource": "*"},
                {
                    "Effect": "Allow",
                    "Action": ["dynamodb:PutItem", "dynamodb:Query"],
                    "Resource": "arn:aws:dynamodb:us-east-1:000000000000:table/old",
                },
                {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": "old"},
                {"Effect": "Allow", "Action": "s3:GetObject", "Resource": "old"},
                {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": "old"},
            ],
        }
        result = sync.update_policy(policy, outputs)
        stmts = result["Statement"]
        assert stmts[0]["Resource"] == "*"  # quicksight untouched
        assert stmts[1]["Resource"] == outputs["McpJobStoreTableArn"]
        assert stmts[2]["Resource"] == outputs["LambdaFunctionArn"]
        assert stmts[3]["Resource"] == f"arn:aws:s3:::{outputs['S3BucketName']}/*"
        assert stmts[4]["Resource"] == f"arn:aws:s3:::{outputs['S3BucketName']}"

    def test_does_not_mutate_input(self, outputs):
        policy = {
            "Statement": [
                {"Action": "lambda:InvokeFunction", "Resource": "old"},
            ]
        }
        sync.update_policy(policy, outputs)
        assert policy["Statement"][0]["Resource"] == "old"


# ── gateway authorizer ───────────────────────────────────────────────────────
#
# This sync existed and never ran. It was nested inside
# `for runtime in updated.get("runtimes", [])` in update_env_vars, and this project has
# no runtime - the gateway invokes the Lambda directly through a Lambda target, so
# `runtimes` is [] and the loop body was unreachable. The authorizer belongs to the
# gateway, under agentCoreGateways[].authorizerConfiguration.
#
# Every deploy reported success and changed nothing, so the authorizer had to be
# maintained by hand against the live gateway. Six tests covered this module and none
# of them touched the authorizer, which is why nobody noticed for six weeks.
#
# The fixture below deliberately mirrors the real agentcore.json shape: runtimes empty,
# the gateway carrying the authorizer. A fixture with a populated `runtimes` would have
# let the old code pass.

REAL_POOL_DISCOVERY = (
    "https://cognito-idp.us-east-1.amazonaws.com/"
    "us-east-1_EXAMPLE/.well-known/openid-configuration"
)


@pytest.fixture
def cognito_outputs(outputs):
    return {
        **outputs,
        "CognitoDiscoveryUrl": REAL_POOL_DISCOVERY,
        "CognitoClientId": "1example2stackclientid3456",
    }


def _gateway_spec(allowed_clients=None, discovery="https://stale.example/openid"):
    """The shape agentcore.json actually has: no runtimes, authorizer on the gateway."""
    return {
        "runtimes": [],
        "agentCoreGateways": [
            {
                "name": "spice-optimiser-gateway",
                "authorizerType": "CUSTOM_JWT",
                "authorizerConfiguration": {
                    # Spelled customJwtAuthorizer to match .llm-context/agentcore.ts.
                    # The deployed API reports customJWTAuthorizer; the CDK translates.
                    "customJwtAuthorizer": {
                        "discoveryUrl": discovery,
                        "allowedClients": (
                            ["1example2stackclientid3456"]
                            if allowed_clients is None
                            else allowed_clients
                        ),
                        "allowedScopes": ["spice-optimiser-gateway/tools.read"],
                    }
                },
            }
        ],
    }


def _jwt(spec):
    return spec["agentCoreGateways"][0]["authorizerConfiguration"]["customJwtAuthorizer"]


class TestUpdateGatewayAuthorizer:
    def test_it_reaches_the_gateway_despite_runtimes_being_empty(self, cognito_outputs):
        """The whole defect in one assertion."""
        spec = _gateway_spec()
        assert spec["runtimes"] == [], "fixture must match reality or it proves nothing"

        updated, _ = sync.update_gateway_authorizer(spec, cognito_outputs)

        assert _jwt(updated)["discoveryUrl"] == REAL_POOL_DISCOVERY

    def test_the_stack_client_is_added_when_absent(self, cognito_outputs):
        updated, _ = sync.update_gateway_authorizer(
            _gateway_spec(allowed_clients=[]), cognito_outputs
        )

        assert _jwt(updated)["allowedClients"] == ["1example2stackclientid3456"]

    def test_it_does_not_duplicate_a_client_already_present(self, cognito_outputs):
        updated, _ = sync.update_gateway_authorizer(_gateway_spec(), cognito_outputs)

        assert _jwt(updated)["allowedClients"] == ["1example2stackclientid3456"]

    def test_a_hand_added_client_is_kept_but_warned_about(self, cognito_outputs):
        """Merging, not replacing - but not in silence either.

        Deleting a client someone added deliberately would be worse than keeping it. The
        cost of merging is that entries never leave on their own, which is exactly how
        2example3orphanedclient789 outlived the deletion of its user pool. So it is kept
        and reported.
        """
        spec = _gateway_spec(
            allowed_clients=["2example3orphanedclient789", "1example2stackclientid3456"]
        )

        updated, warnings = sync.update_gateway_authorizer(spec, cognito_outputs)

        assert "2example3orphanedclient789" in _jwt(updated)["allowedClients"]
        assert len(warnings) == 1
        assert "2example3orphanedclient789" in warnings[0]

    def test_no_warning_when_only_the_stack_client_is_allowed(self, cognito_outputs):
        _, warnings = sync.update_gateway_authorizer(_gateway_spec(), cognito_outputs)

        assert warnings == []

    def test_it_is_a_no_op_without_cognito_outputs(self, outputs):
        """CognitoEnabled=false is a supported configuration, not an error."""
        spec = _gateway_spec()
        updated, warnings = sync.update_gateway_authorizer(spec, outputs)

        assert _jwt(updated)["discoveryUrl"] == "https://stale.example/openid"
        assert warnings == []

    def test_a_gateway_without_a_jwt_authorizer_is_skipped(self, cognito_outputs):
        spec = {"runtimes": [], "agentCoreGateways": [{"name": "gw"}]}

        updated, warnings = sync.update_gateway_authorizer(spec, cognito_outputs)

        assert updated["agentCoreGateways"][0] == {"name": "gw"}
        assert warnings == []

    def test_does_not_mutate_input(self, cognito_outputs):
        spec = _gateway_spec()
        before = json.dumps(spec, sort_keys=True)

        sync.update_gateway_authorizer(spec, cognito_outputs)

        assert json.dumps(spec, sort_keys=True) == before


class TestTheRealConfigStaysInSync:
    """Reads the committed agentcore.json, so drift shows up here rather than on deploy."""

    def _real_spec(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "agentcore_project"
            / "spiceoptimisermcp"
            / "agentcore"
            / "agentcore.json"
        )
        return json.loads(path.read_text())

    def test_syncing_the_committed_config_produces_no_warnings(self, cognito_outputs):
        """It did produce one, until the orphaned client was removed.

        2example3orphanedclient789 was allowed by the gateway while existing in none of
        the account's twelve user pools. Unexploitable - the discoveryUrl pins validation
        to one pool, so no token could ever have been minted for it - but an allow-list
        entry nobody can account for is worth not having.
        """
        _, warnings = sync.update_gateway_authorizer(
            self._real_spec(), cognito_outputs
        )

        assert warnings == [], f"agentcore.json allows a client the stack does not own: {warnings}"

    def test_the_committed_config_still_has_a_jwt_authorizer_to_sync(self):
        """Guards the premise: if the shape changes, the test above passes vacuously."""
        jwt = _jwt(self._real_spec())

        assert jwt["discoveryUrl"].startswith("https://cognito-idp.")
        assert "allowedClients" in jwt

    def test_the_committed_config_names_no_real_account_pool_or_client(self):
        """The file is published. It carried a real account ID, a live user pool ID and a
        live app client ID until the 2026-10 security review; the sync fills all three
        from the stack at deploy time, and deploy_mcp.sh restores the placeholders after,
        so there is no reason for a real value to be in the committed file."""
        text = json.dumps(self._real_spec())
        jwt = _jwt(self._real_spec())

        assert jwt["allowedClients"] == [], "the stack's client is added at deploy time"
        assert "us-east-1_EXAMPLE" in jwt["discoveryUrl"]
        accounts = set(re.findall(r"arn:aws:[a-z-]+:[a-z0-9-]*:(\d{12}):", text))
        assert accounts <= {"123456789012"}, f"real account id in agentcore.json: {accounts}"

    def test_the_committed_policy_names_no_real_account(self):
        path = Path(__file__).resolve().parents[2] / "agentcore-policy.json"
        accounts = set(re.findall(r":(\d{12}):", path.read_text()))

        assert accounts <= {"123456789012"}, f"real account id in agentcore-policy.json: {accounts}"

    def test_aws_targets_is_generated_not_committed(self):
        """It names the deployment account, so the sync writes it and git ignores it."""
        root = Path(__file__).resolve().parents[2]
        tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).split()

        assert "agentcore_project/spiceoptimisermcp/agentcore/aws-targets.json" not in tracked
        assert "agentcore_project/spiceoptimisermcp/agentcore/aws-targets.example.json" in tracked


class TestUpdateGatewayTargets:
    """The gateway's Lambda target ARN is account-specific, so it comes from the stack."""

    def _spec(self):
        placeholder = {
            "lambdaArn": "arn:aws:lambda:us-east-1:123456789012:function:spice-optimiser-function",
            "toolSchemaFile": "tool-schema.json",
        }
        return {
            "agentCoreGateways": [{
                "name": "gw",
                "targets": [{"name": "t", "targetType": "lambdaFunctionArn", "lambdaFunctionArn": dict(placeholder)}],
            }],
            "unassignedTargets": [{"name": "t", "targetType": "lambdaFunctionArn", "lambdaFunctionArn": dict(placeholder)}],
        }

    def test_every_lambda_target_points_at_the_stacks_function(self, outputs):
        updated = sync.update_gateway_targets(self._spec(), outputs)

        arns = [updated["agentCoreGateways"][0]["targets"][0]["lambdaFunctionArn"]["lambdaArn"],
                updated["unassignedTargets"][0]["lambdaFunctionArn"]["lambdaArn"]]
        assert arns == [outputs["LambdaFunctionArn"]] * 2

    def test_the_tool_schema_reference_is_untouched(self, outputs):
        updated = sync.update_gateway_targets(self._spec(), outputs)

        assert updated["agentCoreGateways"][0]["targets"][0]["lambdaFunctionArn"]["toolSchemaFile"] == "tool-schema.json"

    def test_without_the_output_nothing_changes(self):
        spec = self._spec()

        assert sync.update_gateway_targets(spec, {}) == spec

    def test_does_not_mutate_input(self, outputs):
        spec = self._spec()
        before = json.dumps(spec, sort_keys=True)

        sync.update_gateway_targets(spec, outputs)

        assert json.dumps(spec, sort_keys=True) == before

    def test_the_committed_targets_are_rewritten_by_the_sync(self, outputs):
        """End to end on the real file: after a sync no placeholder account remains."""
        path = Path(__file__).resolve().parents[2] / "agentcore_project" / "spiceoptimisermcp" / "agentcore" / "agentcore.json"
        updated = sync.update_gateway_targets(json.loads(path.read_text()), outputs)

        assert "123456789012" not in json.dumps(updated["agentCoreGateways"])
        assert "123456789012" not in json.dumps(updated["unassignedTargets"])


class TestBuildAwsTargets:
    def test_account_comes_from_the_stacks_own_arn(self, outputs):
        targets = sync.build_aws_targets(outputs, "us-east-1")

        assert targets == [{"name": "default", "account": "111122223333", "region": "us-east-1"}]

    def test_target_name_and_region_are_passed_through(self, outputs):
        targets = sync.build_aws_targets(outputs, "eu-west-1", target_name="staging")

        assert targets == [{"name": "staging", "account": "111122223333", "region": "eu-west-1"}]

    def test_it_refuses_to_guess_without_the_lambda_arn(self):
        with pytest.raises(ValueError):
            sync.build_aws_targets({}, "us-east-1")

    @pytest.mark.parametrize("bad", ["not-an-arn", "arn:aws:lambda:us-east-1::function:x", "arn:aws:lambda:us-east-1:12345:function:x"])
    def test_a_malformed_arn_is_rejected(self, bad):
        with pytest.raises(ValueError):
            sync.account_id_from_arn(bad)
