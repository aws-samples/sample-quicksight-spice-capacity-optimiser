#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Sync AgentCore MCP server config from CloudFormation stack outputs.

The gateway depends on resources created by the CloudFormation stack (the optimiser
Lambda, the Cognito user pool and app client, the MCP job-store table, the S3 bucket).
Their identifiers appear in three AgentCore config files:

  - agentcore/agentcore.json      -> gateway Lambda target ARN, CUSTOM_JWT authorizer
                                     (discoveryUrl, allowedClients), runtime `envVars`
  - agentcore-policy.json         -> IAM statement `Resource` ARNs
  - agentcore/aws-targets.json    -> deployment account and region (generated, not
                                     committed)

The committed copies of the first two carry the documentation placeholder account
``123456789012`` and the placeholder pool ``us-east-1_EXAMPLE``; this script reads the
live stack outputs and rewrites all three files so the values match the deployed
infrastructure. Run it before `agentcore deploy` (the deploy_mcp.sh wrapper does exactly
that, and restores the committed placeholders afterwards so a real account ID never
lands in a commit by accident).

The JSON-editing logic lives in pure functions (update_env_vars / update_policy)
so it can be unit-tested without AWS access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Map of AgentCore env-var name -> CloudFormation output key it is sourced from.
_ENV_FROM_OUTPUT = {
    "JOB_STORE_TABLE_NAME": "McpJobStoreTableName",
    "LAMBDA_FUNCTION_ARN": "LambdaFunctionArn",
    "S3_BUCKET": "S3BucketName",
}

# CloudFormation outputs the gateway's CUSTOM_JWT authorizer is sourced from.
_COGNITO_DISCOVERY_OUTPUT = "CognitoDiscoveryUrl"
_COGNITO_CLIENT_OUTPUT = "CognitoClientId"


def _s3_bucket_arn(bucket_name: str) -> str:
    """Build an S3 bucket ARN from a bucket name."""
    return f"arn:aws:s3:::{bucket_name}"


def update_env_vars(spec: dict, outputs: dict[str, str], s3_prefix: str) -> dict:
    """Return a copy of the AgentCore project spec with runtime envVars synced.

    Merges the derived env vars into each runtime's `envVars` list by name
    (existing entries with the same name are replaced; unrelated entries are
    preserved).

    The authorizer is synced separately by :func:`update_gateway_authorizer`. It used
    to be done inside this function's runtime loop, which is why it never ran — see
    that function's docstring.

    Args:
        spec: Parsed agentcore.json content.
        outputs: Map of CloudFormation OutputKey -> OutputValue.
        s3_prefix: Value for the S3_PREFIX env var (not a stack output).

    Returns:
        A new spec dict with updated envVars.
    """
    desired: dict[str, str] = {"S3_PREFIX": s3_prefix}
    for env_name, output_key in _ENV_FROM_OUTPUT.items():
        if output_key in outputs:
            desired[env_name] = outputs[output_key]

    updated = json.loads(json.dumps(spec))  # deep copy

    for runtime in updated.get("runtimes", []):
        existing = runtime.get("envVars", [])
        by_name: dict[str, dict] = {e["name"]: dict(e) for e in existing}
        for name, value in desired.items():
            by_name[name] = {"name": name, "value": value}
        # Preserve a stable order: existing order first, then any new names.
        ordered_names = [e["name"] for e in existing]
        for name in desired:
            if name not in ordered_names:
                ordered_names.append(name)
        runtime["envVars"] = [by_name[n] for n in ordered_names]

    return updated


def update_gateway_authorizer(
    spec: dict, outputs: dict[str, str]
) -> tuple[dict, list[str]]:
    """Return a copy of the spec with each gateway's CUSTOM_JWT authorizer synced.

    This sync existed but never ran. It was nested inside
    ``for runtime in updated.get("runtimes", [])`` in :func:`update_env_vars`, and this
    project has no runtime — the gateway invokes the Lambda directly through a Lambda
    target, so ``runtimes`` is ``[]`` and the loop body was unreachable. The authorizer
    is a property of the gateway, under ``agentCoreGateways[].authorizerConfiguration``,
    which is where this now looks.

    Every deploy reported success and silently changed nothing, so the authorizer had to
    be maintained by hand against the live gateway. Nothing failed; the config simply
    drifted from the stack that owns it.

    ``allowedClients`` is merged rather than replaced, so a client added deliberately by
    hand is not deleted by a routine sync. The cost of merging is that entries never
    leave on their own — which is exactly how an orphaned client ID survived in the
    config after its user pool was deleted. So anything present that the stack does
    not own is returned as a warning: kept, but not silently.

    Args:
        spec: Parsed agentcore.json content.
        outputs: Map of CloudFormation OutputKey -> OutputValue.

    Returns:
        ``(updated_spec, warnings)``. ``warnings`` names allowedClients entries the stack
        does not account for, for the caller to print.
    """
    updated = json.loads(json.dumps(spec))  # deep copy
    warnings: list[str] = []

    discovery = outputs.get(_COGNITO_DISCOVERY_OUTPUT)
    client_id = outputs.get(_COGNITO_CLIENT_OUTPUT)
    if not (discovery and client_id):
        return updated, warnings

    for gateway in updated.get("agentCoreGateways", []):
        # Field name is customJwtAuthorizer here, matching the agentcore.json schema in
        # .llm-context/agentcore.ts. The deployed API reports it as customJWTAuthorizer;
        # the CDK does that translation, so do not "correct" the spelling to match the
        # API response.
        jwt_config = (gateway.get("authorizerConfiguration") or {}).get(
            "customJwtAuthorizer"
        )
        if not jwt_config:
            continue

        jwt_config["discoveryUrl"] = discovery

        allowed = list(jwt_config.get("allowedClients") or [])
        if client_id not in allowed:
            allowed.append(client_id)
        jwt_config["allowedClients"] = allowed

        for entry in allowed:
            if entry != client_id:
                warnings.append(
                    f"gateway '{gateway.get('name')}' allows client '{entry}', which is "
                    f"not the stack's CognitoClientId. Kept — remove it by hand if it is "
                    f"stale."
                )

    return updated, warnings


def update_gateway_targets(spec: dict, outputs: dict[str, str]) -> dict:
    """Return a copy of the spec with every Lambda target pointed at the stack's function.

    The committed agentcore.json carries the documentation placeholder account
    ``123456789012`` in ``lambdaArn`` (the file is published, and a real account ID in
    it was a security-review finding), so the real ARN has to come from the stack at
    deploy time, exactly like the env vars and the authorizer. Covers both the targets
    attached to gateways and ``unassignedTargets``.

    Args:
        spec: Parsed agentcore.json content.
        outputs: Map of CloudFormation OutputKey -> OutputValue.

    Returns:
        A new spec dict with every ``lambdaFunctionArn.lambdaArn`` replaced.
    """
    updated = json.loads(json.dumps(spec))  # deep copy
    lambda_arn = outputs.get("LambdaFunctionArn")
    if not lambda_arn:
        return updated

    def _point_at_stack(target: dict) -> None:
        config = target.get("lambdaFunctionArn")
        if isinstance(config, dict) and "lambdaArn" in config:
            config["lambdaArn"] = lambda_arn

    for gateway in updated.get("agentCoreGateways", []):
        for target in gateway.get("targets", []):
            _point_at_stack(target)
    for target in updated.get("unassignedTargets", []):
        _point_at_stack(target)
    return updated


def account_id_from_arn(arn: str) -> str:
    """Return the account field of an ARN (``arn:partition:service:region:account:...``)."""
    parts = arn.split(":")
    if len(parts) < 6 or not parts[4].isdigit() or len(parts[4]) != 12:
        raise ValueError(f"not an ARN with an account id: {arn!r}")
    return parts[4]


def build_aws_targets(outputs: dict[str, str], region: str, target_name: str = "default") -> list[dict]:
    """Return the ``aws-targets.json`` content for the account and region the stack is in.

    aws-targets.json is not committed: it names the deployment account, and the only
    account it can sensibly name is the one the CloudFormation stack was deployed to,
    which the stack's own output ARNs tell us. Generating it here means a fresh clone
    deploys the gateway next to its Lambda without anyone editing an account ID in by
    hand (or committing one by accident).

    Args:
        outputs: Map of CloudFormation OutputKey -> OutputValue (needs LambdaFunctionArn).
        region: Region the stack was read from.
        target_name: AgentCore target name; the CDK stack is named after it.
    """
    lambda_arn = outputs.get("LambdaFunctionArn")
    if not lambda_arn:
        raise ValueError("LambdaFunctionArn output is required to derive the deployment account")
    return [{"name": target_name, "account": account_id_from_arn(lambda_arn), "region": region}]


def update_policy(policy: dict, outputs: dict[str, str]) -> dict:
    """Return a copy of the runtime IAM policy with Resource ARNs synced.

    Rewrites the `Resource` of each statement based on the actions it grants:
      - DynamoDB actions      -> McpJobStoreTableArn
      - lambda:InvokeFunction -> LambdaFunctionArn
      - s3:GetObject          -> <bucket-arn>/*
      - s3:ListBucket         -> <bucket-arn>

    Statements whose actions don't match (e.g. QuickSight) are left untouched.

    Args:
        policy: Parsed agentcore-policy.json content.
        outputs: Map of CloudFormation OutputKey -> OutputValue.

    Returns:
        A new policy dict with updated Resource values.
    """
    table_arn = outputs.get("McpJobStoreTableArn")
    lambda_arn = outputs.get("LambdaFunctionArn")
    bucket_name = outputs.get("S3BucketName")
    bucket_arn = _s3_bucket_arn(bucket_name) if bucket_name else None

    updated = json.loads(json.dumps(policy))  # deep copy

    for statement in updated.get("Statement", []):
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        action_set = set(actions)

        if any(a.startswith("dynamodb:") for a in action_set) and table_arn:
            statement["Resource"] = table_arn
        elif "lambda:InvokeFunction" in action_set and lambda_arn:
            statement["Resource"] = lambda_arn
        elif action_set == {"s3:GetObject"} and bucket_arn:
            statement["Resource"] = f"{bucket_arn}/*"
        elif action_set == {"s3:ListBucket"} and bucket_arn:
            statement["Resource"] = bucket_arn

    return updated


def _fetch_stack_outputs(stack_name: str, region: str, profile: str | None) -> dict[str, str]:
    """Read CloudFormation stack outputs into an OutputKey -> OutputValue map."""
    import boto3

    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    cfn = session.client("cloudformation", region_name=region)
    resp = cfn.describe_stacks(StackName=stack_name)
    stacks = resp.get("Stacks", [])
    if not stacks:
        raise SystemExit(f"Stack '{stack_name}' not found in {region}.")
    return {
        o["OutputKey"]: o["OutputValue"]
        for o in stacks[0].get("Outputs", [])
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-name", default="spice-optimiser")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--s3-prefix", default="spice-optimiser/")
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent.parent / "agentcore_project" / "spiceoptimisermcp"),
        help="Path to the AgentCore project (contains agentcore/).",
    )
    parser.add_argument(
        "--policy-file",
        default=str(Path(__file__).resolve().parent.parent / "agentcore-policy.json"),
    )
    parser.add_argument(
        "--target-name",
        default="default",
        help="AgentCore deployment target name written to aws-targets.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes, do not write.")
    args = parser.parse_args(argv)

    project_dir = Path(args.project_dir)
    agentcore_json = project_dir / "agentcore" / "agentcore.json"
    aws_targets_json = project_dir / "agentcore" / "aws-targets.json"
    policy_file = Path(args.policy_file)

    for required in (agentcore_json, policy_file):
        if not required.exists():
            raise SystemExit(f"Config file not found: {required}")

    outputs = _fetch_stack_outputs(args.stack_name, args.region, args.profile)

    missing = [k for k in ("McpJobStoreTableName", "McpJobStoreTableArn",
                           "LambdaFunctionArn", "S3BucketName") if k not in outputs]
    if missing:
        raise SystemExit(
            f"Stack '{args.stack_name}' is missing required outputs: {missing}. "
            "Deploy the CloudFormation stack first."
        )

    # Cognito outputs are optional (only present when CognitoEnabled=true)
    cognito_outputs = [k for k in ("CognitoDiscoveryUrl", "CognitoClientId") if k in outputs]
    if cognito_outputs:
        print(f"  Cognito outputs detected: {cognito_outputs}")
    else:
        print("  Cognito outputs not found (CognitoEnabled may be false). Skipping authorizer sync.")

    new_spec = update_env_vars(_read_json(agentcore_json), outputs, args.s3_prefix)
    new_spec, auth_warnings = update_gateway_authorizer(new_spec, outputs)
    new_spec = update_gateway_targets(new_spec, outputs)
    new_policy = update_policy(_read_json(policy_file), outputs)
    new_targets = build_aws_targets(outputs, args.region, args.target_name)

    for warning in auth_warnings:
        print(f"  WARNING: {warning}")

    if args.dry_run:
        print("=== agentcore.json (envVars) ===")
        for rt in new_spec.get("runtimes", []):
            print(f"  runtime {rt.get('name')}: {rt.get('envVars')}")
        if not new_spec.get("runtimes"):
            print("  (no runtimes — this project's gateway invokes the Lambda directly)")
        # Printed because it is the part that used to be skipped in silence. A dry run
        # that does not show the authorizer cannot tell you the sync is working.
        print("=== agentcore.json (gateway authorizer) ===")
        for gw in new_spec.get("agentCoreGateways", []):
            jwt = (gw.get("authorizerConfiguration") or {}).get("customJwtAuthorizer")
            if not jwt:
                print(f"  gateway {gw.get('name')}: no CUSTOM_JWT authorizer")
                continue
            print(f"  gateway {gw.get('name')}:")
            print(f"      discoveryUrl   : {jwt.get('discoveryUrl')}")
            print(f"      allowedClients : {jwt.get('allowedClients')}")
        print("=== agentcore.json (lambda targets) ===")
        for gw in new_spec.get("agentCoreGateways", []):
            for t in gw.get("targets", []):
                print(f"  {gw.get('name')}/{t.get('name')}: {(t.get('lambdaFunctionArn') or {}).get('lambdaArn')}")
        print("=== agentcore-policy.json (resources) ===")
        for s in new_policy.get("Statement", []):
            print(f"  {s.get('Action')} -> {s.get('Resource')}")
        print("=== aws-targets.json ===")
        for t in new_targets:
            print(f"  {t['name']}: account {t['account']}, region {t['region']}")
        return 0

    _write_json(agentcore_json, new_spec)
    _write_json(policy_file, new_policy)
    _write_json(aws_targets_json, new_targets)
    print(
        f"Synced {agentcore_json}, {policy_file} and {aws_targets_json} "
        f"from stack '{args.stack_name}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
