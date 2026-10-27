# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The IAM policy must grant the S3 calls the code actually makes.

This is the third time the same shape of defect has landed: the code calls an API, the
role is not allowed to call it, the failure is swallowed, and the symptom surfaces
somewhere unrelated.

    metrics       PutMetricData denied by a namespace condition -> no telemetry at all
    lock release  DeleteObject never granted                    -> every dataset locked
                                                                   for 30 minutes

Both were invisible for the same reason, and it is a defensible reason in each case:
telemetry must not change what a caller is told, and a release running in a ``finally``
must not mask the exception that sent it there. Swallowing is correct. Not noticing is
the bug, and the place to notice is here, where the template and the code can be
compared without deploying anything.

The lock case was worse than the metrics case in one respect. A missing metric is
invisible. An unreleasable lock is actively misleading: the next operation on that
dataset is refused as "already being optimised by another run", which reads as
contention and sends you looking for a concurrent run that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "deployment" / "template.yaml"
LOCK_MODULE = ROOT / "lambda_source" / "dataset_lock.py"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that tolerates CloudFormation's short-form intrinsics.

    ``!Sub`` and friends are unknown tags to SafeLoader, and the alternative - parsing
    the policy out with a regex - is what let the original grant go unchecked.
    """


def _intrinsic(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> object:
    name = f"Fn::{tag_suffix}" if tag_suffix != "Ref" else "Ref"
    if isinstance(node, yaml.ScalarNode):
        return {name: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {name: loader.construct_sequence(node)}
    return {name: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _intrinsic)


def _template() -> dict:
    return yaml.load(TEMPLATE.read_text(), Loader=_CfnLoader)


def _s3_statements() -> list[dict]:
    role = _template()["Resources"]["LambdaExecutionRole"]["Properties"]
    for policy in role["Policies"]:
        if policy["PolicyName"] == "S3AccessPolicy":
            return policy["PolicyDocument"]["Statement"]
    pytest.fail("S3AccessPolicy not found on LambdaExecutionRole")


def _granted_actions() -> set[str]:
    actions: set[str] = set()
    for statement in _s3_statements():
        if statement.get("Effect") != "Allow":
            continue
        declared = statement.get("Action", [])
        actions.update([declared] if isinstance(declared, str) else declared)
    return actions


def test_the_lock_release_delete_is_granted() -> None:
    """The exact call whose absence locked every dataset for the full TTL."""
    assert "s3:DeleteObject" in _granted_actions(), (
        "dataset_lock.release() calls delete_object. Without s3:DeleteObject the "
        "release fails silently and the lock survives to its 30-minute expiry, so the "
        "next operation on that dataset is refused as locked."
    )


def test_every_s3_call_in_the_lock_module_is_granted() -> None:
    """Derived from the source, so a new call has to be granted or this fails.

    Reads the boto3 method names out of dataset_lock.py rather than listing them here,
    because a hardcoded list is one more thing that can fall out of step with the code -
    which is the failure mode this whole module exists to catch.
    """
    method_to_action = {
        "get_object": "s3:GetObject",
        "put_object": "s3:PutObject",
        "delete_object": "s3:DeleteObject",
        "head_object": "s3:GetObject",
        "list_objects_v2": "s3:ListBucket",
        "copy_object": "s3:PutObject",
    }
    called = set(re.findall(r"_s3_client\.(\w+)\(", LOCK_MODULE.read_text()))
    assert called, "found no S3 calls in dataset_lock.py - has the client attribute been renamed?"

    unknown = called - method_to_action.keys()
    assert not unknown, (
        f"dataset_lock.py calls S3 methods this test does not know how to map to an "
        f"IAM action: {sorted(unknown)}. Add them to method_to_action."
    )

    granted = _granted_actions()
    missing = {method_to_action[m] for m in called} - granted
    assert not missing, f"dataset_lock.py needs {sorted(missing)}, which the template does not grant"


def test_the_delete_grant_is_scoped_to_locks_and_cannot_reach_snapshots() -> None:
    """A snapshot the optimiser cannot delete survives a bug in the optimiser.

    Broadening this to the bucket would work and would be wrong: the lock is the only
    object the function ever deletes, and rollback depends on the snapshots still being
    there.
    """
    delete_resources: list[str] = []
    for statement in _s3_statements():
        declared = statement.get("Action", [])
        actions = [declared] if isinstance(declared, str) else declared
        if "s3:DeleteObject" not in actions:
            continue
        resource = statement["Resource"]
        sub = resource.get("Fn::Sub") if isinstance(resource, dict) else resource
        delete_resources.append(sub)

    assert delete_resources, "no statement grants s3:DeleteObject"
    for resource in delete_resources:
        assert "/locks/" in resource, (
            f"s3:DeleteObject is granted on {resource!r}, which is broader than the "
            "locks prefix. snapshots/ must stay undeletable so a rollback still has "
            "something to restore."
        )
        assert not resource.rstrip("*").endswith("/Arn}/"), (
            f"s3:DeleteObject on {resource!r} covers the whole bucket"
        )


def test_delete_object_is_called_in_exactly_one_place() -> None:
    """Guards the premise of the scoping test above.

    If a second delete appears somewhere else in lambda_source, the locks-only grant
    silently becomes too narrow and that call starts failing the same quiet way.
    """
    callers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "lambda_source").rglob("*.py")
        if "delete_object" in path.read_text()
    )
    assert callers == ["lambda_source/dataset_lock.py"], (
        f"delete_object is now called from {callers}. The IAM grant is scoped to "
        "the locks prefix, so any other delete will be denied - widen the grant "
        "deliberately or route the delete through the lock module."
    )


# ── lifecycle rules ──────────────────────────────────────────────────────────


def test_lifecycle_prefixes_carry_the_key_prefix_the_function_writes_under() -> None:
    """Every key is written under ``${AWS::StackName}/``, so the rules must be too.

    An S3 lifecycle Prefix is a literal string match, not a glob. 'snapshots/' never
    matched 'spice-optimiser/snapshots/', so neither rule expired anything for the
    lifetime of the stack. Confirmed against the live bucket before fixing: zero objects
    at the top-level snapshots/ or reports/.
    """
    template = _template()
    env = template["Resources"]["OptimiserFunction"]["Properties"]["Environment"][
        "Variables"
    ]
    assert env["S3_PREFIX"] == {"Fn::Sub": "${AWS::StackName}/"}, (
        "S3_PREFIX changed; the lifecycle prefixes below are derived from it and need "
        "to change with it"
    )

    rules = template["Resources"]["OptimiserBucket"]["Properties"][
        "LifecycleConfiguration"
    ]["Rules"]
    assert rules, "no lifecycle rules found"
    for rule in rules:
        prefix = rule["Prefix"]
        sub = prefix.get("Fn::Sub") if isinstance(prefix, dict) else prefix
        assert sub.startswith("${AWS::StackName}/"), (
            f"lifecycle rule {rule['Id']} has Prefix {sub!r}, which matches no key the "
            "function writes. Prefixes are literal, so it must include the stack-name "
            "prefix from S3_PREFIX."
        )
