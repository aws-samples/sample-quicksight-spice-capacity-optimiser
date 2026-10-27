# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The published repository must not carry account-specific or internal identifiers.

The 2026-10 security review found a real AWS account id in 21 files, a live Cognito user
pool id and gateway id in the AgentCore configuration, internal repository URLs and a
builder alias. Those were removed by hand once; this test keeps them out. It scans every
tracked file except the material that is deliberately internal-only (the engineering
backlog and the security-review artefacts, which live on the internal development branch
and are gitignored on the published one).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Documentation placeholders AWS uses in examples; anything else that looks like an
# account id is treated as real.
PLACEHOLDER_ACCOUNTS = {"123456789012", "111122223333", "111111111111", "000000000000"}

# Paths that are internal-only by design and may legitimately name the development
# account (they are gitignored on the published branch; see .gitignore).
INTERNAL_PREFIXES = ("docs/backlog.md", "docs/dsr-", ".agents/", "agentcore_project/spiceoptimisermcp/agentcore/.llm-context/")

# Lockfiles and SBOMs carry package hashes that can look like anything.
SKIP_SUFFIXES = (".lock", "package-lock.json", ".cdx.json", ".xlsx", ".png", ".zip")

INTERNAL_DOMAINS = re.compile(r"\b(code\.aws\.dev|w\.amazon\.com|t\.corp\.amazon\.com|code\.amazon\.com|tiny\.amazon\.com)\b")
# The AgentCore CLI's public JSON schema is the one aws.dev host that belongs in the repo.
ALLOWED_AWS_DEV = {"schema.agentcore.aws.dev"}


def _tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    files = []
    for rel in out.split("\n"):
        if not rel or rel.startswith(INTERNAL_PREFIXES) or rel.endswith(SKIP_SUFFIXES):
            continue
        p = ROOT / rel
        if p.is_file():
            files.append(p)
    return files


def _texts():
    for p in _tracked_files():
        try:
            yield p.relative_to(ROOT), p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


class TestNoAccountSpecificIdentifiers:
    def test_no_real_aws_account_id(self):
        offenders = {}
        for rel, text in _texts():
            found = set(re.findall(r"(?<![0-9A-Za-z])[0-9]{12}(?![0-9A-Za-z])", text)) - PLACEHOLDER_ACCOUNTS
            if found:
                offenders[str(rel)] = sorted(found)
        assert offenders == {}, f"12-digit numbers that are not documentation placeholders: {offenders}"

    def test_no_live_cognito_pool_id(self):
        offenders = {}
        for rel, text in _texts():
            found = {m for m in re.findall(r"\b[a-z]{2}-[a-z]+-\d_[A-Za-z0-9]{9}\b", text) if not m.endswith("_EXAMPLE")}
            if found:
                offenders[str(rel)] = sorted(found)
        assert offenders == {}, f"Cognito user pool ids that are not the EXAMPLE placeholder: {offenders}"

    def test_no_live_gateway_or_client_identifiers_in_agentcore_config(self):
        spec = (ROOT / "agentcore_project/spiceoptimisermcp/agentcore/agentcore.json").read_text()
        policy = (ROOT / "agentcore-policy.json").read_text()
        assert "us-east-1_EXAMPLE" in spec
        assert '"allowedClients": []' in spec
        assert "123456789012" in spec and "123456789012" in policy


class TestNoInternalReferences:
    def test_no_internal_amazon_domains(self):
        offenders = {}
        for rel, text in _texts():
            found = set(INTERNAL_DOMAINS.findall(text))
            found |= {m for m in re.findall(r"\b[a-z0-9.-]+\.aws\.dev\b", text) if m not in ALLOWED_AWS_DEV}
            if found:
                offenders[str(rel)] = sorted(found)
        assert offenders == {}, f"internal domains in published files: {offenders}"

    @pytest.mark.parametrize("rel", ["README.md", "docs/wiki.md", "configuration/config.json", "configuration/quick_agent.json"])
    def test_bucket_names_in_documentation_are_placeholders(self, rel):
        text = (ROOT / rel).read_text()
        literal = [m for m in re.findall(r"s3://([^\s'\"`)]+)", text) if not m.startswith("<")]
        assert literal == [], f"{rel} names a bucket a reader might copy and not own: {literal}"


class TestGeneratedStateIsNotTracked:
    """These files name this account's deployment and are produced at deploy time."""

    def test_agentcore_cli_state_and_targets_are_not_tracked(self):
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
        assert "agentcore_project/spiceoptimisermcp/agentcore/.cli/deployed-state.json" not in tracked
        assert "agentcore_project/spiceoptimisermcp/agentcore/aws-targets.json" not in tracked
        assert "agentcore_project/spiceoptimisermcp/agentcore/aws-targets.example.json" in tracked
