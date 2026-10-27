# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The agent-facing config must name the tools the Lambda actually serves.

Three files describe the same tool contract to three different audiences:

    tool-schema.json          -> the gateway, machine-read at deploy time
    quick_agent.json          -> the chat agent's persona and system prompt
    agentcore_gateway.json    -> a human setting the integration up

Only the first was ever checked. The other two drifted to a set of five names -
``list_datasets``, ``analyze_datasets``, ``apply_optimization``,
``check_job_status``, ``get_report`` - that came from a deleted ``mcp_server/tools/*``
layer and match nothing in the dispatcher. A chat agent told to call
``analyze_datasets`` does not fail loudly; it calls a tool that is not there and then
improvises, which is worse than an error.

This is the same drift that ``03acfea`` fixed in the gateway schema. It only fixed the
schema, because the schema was the only one with a test. These assertions are that
test for the other two.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lambda_source"))

from mcp_dispatcher import KNOWN_TOOLS  # noqa: E402

QUICK_AGENT = ROOT / "configuration" / "quick_agent.json"
GATEWAY_REF = ROOT / "configuration" / "agentcore_gateway.json"
SCHEMA = ROOT / "agentcore_project" / "spiceoptimisermcp" / "tool-schema.json"
README = ROOT / "README.md"

# Names from the deleted mcp_server/tools/* layer. They are not merely absent from
# KNOWN_TOOLS - they are actively misleading, because they read like plausible tool
# names and previously appeared in files an operator would copy from.
DEAD_NAMES = frozenset(
    {
        "list_datasets",
        "analyze_datasets",
        "apply_optimization",
        "check_job_status",
        "get_report",
    }
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_dead_names_are_not_live_tools() -> None:
    """Guards the premise of this module: these five really are gone.

    If someone reintroduces a tool by one of these names the rest of the assertions
    here become wrong rather than merely failing, so check it first.
    """
    assert DEAD_NAMES.isdisjoint(KNOWN_TOOLS)


def test_quick_agent_tool_integration_lists_exactly_the_live_tools() -> None:
    declared = {
        tool
        for integration in _load(QUICK_AGENT)["toolIntegrations"]
        for tool in integration["tools"]
    }
    assert declared == set(KNOWN_TOOLS), (
        "quick_agent.json toolIntegrations disagrees with KNOWN_TOOLS. "
        f"only in config: {sorted(declared - KNOWN_TOOLS)}; "
        f"only in code: {sorted(KNOWN_TOOLS - declared)}"
    )


def test_gateway_reference_lists_exactly_the_live_tools() -> None:
    declared = {
        tool for target in _load(GATEWAY_REF)["targets"] for tool in target["tools"]
    }
    assert declared == set(KNOWN_TOOLS), (
        "agentcore_gateway.json disagrees with KNOWN_TOOLS. "
        f"only in config: {sorted(declared - KNOWN_TOOLS)}; "
        f"only in code: {sorted(KNOWN_TOOLS - declared)}"
    )


def test_the_gateway_reference_points_at_the_real_schema_path() -> None:
    """The path the CDK resolves, not the decoy that used to sit under agentcore/."""
    declared = _load(GATEWAY_REF)["targets"][0]["toolSchema"]
    assert (ROOT / declared).is_file(), f"{declared} does not exist"
    assert (ROOT / declared).resolve() == SCHEMA.resolve()


@pytest.mark.parametrize("path", [QUICK_AGENT, GATEWAY_REF], ids=lambda p: p.name)
def test_no_dead_tool_name_appears_anywhere_in_the_file(path: Path) -> None:
    """Catches dead names in prose, not just in the tool arrays.

    The system prompt and the example interactions are what the agent is actually
    steered by. A correct ``toolIntegrations`` array alongside a system prompt that
    says "use analyze_datasets" is still a broken agent.
    """
    text = path.read_text()
    found = {name for name in DEAD_NAMES if re.search(rf"\b{name}\b", text)}
    assert not found, f"{path.name} still refers to tools that do not exist: {sorted(found)}"


def test_every_live_tool_is_named_in_the_system_prompt() -> None:
    """A new tool has to be documented to the agent, not just wired to the gateway.

    Otherwise it ships invisible: advertised by the gateway, never called, because
    nothing told the agent it exists or when to use it.
    """
    prompt = _load(QUICK_AGENT)["systemPrompt"]
    missing = {tool for tool in KNOWN_TOOLS if tool not in prompt}
    assert not missing, f"systemPrompt does not mention: {sorted(missing)}"


def test_the_write_tools_are_documented_with_their_confirmation_token() -> None:
    """Both destructive tools refuse without an explicit action value.

    The agent cannot discover that by trying - it gets an error and no guidance. If
    the prompt omits the token the agent will call the tool, be refused, and have no
    way to proceed.
    """
    prompt = _load(QUICK_AGENT)["systemPrompt"]
    for tool, token in (
        ("remove_unused_columns", "action='remove'"),
        ("rollback_dataset", "action='rollback'"),
    ):
        assert token in prompt, f"systemPrompt documents {tool} without {token}"


def test_the_schema_and_the_persona_describe_the_same_tools() -> None:
    """Closes the loop: gateway, agent, and code all agree.

    Each pair is checked above against KNOWN_TOOLS; this asserts the two config files
    agree with the deployed schema directly, so a failure names the two files that
    disagree rather than sending the reader via the code.
    """
    schema_names = {tool["name"] for tool in _load(SCHEMA)}
    persona_names = {
        tool
        for integration in _load(QUICK_AGENT)["toolIntegrations"]
        for tool in integration["tools"]
    }
    assert schema_names == persona_names


# ── README ───────────────────────────────────────────────────────────────────
#
# The README was a fourth copy of the tool contract, and it had already gone wrong in
# the most confusing way available: its undo workflow instructed the agent to call
# rollback_dataset, while the "Available tools" list four lines below named only four
# tools and omitted both rollback_dataset and list_snapshots. Pasting it produced an
# agent told to use a tool it had also been told did not exist.
#
# The fix was to stop keeping a copy and point at configuration/quick_agent.json, which
# these tests already guard. These assertions keep it that way, because the natural
# instinct on reading a README that references another file is to inline it again.


def _readme_section(heading: str) -> str:
    """Return the body of a README section, up to the next heading of any level.

    Scoped rather than whole-file because the README has several backticked tables -
    config keys, environment variables - and a whole-file scan reads those as tool
    names.
    """
    text = README.read_text()
    start = text.index(heading) + len(heading)
    rest = text[start:]
    end = re.search(r"^#{1,6}\s", rest, re.MULTILINE)
    return rest[: end.start()] if end else rest


def test_the_readme_tools_table_lists_exactly_the_live_tools() -> None:
    """The one place the README still enumerates tools, so it still has to be right."""
    section = _readme_section("## Tools (exposed via MCP)")
    rows = re.findall(r"^\|\s*`([a-z_]+)`\s*\|", section, re.MULTILINE)
    assert set(rows) == set(KNOWN_TOOLS), (
        "README tools table disagrees with KNOWN_TOOLS. "
        f"only in README: {sorted(set(rows) - KNOWN_TOOLS)}; "
        f"only in code: {sorted(KNOWN_TOOLS - set(rows))}"
    )


def test_the_readme_does_not_reintroduce_an_inline_system_prompt() -> None:
    """A second copy of the prompt is what drifted. Point at the JSON instead.

    Keyed on the prompt's opening line, which is what a paste of the real prompt would
    necessarily contain, rather than on anything incidental.
    """
    text = README.read_text()
    assert "You are the SPICE Optimization Assistant" not in text, (
        "README appears to inline the system prompt again. It drifted last time - "
        "reference configuration/quick_agent.json, which the tests above guard."
    )
    assert "configuration/quick_agent.json" in text, (
        "README must tell the reader where the system prompt actually lives"
    )


def test_the_readme_names_no_dead_tool() -> None:
    found = {name for name in DEAD_NAMES if re.search(rf"\b{name}\b", README.read_text())}
    assert not found, f"README refers to tools that do not exist: {sorted(found)}"


def test_the_readme_does_not_claim_a_stale_tool_count() -> None:
    """"4 tools" outlived two tools being added, in three separate places.

    A count in prose is a fact that ages badly and that nothing else contradicts, so it
    survives review. Assert on the number rather than on the phrasing.
    """
    expected = len(KNOWN_TOOLS)
    stale = re.findall(r"\b(\d+)\s+(?:discovered\s+)?tools\b", README.read_text())
    wrong = sorted({n for n in stale if int(n) != expected})
    assert not wrong, (
        f"README claims {wrong} tools somewhere; there are {expected}. "
        "Check the connector setup step and the project structure tree."
    )
