# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""A mock standing in for one of our classes must enforce that class's signatures.

``create_ingestion(dataset_id)`` — one argument to a two-argument method — reached
production and broke rollback. The test covering that path used a bare ``MagicMock``,
which accepts any call, and then asserted ``assert_called_once_with("ds-1")``: the broken
call. The suite was holding the defect in place rather than catching it.

A correction worth recording, because the first attempt at this fix was wrong. ``spec=``
does **not** check call signatures — it only restricts which attributes exist. Only
``create_autospec`` binds arguments against the real signature:

    MagicMock().create_ingestion('ds-1')                      accepted
    MagicMock(spec=QSClient).create_ingestion('ds-1')         accepted
    create_autospec(QSClient).create_ingestion('ds-1')        TypeError

So an earlier claim that ``spec=`` would catch signature mismatches was mistaken; what
actually caught the reintroduced bug in that test was the argument-unpacking assertion,
not the spec. This module asserts the distinction directly so the same mistake is not made
again from memory.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from unittest.mock import MagicMock, create_autospec

from lambda_source.qs_client import QSClient

TESTS = pathlib.Path(__file__).resolve().parents[1]

# Variable names that conventionally stand in for one of our collaborators. Matched on
# name because that is what a reader of the test sees, and what the next person will copy.
COLLABORATOR_NAMES = {
    "qs", "qs_client", "modifier", "analyzer", "snapshot_manager", "snapshots",
    "notifier", "reporter", "lock", "discovery", "quarantine", "attribution",
    "orchestrator",
}


class TestTheDistinctionThisModuleRestsOn:
    """Pin the behaviour, so the reasoning above cannot quietly become false."""

    def test_a_bare_magicmock_accepts_a_wrong_arity_call(self):
        MagicMock().create_ingestion("ds-1")  # no error — this is the hazard

    def test_spec_alone_also_accepts_it(self):
        """The correction. ``spec=`` restricts attributes, not signatures."""
        MagicMock(spec=QSClient).create_ingestion("ds-1")  # still no error

    def test_autospec_rejects_it(self):
        with pytest.raises(TypeError, match="ingestion_id"):
            create_autospec(QSClient, instance=True).create_ingestion("ds-1")

    def test_autospec_still_rejects_attributes_that_do_not_exist(self):
        with pytest.raises(AttributeError):
            create_autospec(QSClient, instance=True).method_that_does_not_exist()


def _bare_mock_collaborators(path: pathlib.Path) -> list[tuple[int, str]]:
    """Find ``<collaborator> = MagicMock()`` with no spec, via AST rather than regex.

    An AST walk sees the real assignment targets and call keywords, so it does not trip
    over a name inside a string or a comment.
    """
    tree = ast.parse(path.read_text())
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name not in {"MagicMock", "Mock"}:
            continue
        if any(kw.arg in {"spec", "spec_set"} for kw in node.value.keywords):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in COLLABORATOR_NAMES:
                offenders.append((node.lineno, target.id))
    return offenders


@pytest.mark.parametrize(
    "path",
    sorted(TESTS.rglob("test_*.py")),
    ids=lambda p: p.name,
)
def test_no_collaborator_is_stubbed_with_an_unspecced_mock(path: pathlib.Path) -> None:
    offenders = _bare_mock_collaborators(path)
    assert not offenders, (
        f"{path.name} stubs a collaborator with a bare mock at "
        f"{[f'line {n}: {v}' for n, v in offenders]}. Use "
        "create_autospec(Class, instance=True) so a call the real class would reject "
        "fails here instead of in production."
    )


def test_the_guard_would_actually_catch_a_regression(tmp_path):
    """A guard that has never fired proves nothing about what it forbids."""
    offending = tmp_path / "test_example.py"
    offending.write_text(
        "from unittest.mock import MagicMock\n"
        "def test_something():\n"
        "    qs_client = MagicMock()\n"
        "    assert qs_client\n"
    )
    assert _bare_mock_collaborators(offending) == [(3, "qs_client")]


def test_the_guard_accepts_the_corrected_form(tmp_path):
    acceptable = tmp_path / "test_example.py"
    acceptable.write_text(
        "from unittest.mock import create_autospec\n"
        "from lambda_source.qs_client import QSClient\n"
        "def test_something():\n"
        "    qs_client = create_autospec(QSClient, instance=True)\n"
        "    assert qs_client\n"
    )
    assert _bare_mock_collaborators(acceptable) == []


def test_an_unrelated_bare_mock_is_left_alone(tmp_path):
    """Scoped deliberately. boto3 clients and DynamoDB tables are not ours to autospec.

    Widening this to every MagicMock in the suite would fail on dozens of legitimate
    stand-ins for third-party objects, and a guard that has to be suppressed everywhere
    gets deleted.
    """
    fine = tmp_path / "test_example.py"
    fine.write_text(
        "from unittest.mock import MagicMock\n"
        "def test_something():\n"
        "    s3 = MagicMock()\n"
        "    dynamodb = MagicMock()\n"
        "    assert s3 and dynamodb\n"
    )
    assert _bare_mock_collaborators(fine) == []
