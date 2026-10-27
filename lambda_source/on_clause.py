# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared parsing for QuickSight join ``OnClause`` strings.

One module so there is a single place that understands the format. Previously
``UsageAnalyzer._parse_on_clause`` and ``ChildDatasetDiscovery._strip_on_clause_braces``
each had their own idea of it, and they disagreed: the child discovery path stripped
braces, the analyzer path did not. That asymmetry was a live defect — see the
``## Bugs`` section of ``docs/backlog.md``.

Formats QuickSight actually emits, all observed in a real QuickSight account:

    id = {inventory_id}                 mixed bare and braced
    {part_num} = {part_num[parts]}      braced, one side table-qualified
    {customer_id} = {customer_id}       braced both sides
    col_a = col_b AND col_c = col_d     multi-condition

The table-qualified form needs care. ``{part_num[parts]}`` reads as "column
``part_num`` from table ``parts``", but QuickSight also materialises a column
*literally named* ``part_num[parts]`` in joined output — both exist as real columns on
``inventories.csv``. So both readings are returned and the caller intersects with the
dataset's actual columns. A false positive costs one retained column; a false negative
removes a join key and breaks the join.
"""

from __future__ import annotations

import re

# Multi-condition joins: "a = b AND c = d".
_CONDITION_SPLIT = re.compile(r"\s+AND\s+", re.IGNORECASE)

# The table-qualified form, "column[table]".
_TABLE_QUALIFIED = re.compile(r"^([^\[\]]+)\[([^\[\]]+)\]$")


def strip_braces(reference: str) -> str:
    """Reduce one column reference to its bare column name.

    ``{part_num[parts]}`` becomes ``part_num``. Use :func:`candidate_names` instead
    when the caller can tolerate more than one answer, because the table-qualified
    form is genuinely ambiguous.
    """
    raw = (reference or "").strip()
    if not raw:
        return raw

    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]

    bracket = raw.find("[")
    if bracket > 0:
        raw = raw[:bracket]

    return raw.strip()


def candidate_names(reference: str) -> set[str]:
    """Every plausible column name for one column reference.

    >>> sorted(candidate_names("id"))
    ['id']
    >>> sorted(candidate_names("{inventory_id}"))
    ['inventory_id']
    >>> sorted(candidate_names("{part_num[parts]}"))
    ['part_num', 'part_num[parts]']
    """
    raw = (reference or "").strip()
    if not raw:
        return set()

    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1].strip()
    if not raw:
        return set()

    names = {raw}

    match = _TABLE_QUALIFIED.match(raw)
    if match:
        bare = match.group(1).strip()
        if bare:
            names.add(bare)

    return names


def parse_on_clause(on_clause: str) -> set[str]:
    """Extract every plausible column name from a join ``OnClause``.

    Args:
        on_clause: The raw ``OnClause`` string.

    Returns:
        Candidate column names from both sides of every condition.
    """
    names: set[str] = set()
    for condition in _CONDITION_SPLIT.split(on_clause or ""):
        for side in condition.split("="):
            names |= candidate_names(side)
    return names


def join_keys_from_logical_table_map(logical_table_map: dict | None) -> set[str]:
    """Join key columns declared in a legacy ``LogicalTableMap``."""
    names: set[str] = set()

    for table in (logical_table_map or {}).values():
        if not isinstance(table, dict):
            continue
        source = table.get("Source")
        if not isinstance(source, dict):
            continue
        join = source.get("JoinInstruction")
        if not isinstance(join, dict):
            continue

        names |= parse_on_clause(join.get("OnClause", ""))

        for side in ("LeftJoinKeyProperties", "RightJoinKeyProperties"):
            props = join.get(side)
            if isinstance(props, dict):
                for column in props.get("Columns") or []:
                    if isinstance(column, str) and column:
                        names |= candidate_names(column)

    return names


def join_keys_from_data_prep(data_prep_config: dict | None) -> set[str]:
    """Join key columns declared in a new-experience ``DataPrepConfiguration``.

    New-experience datasets keep joins in ``TransformStepMap[*].JoinStep``. Nothing
    read this for the *target* dataset previously, so a new-experience dataset that
    projected its join key had no protection at all.
    """
    names: set[str] = set()

    for step in ((data_prep_config or {}).get("TransformStepMap") or {}).values():
        if not isinstance(step, dict):
            continue
        join_step = step.get("JoinStep")
        if isinstance(join_step, dict):
            names |= parse_on_clause(join_step.get("OnClause", ""))

    return names


def join_keys_from_dataset(
    dataset_def: dict, known_columns: set[str] | None = None
) -> set[str]:
    """Every column participating in a join, across both dataset shapes.

    Args:
        dataset_def: A full ``DescribeDataSet`` response.
        known_columns: When supplied, the result is intersected with it so only real
            columns are returned.

    Returns:
        Column names that must not be removed.
    """
    names = join_keys_from_logical_table_map(dataset_def.get("LogicalTableMap"))
    names |= join_keys_from_data_prep(dataset_def.get("DataPrepConfiguration"))

    if known_columns is None:
        return names

    return {n for n in names if n in known_columns}
