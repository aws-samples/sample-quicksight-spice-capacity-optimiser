# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Per-column provenance tracking for the SPICE Dataset Optimiser.

The pipeline decides what to remove by unioning every column reference it can
find into one ``set[str]``. That is the right shape for the decision — a column
is protected if *anything* needs it — but it is the wrong shape for explaining
the decision, and the union is where all the useful detail was being lost.

``UsageAttribution`` records the same references a second time, keyed by column,
retaining who required each one and why. It deliberately has no influence on
classification: the caller keeps building its set exactly as before, and passes
this alongside. That separation is the point — a reporting change must not be
able to alter a removal decision.

The distinction being preserved matters for remediation, which is why the
backlog asked for it:

- A column held by a **direct consumer** means someone is looking at it.
- A column held by a **child's join key** cannot be removed without
  restructuring the child.
- A column held by a **child's ProjectStep** often *can* be removed, once the
  child stops projecting it — actionable, and previously indistinguishable
  from the other two.
- A column retained by a **fallback** means the analysis was incomplete. That
  is not the same as "this column is needed", and reading it as such is how a
  permissions error turns into a deleted column.
"""

from __future__ import annotations

from collections import defaultdict

from lambda_source.models import ColumnAttribution, ColumnUsageSource

# Order attributions are presented in: most decisive first, weakest last. A
# column protected by both a direct consumer and a vague structural import
# should lead with the consumer, and a fallback should always read last because
# it describes a gap in the analysis rather than a genuine dependency.
_SOURCE_PRECEDENCE: dict[str, int] = {
    ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value: 0,
    ColumnUsageSource.JOIN_KEY_DATA_PREP.value: 0,
    ColumnUsageSource.DIRECT_CONSUMER.value: 1,
    ColumnUsageSource.CHILD_JOIN_KEY.value: 2,
    ColumnUsageSource.CHILD_CONSUMER.value: 3,
    ColumnUsageSource.CHILD_PROJECTION.value: 4,
    ColumnUsageSource.CHILD_STRUCTURAL.value: 5,
    ColumnUsageSource.DATASET_CALC_FIELD.value: 6,
    ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value: 90,
    ColumnUsageSource.CHILD_REFS_UNKNOWN.value: 91,
    ColumnUsageSource.UNREACHABLE_CONSUMER.value: 92,
}

_UNKNOWN_PRECEDENCE = 50

# Sources that mean "we could not complete the analysis" rather than "something
# demonstrably needs this column".
_FALLBACK_SOURCES = frozenset(
    {
        ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value,
        ColumnUsageSource.CHILD_REFS_UNKNOWN.value,
        ColumnUsageSource.CHILD_STRUCTURAL.value,
        ColumnUsageSource.UNREACHABLE_CONSUMER.value,
    }
)


class UsageAttribution:
    """Collects why each column is protected, without affecting whether it is.

    Attributions are deduplicated: a diamond dependency reaches the same child
    twice, and the report should not say so twice.
    """

    def __init__(self) -> None:
        self._by_column: dict[str, set[ColumnAttribution]] = defaultdict(set)

    # -- recording ---------------------------------------------------------

    def record(
        self,
        column: str,
        source: str,
        requester_name: str = "",
        requester_id: str = "",
        detail: str = "",
    ) -> None:
        """Attribute one column to one requester.

        Blank column names are ignored rather than rejected. QuickSight
        definitions contain empty ``Name`` fields often enough that raising here
        would turn a cosmetic data quirk into a failed optimisation run.
        """
        if not column:
            return

        self._by_column[column].add(
            ColumnAttribution(
                source=source,
                requester_name=requester_name or "",
                requester_id=requester_id or "",
                detail=detail or "",
            )
        )

    def record_many(
        self,
        columns: set[str] | list[str],
        source: str,
        requester_name: str = "",
        requester_id: str = "",
        detail: str = "",
    ) -> None:
        """Attribute several columns to the same requester."""
        for column in columns or ():
            self.record(
                column,
                source,
                requester_name=requester_name,
                requester_id=requester_id,
                detail=detail,
            )

    # -- querying ----------------------------------------------------------

    def columns(self) -> set[str]:
        """Every column with at least one attribution."""
        return {col for col, attrs in self._by_column.items() if attrs}

    def for_column(self, column: str) -> list[ColumnAttribution]:
        """Attributions for one column, most decisive first.

        Sorted by source precedence, then by requester name, so report output is
        stable across runs. Unstable ordering in a report that humans diff is a
        bug in its own right.
        """
        return sorted(
            self._by_column.get(column, set()),
            key=lambda a: (
                _SOURCE_PRECEDENCE.get(a.source, _UNKNOWN_PRECEDENCE),
                a.requester_name,
                a.detail,
                a.requester_id,
            ),
        )

    def reasons_for(self, column: str) -> list[str]:
        """Human-readable reason strings for one column, in presentation order."""
        seen: set[str] = set()
        reasons: list[str] = []
        for attribution in self.for_column(column):
            described = attribution.describe()
            if described not in seen:
                seen.add(described)
                reasons.append(described)
        return reasons

    def requesters_for(self, column: str) -> list[dict[str, str]]:
        """Structured provenance for one column, for the report payload."""
        return [
            {
                "source": a.source,
                "name": a.requester_name,
                "id": a.requester_id,
                "detail": a.detail,
            }
            for a in self.for_column(column)
        ]

    def sources_for(self, column: str) -> set[str]:
        """The distinct usage sources recorded for one column."""
        return {a.source for a in self._by_column.get(column, set())}

    def has_source(self, source: str) -> bool:
        """Whether any column was attributed to the given source.

        Drives risk indicators: if a fallback fired anywhere, the report should
        say so rather than presenting the result as a clean analysis.
        """
        return any(
            a.source == source for attrs in self._by_column.values() for a in attrs
        )

    def columns_with_source(self, source: str) -> set[str]:
        """Every column attributed to the given source."""
        return {
            col
            for col, attrs in self._by_column.items()
            if any(a.source == source for a in attrs)
        }

    def is_only_fallback(self, column: str) -> bool:
        """True when a column is retained solely because analysis was incomplete.

        These are the columns worth a human's attention: nothing demonstrably
        needs them, but the optimiser could not prove that either.
        """
        sources = self.sources_for(column)
        if not sources:
            return False
        return sources <= _FALLBACK_SOURCES

    def __len__(self) -> int:
        return len(self.columns())

    def __bool__(self) -> bool:
        return bool(self.columns())
