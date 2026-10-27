# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for UsageAttribution — per-column provenance collection.

The pipeline decides what to remove from one flat union of column references.
That union is correct for the decision and useless for the explanation, which is
what these tests pin down: attribution must record who required each column and
why, in a stable presentation order, without ever being able to change the
decision itself.

The distinction that matters most here is proven versus unproven protection. A
column retained because a dashboard displays it and a column retained because a
calculated field defeated the parser were previously indistinguishable in the
report. They call for opposite responses.
"""

from __future__ import annotations

from lambda_source.attribution import UsageAttribution
from lambda_source.models import ColumnAttribution, ColumnUsageSource


class TestRecording:
    def test_records_a_single_column(self):
        attribution = UsageAttribution()
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Sales KPIs",
            requester_id="arn:aws:quicksight:::dashboard/abc",
        )

        assert attribution.columns() == {"region"}
        recorded = attribution.for_column("region")
        assert len(recorded) == 1
        assert recorded[0].source == ColumnUsageSource.DIRECT_CONSUMER.value
        assert recorded[0].requester_name == "Sales KPIs"

    def test_record_many_shares_one_requester(self):
        attribution = UsageAttribution()
        attribution.record_many(
            {"a", "b", "c"},
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Dash",
        )

        assert attribution.columns() == {"a", "b", "c"}
        for column in ("a", "b", "c"):
            assert attribution.for_column(column)[0].requester_name == "Dash"

    def test_blank_column_names_are_ignored(self):
        """QuickSight definitions carry empty Name fields often enough that
        raising here would fail a run over a cosmetic quirk."""
        attribution = UsageAttribution()
        attribution.record("", ColumnUsageSource.DIRECT_CONSUMER.value)
        attribution.record_many({"", "real"}, ColumnUsageSource.DIRECT_CONSUMER.value)

        assert attribution.columns() == {"real"}

    def test_duplicate_attributions_collapse(self):
        """A diamond dependency reaches the same child twice. The report should
        not say so twice."""
        attribution = UsageAttribution()
        for _ in range(3):
            attribution.record(
                "customer_id",
                ColumnUsageSource.CHILD_JOIN_KEY.value,
                requester_name="joined_orders",
                requester_id="child-1",
                detail="depth 1",
            )

        assert len(attribution.for_column("customer_id")) == 1

    def test_same_column_from_different_requesters_is_kept_apart(self):
        attribution = UsageAttribution()
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Dash A",
        )
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Dash B",
        )

        names = {a.requester_name for a in attribution.for_column("region")}
        assert names == {"Dash A", "Dash B"}

    def test_unknown_column_has_no_attributions(self):
        attribution = UsageAttribution()
        assert attribution.for_column("absent") == []
        assert attribution.reasons_for("absent") == []
        assert attribution.requesters_for("absent") == []


class TestPresentationOrder:
    def test_most_decisive_source_leads(self):
        """A column held by both a direct consumer and a vague structural import
        should lead with the consumer."""
        attribution = UsageAttribution()
        attribution.record(
            "region",
            ColumnUsageSource.CHILD_STRUCTURAL.value,
            requester_name="child",
        )
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Dash",
        )

        ordered = [a.source for a in attribution.for_column("region")]
        assert ordered[0] == ColumnUsageSource.DIRECT_CONSUMER.value

    def test_own_join_key_outranks_everything(self):
        attribution = UsageAttribution()
        attribution.record("id", ColumnUsageSource.DIRECT_CONSUMER.value)
        attribution.record("id", ColumnUsageSource.CHILD_CONSUMER.value)
        attribution.record("id", ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value)

        first = attribution.for_column("id")[0]
        assert first.source == ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value

    def test_fallbacks_read_last(self):
        """A fallback describes a gap in the analysis, not a dependency, so it
        should never be the headline reason."""
        attribution = UsageAttribution()
        attribution.record("region", ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value)
        attribution.record("region", ColumnUsageSource.CHILD_PROJECTION.value)
        attribution.record("region", ColumnUsageSource.DIRECT_CONSUMER.value)

        ordered = [a.source for a in attribution.for_column("region")]
        assert ordered[-1] == ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value

    def test_order_is_stable_across_identical_inputs(self):
        """Unstable ordering in a report humans diff is a defect in itself."""
        def build() -> list[str]:
            attribution = UsageAttribution()
            for name in ("Zeta", "Alpha", "Mid"):
                attribution.record(
                    "region",
                    ColumnUsageSource.DIRECT_CONSUMER.value,
                    requester_name=name,
                )
            return [a.requester_name for a in attribution.for_column("region")]

        assert build() == build() == ["Alpha", "Mid", "Zeta"]


class TestReasonRendering:
    def test_reason_names_the_requester(self):
        attribution = UsageAttribution()
        attribution.record(
            "segment",
            ColumnUsageSource.CHILD_JOIN_KEY.value,
            requester_name="joined_orders",
            detail="depth 2",
        )

        reasons = attribution.reasons_for("segment")
        assert reasons == ["join key in child dataset (joined_orders, depth 2)"]

    def test_bare_source_renders_without_parentheses(self):
        attribution = UsageAttribution()
        attribution.record("segment", ColumnUsageSource.DIRECT_CONSUMER.value)

        assert attribution.reasons_for("segment") == ["referenced by consumer"]

    def test_child_and_direct_consumers_read_differently(self):
        """The whole point: these two used to render identically."""
        attribution = UsageAttribution()
        attribution.record(
            "a", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="Dash"
        )
        attribution.record(
            "b", ColumnUsageSource.CHILD_CONSUMER.value, requester_name="Child"
        )

        assert attribution.reasons_for("a") != attribution.reasons_for("b")

    def test_join_key_provenance_distinguishes_the_two_experiences(self):
        legacy = ColumnAttribution(
            source=ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value
        )
        new = ColumnAttribution(source=ColumnUsageSource.JOIN_KEY_DATA_PREP.value)

        assert "LogicalTableMap" in legacy.describe()
        assert "DataPrepConfiguration" in new.describe()
        assert legacy.describe() != new.describe()

    def test_duplicate_rendered_reasons_are_deduplicated(self):
        attribution = UsageAttribution()
        attribution.record(
            "region", ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value
        )
        attribution.record(
            "region", ColumnUsageSource.JOIN_KEY_DATA_PREP.value
        )
        attribution.record("region", ColumnUsageSource.DIRECT_CONSUMER.value)

        reasons = attribution.reasons_for("region")
        assert len(reasons) == len(set(reasons))


class TestStructuredOutput:
    def test_requesters_carry_the_full_shape(self):
        attribution = UsageAttribution()
        attribution.record(
            "region",
            ColumnUsageSource.CHILD_PROJECTION.value,
            requester_name="agg",
            requester_id="child-9",
            detail="depth 3",
        )

        assert attribution.requesters_for("region") == [
            {
                "source": ColumnUsageSource.CHILD_PROJECTION.value,
                "name": "agg",
                "id": "child-9",
                "detail": "depth 3",
            }
        ]


class TestUnprovenProtection:
    def test_fallback_only_column_is_flagged(self):
        attribution = UsageAttribution()
        attribution.record("region", ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value)

        assert attribution.is_only_fallback("region") is True

    def test_column_with_real_evidence_is_not_flagged(self):
        attribution = UsageAttribution()
        attribution.record("region", ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value)
        attribution.record("region", ColumnUsageSource.DIRECT_CONSUMER.value)

        assert attribution.is_only_fallback("region") is False

    def test_structural_import_alone_counts_as_unproven(self):
        """An imported-but-untraced column is protected without proof that
        anything downstream displays it."""
        attribution = UsageAttribution()
        attribution.record("region", ColumnUsageSource.CHILD_STRUCTURAL.value)

        assert attribution.is_only_fallback("region") is True

    def test_unknown_column_is_not_flagged(self):
        assert UsageAttribution().is_only_fallback("absent") is False

    def test_has_source_detects_a_fallback_anywhere(self):
        attribution = UsageAttribution()
        attribution.record("a", ColumnUsageSource.DIRECT_CONSUMER.value)
        attribution.record("b", ColumnUsageSource.CHILD_REFS_UNKNOWN.value)

        assert attribution.has_source(ColumnUsageSource.CHILD_REFS_UNKNOWN.value)
        assert not attribution.has_source(
            ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value
        )

    def test_columns_with_source_filters(self):
        attribution = UsageAttribution()
        attribution.record("a", ColumnUsageSource.DIRECT_CONSUMER.value)
        attribution.record("b", ColumnUsageSource.CHILD_JOIN_KEY.value)
        attribution.record("c", ColumnUsageSource.CHILD_JOIN_KEY.value)

        assert attribution.columns_with_source(
            ColumnUsageSource.CHILD_JOIN_KEY.value
        ) == {"b", "c"}

    def test_sources_for_returns_the_distinct_set(self):
        attribution = UsageAttribution()
        attribution.record(
            "a", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="X"
        )
        attribution.record(
            "a", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="Y"
        )
        attribution.record("a", ColumnUsageSource.CHILD_CONSUMER.value)

        assert attribution.sources_for("a") == {
            ColumnUsageSource.DIRECT_CONSUMER.value,
            ColumnUsageSource.CHILD_CONSUMER.value,
        }


class TestContainerProtocol:
    def test_empty_collector_is_falsey(self):
        assert not UsageAttribution()
        assert len(UsageAttribution()) == 0

    def test_populated_collector_counts_columns_not_attributions(self):
        attribution = UsageAttribution()
        attribution.record(
            "a", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="X"
        )
        attribution.record(
            "a", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="Y"
        )

        assert bool(attribution) is True
        assert len(attribution) == 1
