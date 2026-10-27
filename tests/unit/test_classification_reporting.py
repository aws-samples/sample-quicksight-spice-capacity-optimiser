# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for provenance-aware classification and reporting.

Covers the three layers that turn "referenced by consumer" into an actionable
statement: the analyzer splitting join keys by declaring shape, the handler
attributing child-protected columns to the specific child, and the reporter
surfacing both plus the child dataset summary.

The load-bearing test in here is
``TestAttributionDoesNotAlterDecisions::test_status_is_identical_with_and_without_attribution``.
Everything else is presentation; that one asserts presentation cannot leak into
the removal decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lambda_source.analyzer import UsageAnalyzer
from lambda_source.attribution import UsageAttribution
from lambda_source.child_discovery import ChildDataset
from lambda_source.handler import Orchestrator
from lambda_source.models import ColumnStatus, ColumnUsageSource, Consumer
from lambda_source.reporter import ReportGenerator

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

LEGACY_JOIN_DATASET = {
    "LogicalTableMap": {
        "lt-1": {
            "Source": {
                "JoinInstruction": {
                    # The braced form that was silently unprotected before 3785e17.
                    "OnClause": "id = {inventory_id}",
                }
            }
        }
    },
    "OutputColumns": [
        {"Name": "id", "Type": "INTEGER"},
        {"Name": "inventory_id", "Type": "INTEGER"},
    ],
}

NEW_EXPERIENCE_JOIN_DATASET = {
    "DataPrepConfiguration": {
        "TransformStepMap": {
            "step-1": {"JoinStep": {"OnClause": "{customer_id} = {customer_id}"}}
        }
    },
    "OutputColumns": [{"Name": "customer_id", "Type": "STRING"}],
}


@pytest.fixture
def analyzer() -> UsageAnalyzer:
    return UsageAnalyzer()


@pytest.fixture
def reporter() -> ReportGenerator:
    return ReportGenerator(
        s3_bucket="test-bucket",
        s3_prefix="test-prefix/",
        s3_client=MagicMock(),
    )


def build_report(reporter: ReportGenerator, **overrides) -> dict:
    """Call generate_report with sane defaults, overriding what a test cares about."""
    kwargs = {
        "dataset_id": "ds-1",
        "dataset_name": "orders",
        "execution_mode": "dry_run",
        "classifications": [],
        "consumer_count": 1,
        "safety_threshold_pct": 0.7,
        "threshold_breached": False,
        "consumers": [],
    }
    kwargs.update(overrides)
    return reporter.generate_report(**kwargs)


# --------------------------------------------------------------------------
# Analyzer: join key provenance
# --------------------------------------------------------------------------


class TestJoinKeyProvenance:
    def test_legacy_join_keys_attributed_to_logical_table_map(self, analyzer):
        by_source = analyzer.extract_join_keys_by_source(
            LEGACY_JOIN_DATASET, {"id", "inventory_id"}
        )

        assert by_source[ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value] == {
            "id",
            "inventory_id",
        }
        assert by_source[ColumnUsageSource.JOIN_KEY_DATA_PREP.value] == set()

    def test_new_experience_join_keys_attributed_to_data_prep(self, analyzer):
        """Before the join-key fix these had no protection at all. Reporting them
        as LogicalTableMap keys would now send an operator to the wrong place."""
        by_source = analyzer.extract_join_keys_by_source(
            NEW_EXPERIENCE_JOIN_DATASET, {"customer_id"}
        )

        assert by_source[ColumnUsageSource.JOIN_KEY_DATA_PREP.value] == {"customer_id"}
        assert by_source[ColumnUsageSource.JOIN_KEY_LOGICAL_TABLE_MAP.value] == set()

    @pytest.mark.parametrize(
        "dataset,known",
        [
            (LEGACY_JOIN_DATASET, {"id", "inventory_id"}),
            (NEW_EXPERIENCE_JOIN_DATASET, {"customer_id"}),
            ({}, set()),
        ],
    )
    def test_union_matches_the_undecomposed_extractor(self, analyzer, dataset, known):
        """Splitting by source must not change which columns are protected.
        Intersection distributes over union, so these are equal by construction —
        this pins it so a future edit cannot quietly break it."""
        by_source = analyzer.extract_join_keys_by_source(dataset, known)
        union = set().union(*by_source.values())

        assert union == analyzer.extract_all_join_keys(dataset, known)

    def test_intersection_drops_names_that_are_not_real_columns(self, analyzer):
        """The table-qualified brace form yields two candidate readings; only the
        ones that exist as columns should survive."""
        by_source = analyzer.extract_join_keys_by_source(
            LEGACY_JOIN_DATASET, {"id"}
        )
        union = set().union(*by_source.values())

        assert union == {"id"}


# --------------------------------------------------------------------------
# Analyzer: classification reasons
# --------------------------------------------------------------------------


class TestClassificationReasons:
    ALL_COLUMNS = [
        {"Name": "id", "Type": "INTEGER"},
        {"Name": "region", "Type": "STRING"},
        {"Name": "segment", "Type": "STRING"},
        {"Name": "junk", "Type": "STRING"},
    ]

    def _attribution(self) -> UsageAttribution:
        attribution = UsageAttribution()
        attribution.record(
            "id",
            ColumnUsageSource.JOIN_KEY_DATA_PREP.value,
            requester_name="orders",
        )
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Sales KPIs",
        )
        attribution.record(
            "segment",
            ColumnUsageSource.CHILD_PROJECTION.value,
            requester_name="orders_agg",
            detail="depth 1",
        )
        return attribution

    def test_reasons_name_the_specific_requester(self, analyzer):
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns={"region", "segment"},
            join_key_columns={"id"},
            calc_field_input_columns=set(),
            attribution=self._attribution(),
        )
        by_name = {c.column_name: c for c in result}

        assert "Sales KPIs" in by_name["region"].reasons[0]
        assert "orders_agg" in by_name["segment"].reasons[0]

    def test_child_projection_reads_differently_from_direct_use(self, analyzer):
        """A column held by a child's ProjectStep can often be freed by editing
        the child. One in a dashboard cannot. They must not read alike."""
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns={"region", "segment"},
            join_key_columns={"id"},
            calc_field_input_columns=set(),
            attribution=self._attribution(),
        )
        by_name = {c.column_name: c for c in result}

        assert by_name["region"].reasons != by_name["segment"].reasons

    def test_new_experience_join_key_no_longer_claims_logical_table_map(
        self, analyzer
    ):
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns=set(),
            join_key_columns={"id"},
            calc_field_input_columns=set(),
            attribution=self._attribution(),
        )
        reasons = next(c.reasons for c in result if c.column_name == "id")

        assert "DataPrepConfiguration" in reasons[0]
        assert "LogicalTableMap" not in " ".join(reasons)

    def test_structured_attributions_are_carried(self, analyzer):
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns={"region", "segment"},
            join_key_columns={"id"},
            calc_field_input_columns=set(),
            attribution=self._attribution(),
        )
        by_name = {c.column_name: c for c in result}

        assert by_name["segment"].attributions[0].requester_id == ""
        assert (
            by_name["segment"].attributions[0].source
            == ColumnUsageSource.CHILD_PROJECTION.value
        )

    def test_unused_column_keeps_the_legacy_reason(self, analyzer):
        """An unused column has no requesters, so there is nothing richer to say —
        and tests elsewhere depend on this exact string."""
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns={"region", "segment"},
            join_key_columns={"id"},
            calc_field_input_columns=set(),
            attribution=self._attribution(),
        )
        junk = next(c for c in result if c.column_name == "junk")

        assert junk.reasons == ["no consumer references"]
        assert junk.attributions == []

    def test_without_attribution_the_legacy_strings_are_unchanged(self, analyzer):
        """Callers that do not collect attribution must see the old behaviour."""
        result = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns={"region"},
            join_key_columns={"id"},
            calc_field_input_columns=set(),
        )
        by_name = {c.column_name: c for c in result}

        assert by_name["id"].reasons == [
            "join key in LogicalTableMap JoinInstruction"
        ]
        assert by_name["region"].reasons == ["referenced by consumer"]
        assert by_name["junk"].reasons == ["no consumer references"]
        assert all(c.attributions == [] for c in result)


class TestAttributionDoesNotAlterDecisions:
    ALL_COLUMNS = [
        {"Name": f"col_{i}", "Type": "STRING"} for i in range(8)
    ]

    @pytest.mark.parametrize(
        "used,join_keys,calc_inputs",
        [
            (set(), set(), set()),
            ({"col_0", "col_1"}, set(), set()),
            ({"col_0"}, {"col_1"}, {"col_2"}),
            ({"col_0", "col_1", "col_2"}, {"col_0"}, set()),
            ({f"col_{i}" for i in range(8)}, set(), set()),
        ],
    )
    def test_status_is_identical_with_and_without_attribution(
        self, analyzer, used, join_keys, calc_inputs
    ):
        """The guarantee that makes this change safe to ship: attribution is
        reporting only. If it can move a column between used and unused, a
        reporting change can delete data."""
        attribution = UsageAttribution()
        # Attribute liberally, including columns that are not in any input set.
        for column in (c["Name"] for c in self.ALL_COLUMNS):
            attribution.record(
                column,
                ColumnUsageSource.CHILD_STRUCTURAL.value,
                requester_name="noise",
            )

        without = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns=used,
            join_key_columns=join_keys,
            calc_field_input_columns=calc_inputs,
        )
        with_attr = analyzer.classify_columns(
            all_columns=self.ALL_COLUMNS,
            used_columns=used,
            join_key_columns=join_keys,
            calc_field_input_columns=calc_inputs,
            attribution=attribution,
        )

        assert [(c.column_name, c.status) for c in without] == [
            (c.column_name, c.status) for c in with_attr
        ]


# --------------------------------------------------------------------------
# Handler: child attribution
# --------------------------------------------------------------------------


class TestChildAttribution:
    def _child(self, **overrides) -> ChildDataset:
        defaults = {
            "dataset_id": "child-1",
            "name": "orders_agg",
            "referenced_columns": {"a", "b", "c", "d"},
            "depth": 2,
            "join_key_columns": {"a"},
            "consumer_traced_columns": {"b"},
            "projected_columns": {"c"},
        }
        defaults.update(overrides)
        return ChildDataset(**defaults)

    def test_each_category_maps_to_its_own_source(self):
        attribution = UsageAttribution()
        Orchestrator._attribute_child_columns(
            attribution, [self._child()], {"a", "b", "c"}
        )

        assert attribution.sources_for("a") == {
            ColumnUsageSource.CHILD_JOIN_KEY.value
        }
        assert attribution.sources_for("b") == {
            ColumnUsageSource.CHILD_CONSUMER.value
        }
        assert attribution.sources_for("c") == {
            ColumnUsageSource.CHILD_PROJECTION.value
        }

    def test_child_name_and_depth_are_recorded(self):
        attribution = UsageAttribution()
        Orchestrator._attribute_child_columns(
            attribution, [self._child()], {"a", "b", "c"}
        )
        recorded = attribution.for_column("a")[0]

        assert recorded.requester_name == "orders_agg"
        assert recorded.requester_id == "child-1"
        assert recorded.detail == "depth 2"

    def test_columns_outside_the_traced_union_are_not_claimed(self):
        """The report must never claim protection for a column the pipeline did
        not in fact protect."""
        attribution = UsageAttribution()
        Orchestrator._attribute_child_columns(
            attribution, [self._child()], {"a"}
        )

        assert attribution.columns() == {"a"}

    def test_structural_fallback_marks_imported_columns_unproven(self):
        attribution = UsageAttribution()
        child = self._child(
            structural_fallback=True,
            join_key_columns=set(),
            consumer_traced_columns=set(),
            projected_columns=set(),
        )
        Orchestrator._attribute_child_columns(
            attribution, [child], {"a", "b", "c", "d"}
        )

        assert attribution.is_only_fallback("d") is True

    def test_no_fallback_means_no_structural_attribution(self):
        attribution = UsageAttribution()
        Orchestrator._attribute_child_columns(
            attribution, [self._child()], {"a", "b", "c", "d"}
        )

        assert not attribution.has_source(
            ColumnUsageSource.CHILD_STRUCTURAL.value
        )

    def test_multiple_children_are_both_named(self):
        attribution = UsageAttribution()
        children = [
            self._child(dataset_id="c1", name="first", join_key_columns={"a"}),
            self._child(dataset_id="c2", name="second", join_key_columns={"a"}),
        ]
        Orchestrator._attribute_child_columns(attribution, children, {"a"})

        names = {a.requester_name for a in attribution.for_column("a")}
        assert names == {"first", "second"}

    def test_empty_child_list_records_nothing(self):
        attribution = UsageAttribution()
        Orchestrator._attribute_child_columns(attribution, [], {"a", "b"})

        assert not attribution


# --------------------------------------------------------------------------
# Reporter
# --------------------------------------------------------------------------


class TestReportChildSummary:
    def test_child_datasets_reach_the_report(self, reporter):
        """Previously these were logged and dropped, so a report could not show
        which downstream dataset was blocking an optimisation."""
        child = ChildDataset(
            dataset_id="child-1",
            name="orders_agg",
            referenced_columns={"a", "b"},
            depth=1,
            join_key_columns={"a"},
            consumer_traced_columns={"b"},
            projected_columns=set(),
        )
        report = build_report(reporter, child_datasets=[child])

        assert report["child_datasets"] == [
            {
                "dataset_id": "child-1",
                "name": "orders_agg",
                "depth": 1,
                "required_columns": ["a", "b"],
                "join_key_columns": ["a"],
                "consumer_traced_columns": ["b"],
                "projected_columns": [],
                "structural_fallback": False,
            }
        ]

    def test_absent_children_yield_an_empty_list(self, reporter):
        assert build_report(reporter)["child_datasets"] == []


class TestReportColumnDetail:
    def _classifications(self, attribution: UsageAttribution):
        return UsageAnalyzer().classify_columns(
            all_columns=[
                {"Name": "region", "Type": "STRING"},
                {"Name": "mystery", "Type": "STRING"},
                {"Name": "junk", "Type": "STRING"},
            ],
            used_columns={"region", "mystery"},
            join_key_columns=set(),
            calc_field_input_columns=set(),
            attribution=attribution,
        )

    def test_required_by_names_the_requester(self, reporter):
        attribution = UsageAttribution()
        attribution.record(
            "region",
            ColumnUsageSource.DIRECT_CONSUMER.value,
            requester_name="Sales KPIs",
            requester_id="arn:dash",
        )
        attribution.record(
            "mystery", ColumnUsageSource.CHILD_REFS_UNKNOWN.value
        )

        report = build_report(
            reporter, classifications=self._classifications(attribution)
        )
        detail = {c["column_name"]: c for c in report["column_details"]}

        assert detail["region"]["required_by"][0]["name"] == "Sales KPIs"
        assert detail["region"]["required_by"][0]["id"] == "arn:dash"

    def test_unproven_columns_are_flagged(self, reporter):
        """"Nothing proved this is needed, but we could not prove it is not" calls
        for a different response from a column that is genuinely in use."""
        attribution = UsageAttribution()
        attribution.record(
            "region", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="Dash"
        )
        attribution.record(
            "mystery", ColumnUsageSource.CHILD_REFS_UNKNOWN.value
        )

        report = build_report(
            reporter, classifications=self._classifications(attribution)
        )
        detail = {c["column_name"]: c for c in report["column_details"]}

        assert detail["mystery"]["retained_without_proof"] is True
        assert detail["region"]["retained_without_proof"] is False

    def test_reason_string_is_still_present(self, reporter):
        """Kept for backward compatibility with anything reading the old shape."""
        attribution = UsageAttribution()
        attribution.record(
            "region", ColumnUsageSource.DIRECT_CONSUMER.value, requester_name="Dash"
        )

        report = build_report(
            reporter, classifications=self._classifications(attribution)
        )
        detail = {c["column_name"]: c for c in report["column_details"]}

        assert "referenced by consumer" in detail["region"]["reason"]

    def test_unattributed_columns_omit_the_new_keys(self, reporter):
        report = build_report(
            reporter, classifications=self._classifications(UsageAttribution())
        )

        for detail in report["column_details"]:
            assert "required_by" not in detail
            assert "retained_without_proof" not in detail


class TestReportRiskIndicators:
    def test_child_dependency_count_is_surfaced(self, reporter):
        children = [
            ChildDataset(dataset_id="c1", name="one"),
            ChildDataset(dataset_id="c2", name="two"),
        ]
        report = build_report(reporter, child_datasets=children)

        assert "2 child dataset(s) depend on this" in report["risk_indicators"]

    def test_incomplete_child_tracing_is_surfaced(self, reporter):
        children = [
            ChildDataset(dataset_id="c1", name="one", structural_fallback=True)
        ]
        report = build_report(reporter, child_datasets=children)

        assert any(
            "tracing incomplete" in indicator
            for indicator in report["risk_indicators"]
        )

    def test_unparseable_calc_field_is_surfaced(self, reporter):
        """This condition previously reached CloudWatch only, so a run where the
        analysis could not complete looked clean in the report."""
        attribution = UsageAttribution()
        attribution.record(
            "region", ColumnUsageSource.UNPARSEABLE_CALC_FIELD.value
        )
        classifications = UsageAnalyzer().classify_columns(
            all_columns=[{"Name": "region", "Type": "STRING"}],
            used_columns={"region"},
            join_key_columns=set(),
            calc_field_input_columns=set(),
            attribution=attribution,
        )
        report = build_report(reporter, classifications=classifications)

        assert any(
            "could not be parsed" in indicator
            for indicator in report["risk_indicators"]
        )

    @pytest.mark.parametrize("consumer_count,expected", [(0, True), (1, False), (2, False)])
    def test_no_consumers_indicator_is_unchanged(
        self, reporter, consumer_count, expected
    ):
        report = build_report(reporter, consumer_count=consumer_count)
        assert ("no consumers found" in report["risk_indicators"]) is expected

    @pytest.mark.parametrize("consumer_count,expected", [(0, False), (1, False), (5, True)])
    def test_shared_dataset_indicator_is_unchanged(
        self, reporter, consumer_count, expected
    ):
        """Property tests assert this invariant; adding indicators must not
        disturb it."""
        report = build_report(reporter, consumer_count=consumer_count)
        assert ("shared dataset" in report["risk_indicators"]) is expected

    def test_legacy_single_argument_call_still_works(self, reporter):
        """Property tests call this positionally with one argument."""
        assert reporter._get_risk_indicators(2) == ["shared dataset"]
        assert reporter._get_risk_indicators(0) == ["no consumers found"]


class TestHandlerRiskIndicatorsStayInStep:
    """The DatasetResult summary and the stored report must not disagree."""

    def test_child_indicators_match_the_reporter(self, reporter):
        children = [
            ChildDataset(dataset_id="c1", name="one", structural_fallback=True)
        ]
        from_handler = Orchestrator._get_risk_indicators(1, child_datasets=children)
        from_reporter = reporter._get_risk_indicators(1, child_datasets=children)

        assert from_handler == from_reporter

    def test_consumer_only_indicators_match_the_reporter(self, reporter):
        for count in (0, 1, 3):
            assert Orchestrator._get_risk_indicators(
                count
            ) == reporter._get_risk_indicators(count)
