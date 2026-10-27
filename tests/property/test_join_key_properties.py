# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property tests for join key extraction.

The defect fixed in `3785e17` was invisible to the existing tests because every one of
them built an `OnClause` in the unbraced form, `"col_a = col_b"` — see
`st_join_instruction` in strategies.py. QuickSight actually emits several forms, and the
parser silently produced unmatchable names for the braced ones: `id = {inventory_id}`
yielded the literal `"{inventory_id}"`, which matched nothing in `OutputColumns`, so the
join key was classified unused and reached a removal list.

Two properties are asserted here, and they matter in opposite directions:

**Completeness** — every real column named in a join clause must come back protected,
whatever syntax the clause used. This is the property that was violated. A format the
parser does not understand yields nothing, and "nothing" is indistinguishable from "this
column is not a join key", which is exactly how a join key becomes removable.

**Soundness** — extraction must never return a name that is not a real column. This is
what stops a future unrecognised format from failing silently: an unmatchable name would
otherwise just vanish in the intersection.

Together they mean a parser change cannot quietly stop protecting something.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lambda_source.analyzer import UsageAnalyzer
from lambda_source.on_clause import (
    candidate_names,
    join_keys_from_dataset,
    parse_on_clause,
)

# Column names QuickSight will accept, kept simple so the focus stays on clause syntax.
st_col = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() == s and not s.startswith("_"))


def as_bare(col: str) -> str:
    return col


def as_braced(col: str) -> str:
    return "{" + col + "}"


# Every rendering of a single column reference that QuickSight has been observed to emit.
# All were seen in a real QuickSight account on 2026-08-25.
RENDERERS = [as_bare, as_braced]


@st.composite
def st_clause_and_columns(draw: st.DrawFn) -> tuple[str, set[str]]:
    """An OnClause plus the real column names it references.

    Builds one or more `left = right` conditions joined by AND, rendering each side in
    a randomly chosen syntax, so the test covers mixed clauses like
    `id = {inventory_id}` — the exact form that broke.
    """
    conditions = draw(st.integers(min_value=1, max_value=3))
    columns: set[str] = set()
    parts: list[str] = []

    for _ in range(conditions):
        left = draw(st_col)
        right = draw(st_col)
        columns.update({left, right})
        render_left = draw(st.sampled_from(RENDERERS))
        render_right = draw(st.sampled_from(RENDERERS))
        separator = draw(st.sampled_from([" = ", "=", "  =  "]))
        parts.append(f"{render_left(left)}{separator}{render_right(right)}")

    joiner = draw(st.sampled_from([" AND ", " and ", " And "]))
    return joiner.join(parts), columns


def dataset_with_clause(on_clause: str, columns: set[str]) -> dict:
    return {
        "OutputColumns": [{"Name": c, "Type": "STRING"} for c in sorted(columns)],
        "LogicalTableMap": {
            "lt-1": {"Source": {"JoinInstruction": {"OnClause": on_clause}}}
        },
    }


class TestCompleteness:
    """Every real column named in a clause must come back protected."""

    @settings(max_examples=300)
    @given(st_clause_and_columns())
    def test_every_referenced_column_is_protected(self, clause_and_columns):
        clause, columns = clause_and_columns
        dataset = dataset_with_clause(clause, columns)

        protected = join_keys_from_dataset(dataset, columns)

        missing = columns - protected
        assert not missing, (
            f"clause {clause!r} references {sorted(columns)} but "
            f"{sorted(missing)} were not protected. An unprotected join key becomes a "
            "removal candidate."
        )

    @settings(max_examples=300)
    @given(st_clause_and_columns())
    def test_parsing_alone_yields_a_matchable_candidate_per_column(
        self, clause_and_columns
    ):
        """Checked one level below the intersection. If parse_on_clause never produces
        the real name, the intersection silently returns nothing and the failure looks
        like "not a join key" rather than "parser did not understand this"."""
        clause, columns = clause_and_columns

        candidates = parse_on_clause(clause)

        for column in columns:
            assert column in candidates, (
                f"clause {clause!r} contains column {column!r} but parsing produced "
                f"{sorted(candidates)}"
            )

    @settings(max_examples=200)
    @given(st_col)
    def test_the_table_qualified_form_yields_the_bare_column(self, col):
        """`{part_num[parts]}` means part_num from table parts, and QuickSight also
        materialises a column literally named part_num[parts]. Both readings must be
        offered, because the caller cannot know which exists."""
        qualified = "{" + col + "[sometable]}"

        names = candidate_names(qualified)

        assert col in names
        assert f"{col}[sometable]" in names

    @settings(max_examples=200)
    @given(st_col, st_col)
    def test_a_braced_clause_protects_both_sides(self, left, right):
        """The specific regression. `id = {inventory_id}` protected only `id`."""
        assume(left != right)
        clause = f"{left} = {{{right}}}"
        columns = {left, right}

        protected = join_keys_from_dataset(dataset_with_clause(clause, columns), columns)

        assert protected == columns


class TestSoundness:
    """Extraction must never return a name that is not a real column."""

    @settings(max_examples=300)
    @given(st_clause_and_columns())
    def test_extracted_keys_are_always_real_columns(self, clause_and_columns):
        """Any future format the parser does not understand then fails loudly here,
        instead of silently yielding a name nothing can match."""
        clause, columns = clause_and_columns
        dataset = dataset_with_clause(clause, columns)

        protected = join_keys_from_dataset(dataset, columns)

        assert protected <= columns, (
            f"clause {clause!r} produced {sorted(protected - columns)}, which are not "
            "columns of the dataset"
        )

    @settings(max_examples=200)
    @given(st_clause_and_columns())
    def test_no_syntax_characters_survive_into_a_key(self, clause_and_columns):
        """Braces and equals signs in a returned key mean the parser handed back raw
        clause text — the shape of the original defect."""
        clause, columns = clause_and_columns

        for key in join_keys_from_dataset(dataset_with_clause(clause, columns), columns):
            assert "{" not in key and "}" not in key
            assert "=" not in key
            assert key == key.strip()

    @settings(max_examples=200)
    @given(st_clause_and_columns())
    def test_no_empty_or_whitespace_keys(self, clause_and_columns):
        clause, columns = clause_and_columns

        for key in join_keys_from_dataset(dataset_with_clause(clause, columns), columns):
            assert key and key.strip()


class TestAnalyzerAgreesWithTheParser:
    """UsageAnalyzer must not diverge from the shared parser again.

    The original defect was precisely this divergence: child_discovery stripped braces,
    the analyzer did not, and the two disagreed about the same format.
    """

    @settings(max_examples=200)
    @given(st_clause_and_columns())
    def test_extract_all_join_keys_matches_the_shared_helper(self, clause_and_columns):
        clause, columns = clause_and_columns
        dataset = dataset_with_clause(clause, columns)

        assert UsageAnalyzer().extract_all_join_keys(dataset, columns) == (
            join_keys_from_dataset(dataset, columns)
        )

    @settings(max_examples=200)
    @given(st_clause_and_columns())
    def test_by_source_union_matches_the_flat_extractor(self, clause_and_columns):
        """Splitting by declaring shape must not change which columns are protected."""
        clause, columns = clause_and_columns
        dataset = dataset_with_clause(clause, columns)
        analyzer = UsageAnalyzer()

        by_source = analyzer.extract_join_keys_by_source(dataset, columns)

        assert set().union(*by_source.values()) == analyzer.extract_all_join_keys(
            dataset, columns
        )


class TestNewExperienceParity:
    """A new-experience JoinStep must be protected exactly as a legacy clause is.

    Before `3785e17` these had no protection at all: extract_join_keys read only
    LogicalTableMap, and 11 of 66 datasets in the account are new-experience.
    """

    @settings(max_examples=200)
    @given(st_clause_and_columns())
    def test_data_prep_join_steps_are_protected(self, clause_and_columns):
        clause, columns = clause_and_columns
        dataset = {
            "OutputColumns": [{"Name": c, "Type": "STRING"} for c in sorted(columns)],
            "DataPrepConfiguration": {
                "TransformStepMap": {"s1": {"JoinStep": {"OnClause": clause}}}
            },
        }

        assert join_keys_from_dataset(dataset, columns) == columns

    @settings(max_examples=100)
    @given(st_clause_and_columns())
    def test_both_shapes_present_yields_the_union(self, clause_and_columns):
        clause, columns = clause_and_columns
        dataset = {
            "OutputColumns": [{"Name": c, "Type": "STRING"} for c in sorted(columns)],
            "LogicalTableMap": {
                "lt-1": {"Source": {"JoinInstruction": {"OnClause": clause}}}
            },
            "DataPrepConfiguration": {
                "TransformStepMap": {"s1": {"JoinStep": {"OnClause": clause}}}
            },
        }

        assert join_keys_from_dataset(dataset, columns) == columns
