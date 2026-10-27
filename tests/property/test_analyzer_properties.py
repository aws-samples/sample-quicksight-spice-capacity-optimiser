# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the Usage Analyzer module.

Tests correctness properties related to column classification, including
join key preservation, exhaustive partitioning, and safety threshold evaluation.

Feature: spice-dataset-optimiser
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from lambda_source.analyzer import UsageAnalyzer
from lambda_source.models import ColumnStatus
from tests.property.strategies import (
    st_column_classification_inputs,
    st_column_name,
    st_data_type,
)


# ---------------------------------------------------------------------------
# Property 5: Join key columns always classified as used
# ---------------------------------------------------------------------------


class TestJoinKeyColumnClassification:
    """Property 5: Join key columns always classified as used.

    For any dataset definition containing JoinInstructions, all columns referenced
    in OnClause, LeftJoinKeyProperties, and RightJoinKeyProperties SHALL be classified
    as Used_Column regardless of whether they appear in any consumer's column
    references. The classification reason SHALL indicate "join key".

    Feature: spice-dataset-optimiser, Property 5: Join key columns always classified as used
    **Validates: Requirements 3.1, 3.2, 3.3**
    """

    @given(data=st_column_classification_inputs(min_total=3, max_total=30))
    @settings(max_examples=100)
    def test_join_key_columns_never_classified_as_unused(
        self,
        data: dict,
    ) -> None:
        """All columns in join_key_columns are classified as join_key status,
        never as unused, regardless of whether they appear in used_columns.

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        all_columns = data["all_columns"]
        used_columns = data["used_columns"]
        join_key_columns = data["join_key_columns"]
        calc_field_inputs = data["calc_field_inputs"]

        # Ensure there's at least one join key column to test
        # If strategy generated empty join_key_columns, skip
        if not join_key_columns:
            return

        analyzer = UsageAnalyzer()

        classifications = analyzer.classify_columns(
            all_columns=all_columns,
            used_columns=used_columns,
            join_key_columns=join_key_columns,
            calc_field_input_columns=calc_field_inputs,
        )

        # Build lookup by column name
        classification_map = {c.column_name: c for c in classifications}

        # Assert: every join key column is classified as "join_key" (not "unused")
        for jk_col in join_key_columns:
            assert jk_col in classification_map, (
                f"Join key column '{jk_col}' not found in classifications"
            )
            classification = classification_map[jk_col]
            assert classification.status == ColumnStatus.JOIN_KEY.value, (
                f"Join key column '{jk_col}' classified as '{classification.status}' "
                f"instead of '{ColumnStatus.JOIN_KEY.value}'"
            )

    @given(data=st_column_classification_inputs(min_total=3, max_total=30))
    @settings(max_examples=100)
    def test_join_key_columns_have_join_key_reason(
        self,
        data: dict,
    ) -> None:
        """All columns in join_key_columns have a reason containing "join key".

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        all_columns = data["all_columns"]
        used_columns = data["used_columns"]
        join_key_columns = data["join_key_columns"]
        calc_field_inputs = data["calc_field_inputs"]

        # Skip if no join key columns generated
        if not join_key_columns:
            return

        analyzer = UsageAnalyzer()

        classifications = analyzer.classify_columns(
            all_columns=all_columns,
            used_columns=used_columns,
            join_key_columns=join_key_columns,
            calc_field_input_columns=calc_field_inputs,
        )

        # Build lookup by column name
        classification_map = {c.column_name: c for c in classifications}

        # Assert: every join key column has "join key" text in its reasons
        for jk_col in join_key_columns:
            assert jk_col in classification_map, (
                f"Join key column '{jk_col}' not found in classifications"
            )
            classification = classification_map[jk_col]
            reasons_text = " ".join(classification.reasons).lower()
            assert "join key" in reasons_text, (
                f"Join key column '{jk_col}' reasons do not contain 'join key': "
                f"{classification.reasons}"
            )

    @given(data=st_column_classification_inputs(min_total=3, max_total=30))
    @settings(max_examples=100)
    def test_join_key_not_in_used_columns_still_classified_as_join_key(
        self,
        data: dict,
    ) -> None:
        """Join key columns are classified as join_key even when they do NOT
        appear in any consumer's used_columns set.

        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        all_columns = data["all_columns"]
        join_key_columns = data["join_key_columns"]
        calc_field_inputs = data["calc_field_inputs"]

        # Skip if no join key columns generated
        if not join_key_columns:
            return

        # Explicitly remove join key columns from used_columns to test isolation
        used_columns = data["used_columns"] - join_key_columns

        analyzer = UsageAnalyzer()

        classifications = analyzer.classify_columns(
            all_columns=all_columns,
            used_columns=used_columns,
            join_key_columns=join_key_columns,
            calc_field_input_columns=calc_field_inputs,
        )

        # Build lookup by column name
        classification_map = {c.column_name: c for c in classifications}

        # Assert: join key columns are still classified as join_key
        for jk_col in join_key_columns:
            assert jk_col in classification_map, (
                f"Join key column '{jk_col}' not found in classifications"
            )
            classification = classification_map[jk_col]
            assert classification.status == ColumnStatus.JOIN_KEY.value, (
                f"Join key column '{jk_col}' classified as '{classification.status}' "
                f"when not in used_columns. Expected '{ColumnStatus.JOIN_KEY.value}'"
            )
            # Also verify it has the join key reason
            reasons_text = " ".join(classification.reasons).lower()
            assert "join key" in reasons_text, (
                f"Join key column '{jk_col}' reasons do not contain 'join key': "
                f"{classification.reasons}"
            )


# ---------------------------------------------------------------------------
# Property 6: Column classification is a correct, exhaustive partition
# ---------------------------------------------------------------------------


class TestColumnClassificationPartition:
    """Property 6: Column classification is a correct, exhaustive partition.

    For any dataset with a set of all columns C, a set of used columns U
    (from consumer references), a set of join key columns J, and a set of
    dataset-level calculated field input columns F, every column in C SHALL
    be classified exactly once: as "used" if it is in U ∪ J ∪ F, or as
    "unused" otherwise. The union of the used and unused sets SHALL equal C,
    and their intersection SHALL be empty.

    Feature: spice-dataset-optimiser, Property 6: Column classification is a correct, exhaustive partition
    **Validates: Requirements 4.1, 4.2, 4.3**
    """

    @given(inputs=st_column_classification_inputs())
    @settings(max_examples=100)
    def test_every_column_classified_exactly_once(self, inputs: dict) -> None:
        """Every column in all_columns appears exactly once in classifications.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        analyzer = UsageAnalyzer()
        classifications = analyzer.classify_columns(
            all_columns=inputs["all_columns"],
            used_columns=inputs["used_columns"],
            join_key_columns=inputs["join_key_columns"],
            calc_field_input_columns=inputs["calc_field_inputs"],
        )

        all_names = {col["Name"] for col in inputs["all_columns"]}
        classified_names = [c.column_name for c in classifications]

        # Every column appears exactly once
        assert len(classified_names) == len(all_names)
        assert set(classified_names) == all_names

    @given(inputs=st_column_classification_inputs())
    @settings(max_examples=100)
    def test_used_and_unused_union_equals_all_columns(self, inputs: dict) -> None:
        """The union of used and unused sets equals all columns.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        analyzer = UsageAnalyzer()
        classifications = analyzer.classify_columns(
            all_columns=inputs["all_columns"],
            used_columns=inputs["used_columns"],
            join_key_columns=inputs["join_key_columns"],
            calc_field_input_columns=inputs["calc_field_inputs"],
        )

        all_names = {col["Name"] for col in inputs["all_columns"]}

        # "used" includes both ColumnStatus.USED and ColumnStatus.JOIN_KEY
        used_set = {
            c.column_name
            for c in classifications
            if c.status in (ColumnStatus.USED.value, ColumnStatus.JOIN_KEY.value)
        }
        unused_set = {
            c.column_name
            for c in classifications
            if c.status == ColumnStatus.UNUSED.value
        }

        # Union of used and unused equals all columns
        assert used_set | unused_set == all_names

    @given(inputs=st_column_classification_inputs())
    @settings(max_examples=100)
    def test_used_and_unused_intersection_is_empty(self, inputs: dict) -> None:
        """The intersection of used and unused sets is empty.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        analyzer = UsageAnalyzer()
        classifications = analyzer.classify_columns(
            all_columns=inputs["all_columns"],
            used_columns=inputs["used_columns"],
            join_key_columns=inputs["join_key_columns"],
            calc_field_input_columns=inputs["calc_field_inputs"],
        )

        used_set = {
            c.column_name
            for c in classifications
            if c.status in (ColumnStatus.USED.value, ColumnStatus.JOIN_KEY.value)
        }
        unused_set = {
            c.column_name
            for c in classifications
            if c.status == ColumnStatus.UNUSED.value
        }

        # Intersection is empty
        assert used_set & unused_set == set()

    @given(inputs=st_column_classification_inputs())
    @settings(max_examples=100)
    def test_columns_in_u_j_f_classified_as_used_or_join_key(self, inputs: dict) -> None:
        """Any column in U ∪ J ∪ F is classified as 'used' or 'join_key'.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        analyzer = UsageAnalyzer()
        classifications = analyzer.classify_columns(
            all_columns=inputs["all_columns"],
            used_columns=inputs["used_columns"],
            join_key_columns=inputs["join_key_columns"],
            calc_field_input_columns=inputs["calc_field_inputs"],
        )

        all_names = {col["Name"] for col in inputs["all_columns"]}
        referenced_columns = (
            inputs["used_columns"] | inputs["join_key_columns"] | inputs["calc_field_inputs"]
        )
        # Only check columns that are actually in the dataset
        referenced_in_dataset = referenced_columns & all_names

        classification_map = {c.column_name: c.status for c in classifications}

        for col in referenced_in_dataset:
            assert classification_map[col] in (
                ColumnStatus.USED.value,
                ColumnStatus.JOIN_KEY.value,
            ), f"Column '{col}' is in U∪J∪F but classified as '{classification_map[col]}'"

    @given(inputs=st_column_classification_inputs())
    @settings(max_examples=100)
    def test_columns_not_in_u_j_f_classified_as_unused(self, inputs: dict) -> None:
        """Any column NOT in U ∪ J ∪ F is classified as 'unused'.

        **Validates: Requirements 4.1, 4.2, 4.3**
        """
        analyzer = UsageAnalyzer()
        classifications = analyzer.classify_columns(
            all_columns=inputs["all_columns"],
            used_columns=inputs["used_columns"],
            join_key_columns=inputs["join_key_columns"],
            calc_field_input_columns=inputs["calc_field_inputs"],
        )

        all_names = {col["Name"] for col in inputs["all_columns"]}
        referenced_columns = (
            inputs["used_columns"] | inputs["join_key_columns"] | inputs["calc_field_inputs"]
        )
        unreferenced_in_dataset = all_names - referenced_columns

        classification_map = {c.column_name: c.status for c in classifications}

        for col in unreferenced_in_dataset:
            assert classification_map[col] == ColumnStatus.UNUSED.value, (
                f"Column '{col}' is NOT in U∪J∪F but classified as '{classification_map[col]}'"
            )


# ---------------------------------------------------------------------------
# Property 7: Safety threshold evaluation
# ---------------------------------------------------------------------------


@st.composite
def st_column_classifications_with_ratio(draw: st.DrawFn) -> list:
    """Generate a list of ColumnClassification with varying used/unused ratios.

    Generates between 1 and 50 columns, each randomly assigned as used,
    unused, or join_key status.
    """
    from lambda_source.models import ColumnClassification as CC

    total = draw(st.integers(min_value=1, max_value=50))
    classifications = []

    for i in range(total):
        status = draw(
            st.sampled_from([
                ColumnStatus.USED.value,
                ColumnStatus.UNUSED.value,
                ColumnStatus.JOIN_KEY.value,
            ])
        )
        classifications.append(
            CC(
                column_name=f"col_{i}",
                data_type=draw(st.sampled_from(["STRING", "INTEGER", "DECIMAL", "DATETIME"])),
                status=status,
                reasons=[f"test reason for {status}"],
            )
        )

    return classifications


class TestSafetyThresholdEvaluation:
    """Property 7: Safety threshold evaluation.

    For any column classification result and configured safety threshold T
    (0.0–1.0), the threshold check SHALL return "breached" if and only if
    (unused_count / total_count) > T. The evaluation result SHALL be
    deterministic for the same inputs.

    Feature: spice-dataset-optimiser, Property 7: Safety threshold evaluation
    **Validates: Requirements 5.5, 11.2, 11.3**
    """

    @given(
        classifications=st_column_classifications_with_ratio(),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_threshold_breached_iff_ratio_exceeds_threshold(
        self,
        classifications: list,
        threshold: float,
    ) -> None:
        """The threshold check returns breached (False) if and only if
        (unused_count / total_count) > threshold.

        **Validates: Requirements 5.5, 11.2, 11.3**
        """
        from lambda_source.models import ColumnClassification as CC

        analyzer = UsageAnalyzer()

        # Call the method under test
        result = analyzer.check_safety_threshold(classifications, threshold)

        # Compute expected behaviour
        total_count = len(classifications)
        unused_count = sum(
            1 for c in classifications if c.status == ColumnStatus.UNUSED.value
        )

        if total_count == 0:
            # Edge case: empty list means safe
            assert result is True
        else:
            removal_ratio = unused_count / total_count
            expected_breached = removal_ratio > threshold

            # check_safety_threshold returns True if safe (within threshold),
            # False if breached (too many unused columns)
            if expected_breached:
                assert result is False, (
                    f"Expected breached (False) for ratio={removal_ratio:.4f} > "
                    f"threshold={threshold:.4f}, but got safe (True)"
                )
            else:
                assert result is True, (
                    f"Expected safe (True) for ratio={removal_ratio:.4f} <= "
                    f"threshold={threshold:.4f}, but got breached (False)"
                )

    @given(
        classifications=st_column_classifications_with_ratio(),
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_threshold_evaluation_is_deterministic(
        self,
        classifications: list,
        threshold: float,
    ) -> None:
        """Calling check_safety_threshold with the same inputs always produces
        the same result.

        **Validates: Requirements 5.5, 11.2, 11.3**
        """
        analyzer = UsageAnalyzer()

        result_first = analyzer.check_safety_threshold(classifications, threshold)
        result_second = analyzer.check_safety_threshold(classifications, threshold)

        assert result_first == result_second, (
            "check_safety_threshold is not deterministic: "
            f"first call returned {result_first}, second call returned {result_second}"
        )
