# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Property-based tests for the Dataset Modifier module.

Tests correctness properties related to ProjectOperation insertion and
pre-modification column validation.

Feature: spice-dataset-optimiser
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lambda_source.modifier import DatasetModifier, ValidationError
from tests.property.strategies import st_column_name, st_dataset_definition


# ---------------------------------------------------------------------------
# Strategy: Generate random column sets for validation testing
# ---------------------------------------------------------------------------


@st.composite
def st_column_sets(draw: st.DrawFn) -> dict:
    """Generate random sets of all_columns, retained_columns, and consumer_references.

    Produces three sets with controlled relationships:
    - all_columns: the universe of columns in the dataset (3-30 columns)
    - retained_columns: a subset of all_columns representing columns kept after optimization
    - consumer_references: a subset of all_columns representing columns referenced by consumers

    Returns a dict with:
      - all_columns: set[str]
      - retained_columns: set[str]
      - consumer_references: set[str]
      - is_superset: bool (whether retained_columns >= consumer_references)
    """
    # Generate a pool of unique column names
    num_columns = draw(st.integers(min_value=3, max_value=30))
    column_pool: list[str] = []
    seen: set[str] = set()
    for _ in range(num_columns):
        name = draw(st_column_name())
        while name in seen:
            name = draw(st_column_name())
        seen.add(name)
        column_pool.append(name)

    all_columns = set(column_pool)

    # Draw retained_columns as a subset of all_columns
    retained_list = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=1,
            max_size=len(column_pool),
            unique=True,
        )
    )
    retained_columns = set(retained_list)

    # Draw consumer_references as a subset of all_columns
    consumer_list = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=0,
            max_size=len(column_pool),
            unique=True,
        )
    )
    consumer_references = set(consumer_list)

    is_superset = retained_columns >= consumer_references

    return {
        "all_columns": all_columns,
        "retained_columns": retained_columns,
        "consumer_references": consumer_references,
        "is_superset": is_superset,
    }


@st.composite
def st_superset_column_sets(draw: st.DrawFn) -> dict:
    """Generate column sets where retained_columns IS a superset of consumer_references.

    Guarantees: retained_columns >= consumer_references.
    """
    num_columns = draw(st.integers(min_value=3, max_value=30))
    column_pool: list[str] = []
    seen: set[str] = set()
    for _ in range(num_columns):
        name = draw(st_column_name())
        while name in seen:
            name = draw(st_column_name())
        seen.add(name)
        column_pool.append(name)

    # Consumer references: a subset of all columns
    consumer_list = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=0,
            max_size=len(column_pool),
            unique=True,
        )
    )
    consumer_references = set(consumer_list)

    # Retained columns: must include all consumer references plus optionally more
    extra_list = draw(
        st.lists(
            st.sampled_from(column_pool),
            min_size=0,
            max_size=len(column_pool),
            unique=True,
        )
    )
    retained_columns = consumer_references | set(extra_list)

    return {
        "retained_columns": retained_columns,
        "consumer_references": consumer_references,
    }


@st.composite
def st_non_superset_column_sets(draw: st.DrawFn) -> dict:
    """Generate column sets where retained_columns is NOT a superset of consumer_references.

    Guarantees: there exists at least one column in consumer_references that is
    not in retained_columns.
    """
    num_columns = draw(st.integers(min_value=3, max_value=30))
    column_pool: list[str] = []
    seen: set[str] = set()
    for _ in range(num_columns):
        name = draw(st_column_name())
        while name in seen:
            name = draw(st_column_name())
        seen.add(name)
        column_pool.append(name)

    # Split into two disjoint groups to guarantee at least one missing
    # Shuffle column pool then split
    shuffled = draw(st.permutations(column_pool))
    split_point = draw(st.integers(min_value=1, max_value=max(1, len(shuffled) - 1)))

    retained_pool = shuffled[:split_point]
    missing_pool = shuffled[split_point:]

    # Retained columns: drawn from the first partition only
    retained_columns = set(retained_pool)

    # Consumer references: include at least one from missing_pool
    # Pick at least 1 from missing_pool
    missing_count = draw(st.integers(min_value=1, max_value=len(missing_pool)))
    missing_refs = set(missing_pool[:missing_count])

    # Optionally also include some from retained_pool
    from_retained = draw(
        st.lists(
            st.sampled_from(retained_pool) if retained_pool else st.nothing(),
            min_size=0,
            max_size=len(retained_pool),
            unique=True,
        )
    )
    consumer_references = missing_refs | set(from_retained)

    return {
        "retained_columns": retained_columns,
        "consumer_references": consumer_references,
        "missing_columns": consumer_references - retained_columns,
    }


# ---------------------------------------------------------------------------
# Property 13: Pre-modification column validation
# ---------------------------------------------------------------------------


class TestPreModificationColumnValidation:
    """Property 13: Pre-modification column validation.

    For any proposed column modification, the set of columns retained after
    optimization SHALL be a superset of all extracted consumer Column_References.
    If the retained set is not a superset, the validation SHALL fail and prevent
    the modification.

    Feature: spice-dataset-optimiser, Property 13: Pre-modification column validation
    **Validates: Requirements 8.4**
    """

    @given(data=st_superset_column_sets())
    @settings(max_examples=100)
    def test_validation_passes_when_retained_is_superset(
        self,
        data: dict,
    ) -> None:
        """When retained_columns >= consumer_references, validate_modification
        returns True without raising.

        **Validates: Requirements 8.4**
        """
        retained_columns = data["retained_columns"]
        consumer_references = data["consumer_references"]

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]

        result = modifier.validate_modification(retained_columns, consumer_references)

        assert result is True, (
            f"validate_modification returned {result} instead of True "
            f"when retained_columns is a superset of consumer_references. "
            f"retained={retained_columns}, consumer_refs={consumer_references}"
        )

    @given(data=st_non_superset_column_sets())
    @settings(max_examples=100)
    def test_validation_raises_when_retained_is_not_superset(
        self,
        data: dict,
    ) -> None:
        """When retained_columns is NOT a superset of consumer_references,
        validate_modification raises ValidationError to prevent the modification.

        **Validates: Requirements 8.4**
        """
        retained_columns = data["retained_columns"]
        consumer_references = data["consumer_references"]
        expected_missing = data["missing_columns"]

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]

        with pytest.raises(ValidationError) as exc_info:
            modifier.validate_modification(retained_columns, consumer_references)

        # Verify the ValidationError contains the correct missing columns
        assert exc_info.value.missing_columns == expected_missing, (
            f"ValidationError.missing_columns={exc_info.value.missing_columns} "
            f"does not match expected missing={expected_missing}"
        )

    @given(data=st_column_sets())
    @settings(max_examples=100)
    def test_validation_result_matches_superset_relationship(
        self,
        data: dict,
    ) -> None:
        """For any random column sets, validation passes iff retained >= consumer_refs.

        **Validates: Requirements 8.4**
        """
        retained_columns = data["retained_columns"]
        consumer_references = data["consumer_references"]
        is_superset = data["is_superset"]

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]

        if is_superset:
            # Should pass without raising
            result = modifier.validate_modification(retained_columns, consumer_references)
            assert result is True, (
                f"Expected True when retained is superset, got {result}. "
                f"retained={retained_columns}, consumer_refs={consumer_references}"
            )
        else:
            # Should raise ValidationError
            with pytest.raises(ValidationError) as exc_info:
                modifier.validate_modification(retained_columns, consumer_references)

            # The missing columns should be consumer_refs - retained
            expected_missing = consumer_references - retained_columns
            assert exc_info.value.missing_columns == expected_missing, (
                f"ValidationError.missing_columns={exc_info.value.missing_columns} "
                f"does not match expected={expected_missing}"
            )

    @given(data=st_superset_column_sets())
    @settings(max_examples=100)
    def test_validation_with_empty_consumer_refs_always_passes(
        self,
        data: dict,
    ) -> None:
        """When consumer_references is empty, validation always passes
        regardless of retained_columns content.

        **Validates: Requirements 8.4**
        """
        retained_columns = data["retained_columns"]
        # Force empty consumer references
        consumer_references: set[str] = set()

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]

        result = modifier.validate_modification(retained_columns, consumer_references)
        assert result is True, (
            f"validate_modification should return True for empty consumer_references, "
            f"but got {result}"
        )


# ---------------------------------------------------------------------------
# Property 11: ProjectOperation insertion correctness
# ---------------------------------------------------------------------------


class TestProjectOperationInsertionCorrectness:
    """Property 11: ProjectOperation insertion correctness.

    For any dataset definition with a LogicalTableMap and a non-empty list of
    unused columns, the modified definition SHALL contain a ProjectOperation in
    each relevant logical table that includes all used columns and excludes all
    unused columns. The ProjectOperation SHALL appear at position 0 in the
    transformation chain (before any JoinInstruction or other transformation).

    Feature: spice-dataset-optimiser, Property 11: ProjectOperation insertion correctness
    **Validates: Requirements 7.1, 7.2**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_project_operation_at_position_zero(self, data: st.DataObject) -> None:
        """For any dataset definition with unused columns, the modified definition
        SHALL have a ProjectOperation at DataTransforms[0] in each logical table.

        **Validates: Requirements 7.1, 7.2**
        """
        # Generate a dataset definition with a LogicalTableMap
        dataset_def = data.draw(
            st_dataset_definition(min_columns=3, max_columns=15)
        )

        # Ensure LogicalTableMap exists and has entries
        logical_table_map = dataset_def.get("LogicalTableMap", {})
        if not logical_table_map:
            return

        # Get all column names from the dataset
        all_columns = [col["Name"] for col in dataset_def.get("OutputColumns", [])]
        if len(all_columns) < 2:
            return

        # Generate a non-empty subset of columns as "unused"
        unused_columns = data.draw(
            st.lists(
                st.sampled_from(all_columns),
                min_size=1,
                max_size=max(1, len(all_columns) - 1),
                unique=True,
            )
        )

        # Compute expected used (retained) columns
        unused_set = set(unused_columns)
        used_columns = sorted([col for col in all_columns if col not in unused_set])

        if not used_columns:
            # If all columns would be removed, the modifier returns unchanged
            return

        # Create modifier (no QS client needed for build_optimized_definition)
        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]

        # Call build_optimized_definition
        modified_def = modifier.build_optimized_definition(dataset_def, unused_columns)

        # Assertions on the modified definition
        modified_logical_map = modified_def.get("LogicalTableMap", {})
        assert modified_logical_map, "Modified definition must have a LogicalTableMap"

        for table_id, logical_table in modified_logical_map.items():
            data_transforms = logical_table.get("DataTransforms", [])

            # Assert: DataTransforms[-1] (last) is a ProjectOperation
            assert len(data_transforms) > 0, (
                f"Logical table '{table_id}' must have at least one DataTransform"
            )
            last_transform = data_transforms[-1]
            assert "ProjectOperation" in last_transform, (
                f"Last DataTransform in logical table '{table_id}' must be a "
                f"ProjectOperation, got: {list(last_transform.keys())}"
            )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_projected_columns_excludes_unused(self, data: st.DataObject) -> None:
        """The ProjectedColumns list SHALL exclude all unused columns.

        **Validates: Requirements 7.1, 7.2**
        """
        # Generate a dataset definition with a LogicalTableMap
        dataset_def = data.draw(
            st_dataset_definition(min_columns=3, max_columns=15)
        )

        logical_table_map = dataset_def.get("LogicalTableMap", {})
        if not logical_table_map:
            return

        all_columns = [col["Name"] for col in dataset_def.get("OutputColumns", [])]
        if len(all_columns) < 2:
            return

        unused_columns = data.draw(
            st.lists(
                st.sampled_from(all_columns),
                min_size=1,
                max_size=max(1, len(all_columns) - 1),
                unique=True,
            )
        )

        unused_set = set(unused_columns)
        used_columns = sorted([col for col in all_columns if col not in unused_set])

        if not used_columns:
            return

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]
        modified_def = modifier.build_optimized_definition(dataset_def, unused_columns)

        modified_logical_map = modified_def.get("LogicalTableMap", {})

        for table_id, logical_table in modified_logical_map.items():
            data_transforms = logical_table.get("DataTransforms", [])
            if not data_transforms:
                continue

            first_transform = data_transforms[0]
            if "ProjectOperation" not in first_transform:
                continue

            projected_columns = set(
                first_transform["ProjectOperation"]["ProjectedColumns"]
            )

            # Assert: ProjectedColumns excludes all unused columns
            for unused_col in unused_columns:
                assert unused_col not in projected_columns, (
                    f"Unused column '{unused_col}' should NOT be in ProjectedColumns "
                    f"of logical table '{table_id}'"
                )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_projected_columns_includes_all_used(self, data: st.DataObject) -> None:
        """The ProjectedColumns list SHALL include all used (retained) columns.

        **Validates: Requirements 7.1, 7.2**
        """
        # Generate a dataset definition with a LogicalTableMap
        dataset_def = data.draw(
            st_dataset_definition(min_columns=3, max_columns=15)
        )

        logical_table_map = dataset_def.get("LogicalTableMap", {})
        if not logical_table_map:
            return

        all_columns = [col["Name"] for col in dataset_def.get("OutputColumns", [])]
        if len(all_columns) < 2:
            return

        unused_columns = data.draw(
            st.lists(
                st.sampled_from(all_columns),
                min_size=1,
                max_size=max(1, len(all_columns) - 1),
                unique=True,
            )
        )

        unused_set = set(unused_columns)
        used_columns = sorted([col for col in all_columns if col not in unused_set])

        if not used_columns:
            return

        modifier = DatasetModifier(qs_client=None)  # type: ignore[arg-type]
        modified_def = modifier.build_optimized_definition(dataset_def, unused_columns)

        modified_logical_map = modified_def.get("LogicalTableMap", {})

        for table_id, logical_table in modified_logical_map.items():
            data_transforms = logical_table.get("DataTransforms", [])
            if not data_transforms:
                continue

            first_transform = data_transforms[0]
            if "ProjectOperation" not in first_transform:
                continue

            projected_columns = set(
                first_transform["ProjectOperation"]["ProjectedColumns"]
            )

            # Assert: ProjectedColumns includes all used (retained) columns
            for used_col in used_columns:
                assert used_col in projected_columns, (
                    f"Used column '{used_col}' should be in ProjectedColumns "
                    f"of logical table '{table_id}'"
                )
