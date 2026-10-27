# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom Hypothesis strategies for generating valid QuickSight structures.

Provides reusable composite strategies for property-based testing of the
SPICE Dataset Optimiser. All strategies generate structures that conform to
what the QuickSight APIs return (DescribeDataSet, DescribeAnalysisDefinition,
DescribeDashboardDefinition).

Usage:
    from tests.property.strategies import (
        st_column_name,
        st_dataset_definition,
        st_analysis_definition,
        st_logical_table_map,
        st_config,
    )

Configuration:
    Minimum 100 iterations per property test via settings profile.
"""

from __future__ import annotations

import string
from hypothesis import settings, HealthCheck
from hypothesis import strategies as st

from lambda_source.models import ExecutionMode, OptimiserConfig


# ---------------------------------------------------------------------------
# Global Hypothesis settings: minimum 100 iterations per property test
# ---------------------------------------------------------------------------

settings.register_profile(
    "property",
    max_examples=100,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.large_base_example,
    ],
)
settings.load_profile("property")


# ---------------------------------------------------------------------------
# QuickSight data types
# ---------------------------------------------------------------------------

QS_DATA_TYPES = ["STRING", "INTEGER", "DECIMAL", "DATETIME"]

QS_CONSUMER_STATUSES = [
    "CREATION_SUCCESSFUL",
    "CREATION_FAILED",
    "UPDATE_SUCCESSFUL",
    "UPDATE_FAILED",
    "DELETED",
]


# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------


@st.composite
def st_column_name(draw: st.DrawFn) -> str:
    """Generate a valid QuickSight column name string.

    QuickSight column names can contain alphanumeric characters, underscores,
    hyphens, spaces, and periods. They must be 1-128 characters long and
    start with a letter or underscore.
    """
    # First character: letter or underscore
    first_char = draw(
        st.sampled_from(
            list(string.ascii_letters) + ["_"]
        )
    )
    # Remaining characters: alphanumeric, underscore, hyphen, space, period
    valid_chars = string.ascii_letters + string.digits + "_- ."
    rest_length = draw(st.integers(min_value=0, max_value=63))
    rest = draw(
        st.text(
            alphabet=valid_chars,
            min_size=rest_length,
            max_size=rest_length,
        )
    )
    name = first_char + rest
    # Strip trailing spaces to avoid edge cases
    return name.rstrip()


@st.composite
def st_dataset_id(draw: st.DrawFn) -> str:
    """Generate a valid QuickSight dataset identifier (UUID-like)."""
    hex_chars = string.hexdigits[:16]
    segments = [8, 4, 4, 4, 12]
    parts = []
    for seg_len in segments:
        part = draw(st.text(alphabet=hex_chars, min_size=seg_len, max_size=seg_len))
        parts.append(part)
    return "-".join(parts)


@st.composite
def st_dataset_identifier(draw: st.DrawFn) -> str:
    """Generate a DataSetIdentifier string used in analysis/dashboard definitions."""
    prefix = draw(st.sampled_from(["ds", "dataset", "data", "src"]))
    suffix = draw(st.integers(min_value=1, max_value=999))
    return f"{prefix}_{suffix}"


@st.composite
def st_aws_account_id(draw: st.DrawFn) -> str:
    """Generate a valid 12-digit AWS account ID."""
    return draw(st.text(alphabet=string.digits, min_size=12, max_size=12))


@st.composite
def st_aws_region(draw: st.DrawFn) -> str:
    """Generate a valid AWS region string."""
    return draw(
        st.sampled_from([
            "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
            "ap-southeast-1", "ap-northeast-1", "us-east-2",
        ])
    )


@st.composite
def st_arn(draw: st.DrawFn, resource_type: str = "analysis") -> str:
    """Generate a valid QuickSight ARN."""
    account_id = draw(st_aws_account_id())
    region = draw(st_aws_region())
    resource_id = draw(st_dataset_id())
    return f"arn:aws:quicksight:{region}:{account_id}:{resource_type}/{resource_id}"


@st.composite
def st_s3_bucket_name(draw: st.DrawFn) -> str:
    """Generate a valid S3 bucket name."""
    prefix = draw(st.sampled_from(["my-bucket", "qs-optimiser", "spice-data", "reports"]))
    suffix = draw(st.integers(min_value=1, max_value=9999))
    return f"{prefix}-{suffix}"


@st.composite
def st_data_type(draw: st.DrawFn) -> str:
    """Generate a valid QuickSight column data type."""
    return draw(st.sampled_from(QS_DATA_TYPES))


# ---------------------------------------------------------------------------
# Column and Output Column strategies
# ---------------------------------------------------------------------------


@st.composite
def st_output_column(draw: st.DrawFn, name: str | None = None) -> dict:
    """Generate a single OutputColumn entry.

    Args:
        name: Optional fixed column name. If None, generates one.

    Returns:
        Dict with Name and Type keys.
    """
    col_name = name if name is not None else draw(st_column_name())
    col_type = draw(st_data_type())
    return {"Name": col_name, "Type": col_type}


@st.composite
def st_output_columns(
    draw: st.DrawFn,
    min_columns: int = 1,
    max_columns: int = 20,
    column_names: list[str] | None = None,
) -> list[dict]:
    """Generate a list of OutputColumns with unique names.

    Args:
        min_columns: Minimum number of columns to generate.
        max_columns: Maximum number of columns to generate.
        column_names: If provided, use these names (ignoring min/max).
    """
    if column_names is not None:
        columns = []
        for name in column_names:
            col = draw(st_output_column(name=name))
            columns.append(col)
        return columns

    num_columns = draw(st.integers(min_value=min_columns, max_value=max_columns))
    names: set[str] = set()
    columns: list[dict] = []
    for _ in range(num_columns):
        # Ensure unique names
        name = draw(st_column_name())
        while name in names:
            name = draw(st_column_name())
        names.add(name)
        col = draw(st_output_column(name=name))
        columns.append(col)
    return columns


# ---------------------------------------------------------------------------
# Physical Table Map strategies
# ---------------------------------------------------------------------------


@st.composite
def st_physical_table_map(
    draw: st.DrawFn, column_names: list[str] | None = None
) -> dict:
    """Generate a valid PhysicalTableMap.

    Each physical table has a unique key and contains either a
    RelationalTable, CustomSql, or S3Source with InputColumns.
    """
    num_tables = draw(st.integers(min_value=1, max_value=3))
    physical_map: dict = {}

    for i in range(num_tables):
        table_id = f"physical-table-{i}"
        table_type = draw(st.sampled_from(["RelationalTable", "CustomSql", "S3Source"]))

        # Use provided column names for first table, otherwise generate
        if column_names and i == 0:
            input_columns = [
                {"Name": name, "Type": draw(st_data_type())}
                for name in column_names
            ]
        else:
            num_cols = draw(st.integers(min_value=1, max_value=8))
            input_columns = []
            seen: set[str] = set()
            for _ in range(num_cols):
                name = draw(st_column_name())
                while name in seen:
                    name = draw(st_column_name())
                seen.add(name)
                input_columns.append({"Name": name, "Type": draw(st_data_type())})

        table_def: dict = {"InputColumns": input_columns}

        if table_type == "RelationalTable":
            table_def["DataSourceArn"] = draw(st_arn(resource_type="datasource"))
            table_def["Schema"] = "public"
            table_def["Name"] = f"table_{i}"
            physical_map[table_id] = {"RelationalTable": table_def}
        elif table_type == "CustomSql":
            table_def["DataSourceArn"] = draw(st_arn(resource_type="datasource"))
            table_def["SqlQuery"] = f"SELECT * FROM table_{i}"
            table_def["Name"] = f"custom_sql_{i}"
            physical_map[table_id] = {"CustomSql": table_def}
        else:
            table_def["DataSourceArn"] = draw(st_arn(resource_type="datasource"))
            physical_map[table_id] = {"S3Source": table_def}

    return physical_map


# ---------------------------------------------------------------------------
# Logical Table Map strategies
# ---------------------------------------------------------------------------


@st.composite
def st_join_instruction(
    draw: st.DrawFn,
    left_table: str,
    right_table: str,
    left_columns: list[str] | None = None,
    right_columns: list[str] | None = None,
) -> dict:
    """Generate a valid JoinInstruction.

    Args:
        left_table: Logical table ID for left side.
        right_table: Logical table ID for right side.
        left_columns: Available columns on left side for join keys.
        right_columns: Available columns on right side for join keys.
    """
    join_type = draw(st.sampled_from(["INNER", "LEFT", "RIGHT", "OUTER"]))

    # Pick join key columns
    if left_columns and right_columns:
        left_key = draw(st.sampled_from(left_columns))
        right_key = draw(st.sampled_from(right_columns))
    else:
        left_key = draw(st_column_name())
        right_key = draw(st_column_name())

    join_instruction = {
        "LeftOperand": left_table,
        "RightOperand": right_table,
        "Type": join_type,
        "OnClause": f"{left_key} = {right_key}",
        "LeftJoinKeyProperties": {"UniqueKey": draw(st.booleans())},
        "RightJoinKeyProperties": {"UniqueKey": draw(st.booleans())},
    }

    return join_instruction


@st.composite
def st_project_operation(draw: st.DrawFn, columns: list[str] | None = None) -> dict:
    """Generate a ProjectOperation transformation."""
    if columns is None:
        num_cols = draw(st.integers(min_value=1, max_value=10))
        columns = [draw(st_column_name()) for _ in range(num_cols)]
    return {"ProjectOperation": {"ProjectedColumns": columns}}


@st.composite
def st_logical_table_map(
    draw: st.DrawFn,
    physical_table_ids: list[str] | None = None,
    include_joins: bool | None = None,
    join_key_columns: list[str] | None = None,
) -> dict:
    """Generate a valid LogicalTableMap with optional JoinInstructions and transformations.

    Args:
        physical_table_ids: Physical table IDs to reference. Generates defaults if None.
        include_joins: Whether to include JoinInstructions. Random if None.
        join_key_columns: Specific columns to use as join keys.

    Returns:
        Dict representing a LogicalTableMap.
    """
    if physical_table_ids is None:
        num_physical = draw(st.integers(min_value=1, max_value=3))
        physical_table_ids = [f"physical-table-{i}" for i in range(num_physical)]

    if include_joins is None:
        include_joins = draw(st.booleans()) and len(physical_table_ids) > 1

    logical_map: dict = {}

    # Create a logical table for each physical table (source reference)
    for i, phys_id in enumerate(physical_table_ids):
        logical_id = f"logical-table-{i}"
        logical_table: dict = {
            "Alias": f"Table_{i}",
            "Source": {"PhysicalTableId": phys_id},
        }

        # Optionally add DataTransforms (excluding joins for now)
        if draw(st.booleans()):
            transforms = []
            # Optional RenameColumn
            if draw(st.booleans()):
                transforms.append({
                    "RenameColumnOperation": {
                        "ColumnName": draw(st_column_name()),
                        "NewColumnName": draw(st_column_name()),
                    }
                })
            # Optional CastColumnType
            if draw(st.booleans()):
                transforms.append({
                    "CastColumnTypeOperation": {
                        "ColumnName": draw(st_column_name()),
                        "NewColumnType": draw(st_data_type()),
                    }
                })
            if transforms:
                logical_table["DataTransforms"] = transforms

        logical_map[logical_id] = logical_table

    # Optionally create a join logical table
    if include_joins and len(physical_table_ids) >= 2:
        join_logical_id = f"logical-table-joined"
        left_table = "logical-table-0"
        right_table = "logical-table-1"

        left_cols = join_key_columns[:1] if join_key_columns else None
        right_cols = join_key_columns[1:2] if join_key_columns and len(join_key_columns) > 1 else None

        join_instr = draw(st_join_instruction(
            left_table=left_table,
            right_table=right_table,
            left_columns=left_cols,
            right_columns=right_cols,
        ))

        join_logical_table: dict = {
            "Alias": "JoinedTable",
            "Source": {"JoinInstruction": join_instr},
        }

        # Optionally add post-join DataTransforms
        if draw(st.booleans()):
            join_logical_table["DataTransforms"] = [
                draw(st_project_operation())
            ]

        logical_map[join_logical_id] = join_logical_table

    return logical_map


# ---------------------------------------------------------------------------
# Dataset Definition strategy (DescribeDataSet response)
# ---------------------------------------------------------------------------


@st.composite
def st_dataset_definition(
    draw: st.DrawFn,
    min_columns: int = 2,
    max_columns: int = 20,
    include_joins: bool | None = None,
    column_names: list[str] | None = None,
) -> dict:
    """Generate a valid DescribeDataSet response structure.

    Produces a structure matching what QuickSight's DescribeDataSet API returns,
    including OutputColumns, PhysicalTableMap, LogicalTableMap, and metadata.

    Args:
        min_columns: Minimum number of output columns.
        max_columns: Maximum number of output columns.
        include_joins: Whether to include JoinInstructions in LogicalTableMap.
        column_names: If provided, use these as the column names.
    """
    dataset_id = draw(st_dataset_id())
    dataset_name = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + " _-",
            min_size=3,
            max_size=30,
        )
    )

    # Generate output columns
    output_columns = draw(
        st_output_columns(
            min_columns=min_columns,
            max_columns=max_columns,
            column_names=column_names,
        )
    )
    col_names = [c["Name"] for c in output_columns]

    # Generate physical table map
    physical_table_map = draw(st_physical_table_map(column_names=col_names))
    physical_ids = list(physical_table_map.keys())

    # Generate logical table map
    logical_table_map = draw(
        st_logical_table_map(
            physical_table_ids=physical_ids,
            include_joins=include_joins,
        )
    )

    # Optionally add dataset-level calculated fields
    calc_fields: list[dict] = []
    if draw(st.booleans()) and len(col_names) >= 2:
        num_calc = draw(st.integers(min_value=1, max_value=3))
        for j in range(num_calc):
            source_col = draw(st.sampled_from(col_names))
            calc_fields.append({
                "DataSetId": dataset_id,
                "Name": f"calc_field_{j}",
                "Expression": f"ifelse({{[{source_col}]}} > 0, 1, 0)",
            })

    dataset_def: dict = {
        "DataSetId": dataset_id,
        "Name": dataset_name,
        "Arn": f"arn:aws:quicksight:us-east-1:123456789012:dataset/{dataset_id}",
        "ImportMode": draw(st.sampled_from(["SPICE", "DIRECT_QUERY"])),
        "OutputColumns": output_columns,
        "PhysicalTableMap": physical_table_map,
        "LogicalTableMap": logical_table_map,
    }

    if calc_fields:
        dataset_def["CalculatedFields"] = calc_fields

    # Optional RowLevelPermissionDataSet
    if draw(st.booleans()):
        rls_col = draw(st.sampled_from(col_names))
        dataset_def["RowLevelPermissionDataSet"] = {
            "Arn": draw(st_arn(resource_type="dataset")),
            "PermissionPolicy": "GRANT_ACCESS",
            "FormatVersion": "VERSION_2",
        }

    # Optional ColumnLevelPermissionRules
    if draw(st.booleans()):
        cls_cols = draw(
            st.lists(
                st.sampled_from(col_names),
                min_size=1,
                max_size=min(3, len(col_names)),
                unique=True,
            )
        )
        dataset_def["ColumnLevelPermissionRules"] = [
            {
                "Principals": [draw(st_arn(resource_type="user"))],
                "ColumnNames": cls_cols,
            }
        ]

    return dataset_def


# ---------------------------------------------------------------------------
# Visual and sheet strategies for analysis/dashboard definitions
# ---------------------------------------------------------------------------


@st.composite
def st_field_well_column(draw: st.DrawFn, dataset_identifier: str, column_names: list[str]) -> dict:
    """Generate a column reference used in a visual field well."""
    col_name = draw(st.sampled_from(column_names))
    return {
        "Column": {
            "DataSetIdentifier": dataset_identifier,
            "ColumnName": col_name,
        }
    }


@st.composite
def st_visual(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a visual structure with field wells referencing dataset columns.

    Generates a table/bar/line visual with dimensions and measures.
    """
    visual_id = draw(st.uuids().map(str))
    visual_type = draw(st.sampled_from([
        "TableVisual", "BarChartVisual", "LineChartVisual", "PieChartVisual",
    ]))

    # Generate field wells with dimension and measure columns
    num_dimensions = draw(st.integers(min_value=1, max_value=min(4, len(column_names))))
    num_measures = draw(st.integers(min_value=0, max_value=min(3, len(column_names))))

    dimension_cols = draw(
        st.lists(
            st.sampled_from(column_names),
            min_size=num_dimensions,
            max_size=num_dimensions,
        )
    )
    measure_cols = draw(
        st.lists(
            st.sampled_from(column_names),
            min_size=num_measures,
            max_size=num_measures,
        )
    )

    dimensions = [
        {
            "CategoricalDimensionField": {
                "FieldId": f"dim_{i}",
                "Column": {
                    "DataSetIdentifier": dataset_identifier,
                    "ColumnName": col,
                },
            }
        }
        for i, col in enumerate(dimension_cols)
    ]

    measures = [
        {
            "NumericalMeasureField": {
                "FieldId": f"measure_{i}",
                "Column": {
                    "DataSetIdentifier": dataset_identifier,
                    "ColumnName": col,
                },
                "AggregationFunction": {"SimpleNumericalAggregation": "SUM"},
            }
        }
        for i, col in enumerate(measure_cols)
    ]

    # Build visual structure based on type
    field_wells: dict = {}
    if visual_type == "TableVisual":
        field_wells = {
            "TableAggregatedFieldWells": {
                "GroupBy": dimensions,
                "Values": measures,
            }
        }
    elif visual_type == "BarChartVisual":
        field_wells = {
            "BarChartAggregatedFieldWells": {
                "Category": dimensions,
                "Values": measures,
            }
        }
    elif visual_type == "LineChartVisual":
        field_wells = {
            "LineChartAggregatedFieldWells": {
                "Category": dimensions,
                "Values": measures,
            }
        }
    else:  # PieChartVisual
        field_wells = {
            "PieChartAggregatedFieldWells": {
                "Category": dimensions,
                "Values": measures,
            }
        }

    visual: dict = {
        f"{visual_type}": {
            "VisualId": visual_id,
            "ChartConfiguration": {
                "FieldWells": field_wells,
            },
        }
    }

    # Optionally add conditional formatting
    if draw(st.booleans()) and column_names:
        cond_col = draw(st.sampled_from(column_names))
        visual[visual_type]["ConditionalFormatting"] = {
            "ConditionalFormattingOptions": [
                {
                    "Cell": {
                        "FieldId": "cond_field_1",
                        "TextFormat": {
                            "BackgroundColor": {
                                "Solid": {
                                    "Expression": f"IF({{{dataset_identifier}.{cond_col}}} > 100, 'green', 'red')",
                                }
                            }
                        },
                    }
                }
            ]
        }

    # Optionally add tooltip columns
    if draw(st.booleans()) and column_names:
        tooltip_col = draw(st.sampled_from(column_names))
        visual[visual_type]["ChartConfiguration"]["Tooltip"] = {
            "TooltipVisibility": "VISIBLE",
            "SelectedTooltipType": "DETAILED",
            "FieldBasedTooltip": {
                "TooltipFields": [
                    {
                        "FieldTooltipItem": {
                            "FieldId": "tooltip_1",
                            "Column": {
                                "DataSetIdentifier": dataset_identifier,
                                "ColumnName": tooltip_col,
                            },
                        }
                    }
                ]
            },
        }

    return visual


@st.composite
def st_calculated_field_expression(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> str:
    """Generate a calculated field expression referencing dataset columns.

    Format: uses {DataSetIdentifier.ColumnName} tokens.
    """
    num_refs = draw(st.integers(min_value=1, max_value=min(3, len(column_names))))
    cols = draw(
        st.lists(
            st.sampled_from(column_names),
            min_size=num_refs,
            max_size=num_refs,
        )
    )

    # Build expression using column references
    func = draw(st.sampled_from(["ifelse", "sum", "avg", "concat", "strlen"]))
    col_refs = [f"{{{dataset_identifier}.{col}}}" for col in cols]

    if func == "ifelse" and len(col_refs) >= 1:
        return f"ifelse({col_refs[0]} > 0, 1, 0)"
    elif func == "concat" and len(col_refs) >= 2:
        return f"concat({col_refs[0]}, ' ', {col_refs[1]})"
    elif func in ("sum", "avg"):
        return f"{func}({col_refs[0]})"
    else:
        return f"strlen({col_refs[0]})"


@st.composite
def st_calculated_field(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a calculated field definition for an analysis/dashboard."""
    calc_name = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=3,
            max_size=20,
        )
    )
    expression = draw(
        st_calculated_field_expression(
            dataset_identifier=dataset_identifier,
            column_names=column_names,
        )
    )
    return {
        "DataSetIdentifier": dataset_identifier,
        "Name": calc_name,
        "Expression": expression,
    }


@st.composite
def st_filter(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a filter configuration referencing a dataset column."""
    filter_col = draw(st.sampled_from(column_names))
    filter_type = draw(st.sampled_from([
        "CategoryFilter",
        "NumericRangeFilter",
        "NumericEqualityFilter",
        "TimeRangeFilter",
        "RelativeDateTimeFilter",
        "TopBottomFilter",
    ]))

    filter_id = draw(st.uuids().map(str))

    base_filter: dict = {
        "FilterId": filter_id,
        "Column": {
            "DataSetIdentifier": dataset_identifier,
            "ColumnName": filter_col,
        },
    }

    if filter_type == "CategoryFilter":
        base_filter["Configuration"] = {
            "FilterListConfiguration": {
                "MatchOperator": "CONTAINS",
                "CategoryValues": ["value1", "value2"],
            }
        }
        return {"CategoryFilter": base_filter}
    elif filter_type == "NumericRangeFilter":
        base_filter["RangeMinimum"] = {"StaticValue": 0.0}
        base_filter["RangeMaximum"] = {"StaticValue": 100.0}
        base_filter["NullOption"] = "ALL_VALUES"
        return {"NumericRangeFilter": base_filter}
    elif filter_type == "NumericEqualityFilter":
        base_filter["Value"] = 42.0
        base_filter["MatchOperator"] = "EQUALS"
        base_filter["NullOption"] = "ALL_VALUES"
        return {"NumericEqualityFilter": base_filter}
    elif filter_type == "TimeRangeFilter":
        base_filter["NullOption"] = "ALL_VALUES"
        base_filter["RangeMinimumValue"] = {"StaticValue": "2023-01-01T00:00:00Z"}
        base_filter["RangeMaximumValue"] = {"StaticValue": "2024-01-01T00:00:00Z"}
        return {"TimeRangeFilter": base_filter}
    elif filter_type == "RelativeDateTimeFilter":
        base_filter["NullOption"] = "ALL_VALUES"
        base_filter["RelativeDateType"] = "LAST"
        base_filter["TimeGranularity"] = "DAY"
        base_filter["RelativeDateValue"] = 30
        return {"RelativeDateTimeFilter": base_filter}
    else:  # TopBottomFilter
        base_filter["Limit"] = 10
        base_filter["AggregationSortConfigurations"] = [{
            "Column": {
                "DataSetIdentifier": dataset_identifier,
                "ColumnName": filter_col,
            },
            "SortDirection": "DESC",
            "AggregationFunction": {"NumericalAggregationFunction": {"SimpleNumericalAggregation": "SUM"}},
        }]
        return {"TopBottomFilter": base_filter}


@st.composite
def st_filter_group(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a FilterGroup containing one or more filters."""
    num_filters = draw(st.integers(min_value=1, max_value=3))
    filters = [
        draw(st_filter(dataset_identifier=dataset_identifier, column_names=column_names))
        for _ in range(num_filters)
    ]
    return {
        "FilterGroupId": draw(st.uuids().map(str)),
        "Filters": filters,
        "ScopeConfiguration": {
            "SelectedSheets": {
                "SheetVisualScopingConfigurations": [
                    {
                        "SheetId": "sheet_1",
                        "Scope": "ALL_VISUALS",
                    }
                ]
            }
        },
        "CrossDataset": "ALL_DATASETS",
    }


@st.composite
def st_parameter_declaration(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a parameter declaration with optional dynamic default from a dataset column."""
    param_name = draw(
        st.text(
            alphabet=string.ascii_letters + string.digits + "_",
            min_size=3,
            max_size=15,
        )
    )
    param_type = draw(st.sampled_from([
        "StringParameterDeclaration",
        "IntegerParameterDeclaration",
        "DecimalParameterDeclaration",
        "DateTimeParameterDeclaration",
    ]))

    param_def: dict = {
        "Name": param_name,
        "ParameterValueType": "SINGLE_VALUED",
    }

    # Add dynamic default referencing a column
    if draw(st.booleans()) and column_names:
        default_col = draw(st.sampled_from(column_names))
        param_def["DefaultValues"] = {
            "DynamicValue": {
                "DefaultValueColumn": {
                    "DataSetIdentifier": dataset_identifier,
                    "ColumnName": default_col,
                },
                "UserNameColumn": {
                    "DataSetIdentifier": dataset_identifier,
                    "ColumnName": default_col,
                },
                "GroupNameColumn": {
                    "DataSetIdentifier": dataset_identifier,
                    "ColumnName": default_col,
                },
            }
        }
    else:
        if param_type == "StringParameterDeclaration":
            param_def["DefaultValues"] = {"StaticValues": ["default"]}
        elif param_type == "IntegerParameterDeclaration":
            param_def["DefaultValues"] = {"StaticValues": [0]}
        elif param_type == "DecimalParameterDeclaration":
            param_def["DefaultValues"] = {"StaticValues": [0.0]}
        else:
            param_def["DefaultValues"] = {"StaticValues": ["2024-01-01T00:00:00Z"]}

    return {param_type: param_def}


@st.composite
def st_drill_down_hierarchy(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a drill-down hierarchy referencing dataset columns."""
    num_cols = draw(st.integers(min_value=1, max_value=min(3, len(column_names))))
    cols = draw(
        st.lists(
            st.sampled_from(column_names),
            min_size=num_cols,
            max_size=num_cols,
        )
    )
    return {
        "HierarchyId": draw(st.uuids().map(str)),
        "Columns": [
            {
                "DataSetIdentifier": dataset_identifier,
                "ColumnName": col,
            }
            for col in cols
        ],
    }


@st.composite
def st_sheet(
    draw: st.DrawFn,
    dataset_identifier: str,
    column_names: list[str],
) -> dict:
    """Generate a sheet with visuals, optional drill-downs."""
    sheet_id = draw(st.uuids().map(str))
    num_visuals = draw(st.integers(min_value=1, max_value=4))

    visuals = [
        draw(st_visual(dataset_identifier=dataset_identifier, column_names=column_names))
        for _ in range(num_visuals)
    ]

    sheet: dict = {
        "SheetId": sheet_id,
        "Name": f"Sheet_{draw(st.integers(min_value=1, max_value=99))}",
        "Visuals": visuals,
    }

    # Optionally add sheet-level filter controls
    if draw(st.booleans()) and column_names:
        filter_col = draw(st.sampled_from(column_names))
        sheet["FilterControls"] = [
            {
                "Dropdown": {
                    "FilterControlId": "ctrl_1",
                    "SourceFilterId": "filter_1",
                    "Title": "Filter Control",
                    "SelectableValues": {
                        "Column": {
                            "DataSetIdentifier": dataset_identifier,
                            "ColumnName": filter_col,
                        }
                    },
                }
            }
        ]

    return sheet


# ---------------------------------------------------------------------------
# Analysis/Dashboard Definition strategy
# ---------------------------------------------------------------------------


@st.composite
def st_analysis_definition(
    draw: st.DrawFn,
    dataset_identifier: str | None = None,
    column_names: list[str] | None = None,
    min_columns: int = 3,
    max_columns: int = 15,
    min_sheets: int = 1,
    max_sheets: int = 3,
) -> dict:
    """Generate a valid analysis/dashboard definition.

    Produces a structure matching what QuickSight's DescribeAnalysisDefinition
    or DescribeDashboardDefinition returns, including:
    - DataSetIdentifierDeclarations
    - Sheets with Visuals (field wells with dimensions/measures)
    - CalculatedFields with expressions using {DataSetId.ColumnName} format
    - FilterGroups with filters referencing columns
    - ParameterDeclarations with dynamic defaults

    Args:
        dataset_identifier: The DataSetIdentifier to use. Generates one if None.
        column_names: Column names to reference. Generates if None.
        min_columns: Minimum columns to generate if column_names not provided.
        max_columns: Maximum columns to generate if column_names not provided.
        min_sheets: Minimum number of sheets.
        max_sheets: Maximum number of sheets.
    """
    if dataset_identifier is None:
        dataset_identifier = draw(st_dataset_identifier())

    if column_names is None:
        num_cols = draw(st.integers(min_value=min_columns, max_value=max_columns))
        names_set: set[str] = set()
        while len(names_set) < num_cols:
            names_set.add(draw(st_column_name()))
        column_names = list(names_set)

    dataset_id = draw(st_dataset_id())

    # DataSetIdentifierDeclarations
    dataset_declarations = [
        {
            "Identifier": dataset_identifier,
            "DataSetArn": f"arn:aws:quicksight:us-east-1:123456789012:dataset/{dataset_id}",
        }
    ]

    # Optionally add additional dataset declarations (for multi-dataset testing)
    if draw(st.booleans()):
        extra_ds_id = draw(st_dataset_identifier())
        while extra_ds_id == dataset_identifier:
            extra_ds_id = draw(st_dataset_identifier())
        dataset_declarations.append({
            "Identifier": extra_ds_id,
            "DataSetArn": draw(st_arn(resource_type="dataset")),
        })

    # Sheets
    num_sheets = draw(st.integers(min_value=min_sheets, max_value=max_sheets))
    sheets = [
        draw(st_sheet(dataset_identifier=dataset_identifier, column_names=column_names))
        for _ in range(num_sheets)
    ]

    definition: dict = {
        "DataSetIdentifierDeclarations": dataset_declarations,
        "Sheets": sheets,
    }

    # Optionally add calculated fields
    if draw(st.booleans()) and column_names:
        num_calc = draw(st.integers(min_value=1, max_value=3))
        calc_fields = [
            draw(st_calculated_field(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            ))
            for _ in range(num_calc)
        ]
        definition["CalculatedFields"] = calc_fields

    # Optionally add filter groups
    if draw(st.booleans()) and column_names:
        num_filter_groups = draw(st.integers(min_value=1, max_value=2))
        filter_groups = [
            draw(st_filter_group(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            ))
            for _ in range(num_filter_groups)
        ]
        definition["FilterGroups"] = filter_groups

    # Optionally add parameter declarations
    if draw(st.booleans()) and column_names:
        num_params = draw(st.integers(min_value=1, max_value=3))
        params = [
            draw(st_parameter_declaration(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            ))
            for _ in range(num_params)
        ]
        definition["ParameterDeclarations"] = params

    # Optionally add column hierarchies (drill-downs at analysis level)
    if draw(st.booleans()) and column_names:
        hierarchies = [
            draw(st_drill_down_hierarchy(
                dataset_identifier=dataset_identifier,
                column_names=column_names,
            ))
        ]
        definition["ColumnHierarchies"] = hierarchies

    return definition


@st.composite
def st_dashboard_definition(
    draw: st.DrawFn,
    dataset_identifier: str | None = None,
    column_names: list[str] | None = None,
) -> dict:
    """Generate a valid dashboard definition (same structure as analysis definition).

    Dashboards and analyses share the same definition structure in QuickSight.
    """
    return draw(
        st_analysis_definition(
            dataset_identifier=dataset_identifier,
            column_names=column_names,
        )
    )


# ---------------------------------------------------------------------------
# Consumer strategy
# ---------------------------------------------------------------------------


@st.composite
def st_consumer(
    draw: st.DrawFn,
    dataset_identifier: str | None = None,
    status: str | None = None,
) -> dict:
    """Generate a Consumer dict matching the Consumer dataclass fields."""
    if dataset_identifier is None:
        dataset_identifier = draw(st_dataset_identifier())
    if status is None:
        status = draw(st.sampled_from(QS_CONSUMER_STATUSES))

    consumer_type = draw(st.sampled_from(["analysis", "dashboard"]))
    return {
        "arn": draw(st_arn(resource_type=consumer_type)),
        "consumer_type": consumer_type,
        "name": draw(
            st.text(
                alphabet=string.ascii_letters + string.digits + " _-",
                min_size=3,
                max_size=30,
            )
        ),
        "dataset_identifier": dataset_identifier,
        "status": status,
    }


# ---------------------------------------------------------------------------
# OptimiserConfig strategy
# ---------------------------------------------------------------------------


@st.composite
def st_config(draw: st.DrawFn) -> OptimiserConfig:
    """Generate a valid OptimiserConfig instance with proper field ranges.

    Generates configs that would pass validation:
    - safety_threshold_pct in [0.0, 1.0]
    - execution_mode is 'dry_run' or 'apply'
    - Required fields are populated
    """
    num_datasets = draw(st.integers(min_value=1, max_value=5))
    target_datasets = [draw(st_dataset_id()) for _ in range(num_datasets)]

    execution_mode = draw(st.sampled_from([mode.value for mode in ExecutionMode]))
    safety_threshold = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    quarantine_days = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=1, max_value=90),
        )
    )

    s3_bucket = draw(st_s3_bucket_name())
    s3_prefix = draw(
        st.sampled_from(["spice-optimiser/", "reports/", "data/", ""])
    )

    sns_topic_arn = draw(
        st.one_of(
            st.none(),
            st_arn(resource_type="sns"),
        )
    )

    dynamodb_table = draw(
        st.one_of(
            st.none(),
            st.text(
                alphabet=string.ascii_letters + string.digits + "_-",
                min_size=3,
                max_size=30,
            ),
        )
    )

    aws_account_id = draw(st_aws_account_id())
    aws_region = draw(st_aws_region())

    return OptimiserConfig(
        target_datasets=target_datasets,
        execution_mode=execution_mode,
        safety_threshold_pct=safety_threshold,
        quarantine_period_days=quarantine_days,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        sns_topic_arn=sns_topic_arn,
        dynamodb_table=dynamodb_table,
        aws_account_id=aws_account_id,
        aws_region=aws_region,
    )


# ---------------------------------------------------------------------------
# Utility strategies for specific test scenarios
# ---------------------------------------------------------------------------


@st.composite
def st_column_classification_inputs(
    draw: st.DrawFn,
    min_total: int = 3,
    max_total: int = 30,
) -> dict:
    """Generate inputs for column classification testing.

    Returns a dict with:
    - all_columns: list of {Name, Type} dicts
    - used_columns: set of column names marked as used
    - join_key_columns: set of column names that are join keys
    - calc_field_inputs: set of column names used in calc fields
    """
    num_total = draw(st.integers(min_value=min_total, max_value=max_total))

    # Generate unique column names
    all_names: list[str] = []
    seen: set[str] = set()
    while len(all_names) < num_total:
        name = draw(st_column_name())
        if name not in seen:
            seen.add(name)
            all_names.append(name)

    all_columns = [
        {"Name": name, "Type": draw(st_data_type())}
        for name in all_names
    ]

    # Partition columns into used/join_key/calc subsets (with possible overlap)
    used_columns = set(
        draw(
            st.lists(
                st.sampled_from(all_names),
                min_size=0,
                max_size=num_total,
                unique=True,
            )
        )
    )
    join_key_columns = set(
        draw(
            st.lists(
                st.sampled_from(all_names),
                min_size=0,
                max_size=min(4, num_total),
                unique=True,
            )
        )
    )
    calc_field_inputs = set(
        draw(
            st.lists(
                st.sampled_from(all_names),
                min_size=0,
                max_size=min(3, num_total),
                unique=True,
            )
        )
    )

    return {
        "all_columns": all_columns,
        "used_columns": used_columns,
        "join_key_columns": join_key_columns,
        "calc_field_inputs": calc_field_inputs,
    }


@st.composite
def st_timestamp_iso8601(draw: st.DrawFn) -> str:
    """Generate a valid ISO-8601 timestamp string."""
    year = draw(st.integers(min_value=2020, max_value=2025))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))
    hour = draw(st.integers(min_value=0, max_value=23))
    minute = draw(st.integers(min_value=0, max_value=59))
    second = draw(st.integers(min_value=0, max_value=59))
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
