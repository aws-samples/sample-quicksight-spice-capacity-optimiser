# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Single builder for `UpdateDataSet` parameters.

`snapshot.py` and `modifier.py` each had their own `_build_update_params`, and they had
drifted. The snapshot copy omitted two fields the modifier copy handled:

    DataPrepConfiguration       every transform a new-experience dataset has -
                                joins, casts, projections
    SemanticModelConfiguration  present on current API responses
    PerformanceConfiguration

`UpdateDataSet` is a full replace, so a field left out of the request is removed from the
dataset. Rolling back a new-experience dataset would therefore have restored its columns
and destroyed its entire transform chain. 11 of 66 datasets in the account this was first run against are
new-experience, and `rollback_dataset()` had no callers, so nothing ever exercised it.

Exactly the shape of the braced-`OnClause` defect: one concept, two implementations, one
of them quietly behind. Hence one function, imported by both.
"""

from __future__ import annotations

from typing import Any

# Copied verbatim from a DescribeDataSet response when present. Order is irrelevant to the
# API; it is grouped here to make review against the API reference straightforward.
#
# Anything omitted from this list is silently dropped from the dataset on update, because
# UpdateDataSet replaces rather than merges. Adding a field the API later introduces is
# the maintenance burden this list carries.
_PASSTHROUGH_FIELDS = (
    "Name",
    "PhysicalTableMap",
    "ImportMode",
    "ColumnGroups",
    "FieldFolders",
    "RowLevelPermissionDataSet",
    "RowLevelPermissionTagConfiguration",
    "ColumnLevelPermissionRules",
    "DataSetUsageConfiguration",
    "DatasetParameters",
    "PerformanceConfiguration",
    "SemanticModelConfiguration",
)


def build_update_params(dataset_def: dict) -> dict[str, Any]:
    """Build `UpdateDataSet` parameters from a dataset definition.

    Args:
        dataset_def: A `DescribeDataSet` response, or a modified copy of one.

    Returns:
        Parameters suitable for `QSClient.update_dataset`.
    """
    params: dict[str, Any] = {
        field: dataset_def[field]
        for field in _PASSTHROUGH_FIELDS
        if field in dataset_def
    }

    # The two preparation experiences are mutually exclusive, and which one a dataset uses
    # is decided by whether LogicalTableMap is populated. A new-experience dataset returns
    # LogicalTableMap as None rather than omitting it, so a truthiness check is required —
    # `"LogicalTableMap" in dataset_def` would take the legacy branch and send None.
    if dataset_def.get("LogicalTableMap"):
        params["LogicalTableMap"] = dataset_def["LogicalTableMap"]
    elif dataset_def.get("DataPrepConfiguration"):
        params["DataPrepConfiguration"] = dataset_def["DataPrepConfiguration"]

    return params
