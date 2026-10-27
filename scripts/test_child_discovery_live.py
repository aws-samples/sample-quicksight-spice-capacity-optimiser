# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Live integration test for ChildDatasetDiscovery against a real AWS account.

Usage:
    python3 scripts/test_child_discovery_live.py [target_dataset_id]

If no target_dataset_id is provided, the script will find a parent dataset
(one that has children) automatically, or use the first dataset available.

Environment:
    Requires valid AWS credentials with QuickSight read permissions.
    Set ACCOUNT_ID and REGION below or via environment variables.
"""

import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lambda_source.qs_client import QSClient
from lambda_source.child_discovery import ChildDatasetDiscovery, ChildDiscoveryError

ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")  # required: the account that owns the datasets
REGION = os.environ.get("AWS_REGION", "us-east-1")


def main():
    if not ACCOUNT_ID:
        sys.exit("Set AWS_ACCOUNT_ID to the account that owns the datasets before running this live test.")
    print("=" * 70)
    print("LIVE TEST: ChildDatasetDiscovery")
    print(f"Account: {ACCOUNT_ID} | Region: {REGION}")
    print("=" * 70)

    qs_client = QSClient(account_id=ACCOUNT_ID, region=REGION)
    child_discovery = ChildDatasetDiscovery(
        qs_client=qs_client,
        account_id=ACCOUNT_ID,
        region=REGION,
    )

    # List all datasets
    print("\n[1] Listing all datasets...")
    start = time.time()
    all_datasets = qs_client.list_datasets()
    print(f"    Found {len(all_datasets)} datasets in {time.time() - start:.2f}s")

    # Scan for parent-child relationships
    print("\n[2] Scanning for parent-child relationships...")
    parent_child_map: dict[str, list[str]] = {}

    start = time.time()
    for ds in all_datasets:
        ds_id = ds.get("DataSetId", "")
        if not ds_id:
            continue
        try:
            ds_def = qs_client.describe_dataset(ds_id)
        except Exception as e:
            print(f"    SKIP {ds_id}: {e}")
            continue

        logical_map = ds_def.get("LogicalTableMap", {})
        for table_def in logical_map.values():
            source = table_def.get("Source", {}) if isinstance(table_def, dict) else {}
            ds_arn = source.get("DataSetArn", "")
            if ds_arn and "dataset/" in ds_arn:
                parent_id = ds_arn.split("dataset/")[-1]
                parent_child_map.setdefault(parent_id, []).append(ds_id)

        data_prep = ds_def.get("DataPrepConfiguration", {})
        for src_table in data_prep.get("SourceTableMap", {}).values():
            if not isinstance(src_table, dict):
                continue
            ds_source = src_table.get("DataSet", {})
            if not isinstance(ds_source, dict):
                continue
            ds_arn = ds_source.get("DataSetArn", "")
            if ds_arn and "dataset/" in ds_arn:
                parent_id = ds_arn.split("dataset/")[-1]
                parent_child_map.setdefault(parent_id, []).append(ds_id)

    print(f"    Scan complete in {time.time() - start:.2f}s")
    print(f"    Found {len(parent_child_map)} parent dataset(s) with children:")
    for parent_id, children in parent_child_map.items():
        name = next((d.get("Name", "?") for d in all_datasets if d.get("DataSetId") == parent_id), "?")
        print(f"      {name} ({parent_id}) → {len(children)} child(ren)")

    # Determine target
    if len(sys.argv) > 1:
        target_dataset_id = sys.argv[1]
    elif parent_child_map:
        target_dataset_id = list(parent_child_map.keys())[0]
    else:
        target_dataset_id = all_datasets[0]["DataSetId"] if all_datasets else None

    if not target_dataset_id:
        print("\n[ERROR] No datasets found.")
        return

    target_name = next(
        (d.get("Name", "?") for d in all_datasets if d.get("DataSetId") == target_dataset_id), "?"
    )

    # Run discovery
    print(f"\n[3] Running ChildDatasetDiscovery for: {target_name} ({target_dataset_id})")
    start = time.time()
    try:
        children = child_discovery.discover_child_datasets(
            target_dataset_id=target_dataset_id,
            skip_unreachable=True,
        )
    except ChildDiscoveryError as e:
        print(f"    ERROR: {e}")
        return

    print(f"\n[4] Results ({time.time() - start:.2f}s):")
    print(f"    Children found: {len(children)}")

    if children:
        all_refs = child_discovery.get_all_child_referenced_columns(children)
        print(f"    Total unique referenced columns: {len(all_refs)}")
        for child in children:
            print(f"\n    Child: {child.name} ({child.dataset_id})")
            if child.referenced_columns:
                for col in sorted(child.referenced_columns):
                    print(f"      - {col}")
            else:
                print("      (unknown — conservative fallback applies)")

        if all_refs:
            print(f"\n    Columns to protect: {sorted(all_refs)}")
        else:
            print("\n    All columns must be protected (unknown usage)")
    else:
        print("    No child datasets found.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
