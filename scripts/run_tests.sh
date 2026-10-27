#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Runs the full test suite the way CI should run it.
#
# Kept as a script rather than inlined into a pipeline definition so the same command
# works locally and in CI, and so there is one place to change when the suite changes.
#
# Usage:
#   scripts/run_tests.sh              # full suite
#   scripts/run_tests.sh --coverage   # full suite plus a coverage report
#
# Requires: pip install -e ".[dev]"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

COVERAGE=0
if [[ "${1:-}" == "--coverage" ]]; then
  COVERAGE=1
fi

echo "==> Verifying the SDK floor"
# An older SDK silently mis-reads file-upload and new-experience datasets, so a test run
# against one proves less than it appears to. See check_boto3_version in qs_client.py.
python3 - <<'PY'
import sys
from lambda_source.qs_client import check_boto3_version
warning = check_boto3_version()
if warning:
    print(f"WARNING: {warning}", file=sys.stderr)
PY

echo "==> Running the test suite"
if [[ "$COVERAGE" -eq 1 ]]; then
  python3 -m pytest tests/ \
    --cov=lambda_source \
    --cov-report=term-missing \
    --cov-report=xml:coverage.xml
else
  python3 -m pytest tests/ -q
fi

echo "==> All tests passed"
