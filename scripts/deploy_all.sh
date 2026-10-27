#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# One-step deployment for the QuickSight SPICE Capacity Optimiser.
#
# Architecture:
#   Amazon Q Chat Agent → Cognito (2LO) → AgentCore Gateway → Lambda (direct)
#
# Runs all phases in the correct order:
#   1. Package the Lambda (lambda_source/) and upload it to the code bucket.
#   2. Deploy/update the CloudFormation stack (Lambda, S3, DynamoDB, Cognito,
#      SNS, EventBridge, alarms).
#   3. Force the Lambda to the freshly-uploaded code (handles same-key updates).
#   4. Sync AgentCore config from the stack outputs and deploy the gateway
#      (delegates to deploy_mcp.sh).
#
# Prerequisites:
#   - AWS credentials configured (env vars or --profile).
#   - The code bucket (--code-bucket) must already exist.
#   - Node.js 20+ on PATH and the AgentCore CLI: npm install -g @aws/agentcore
#   - Python 3 with boto3, and the zip utility.
#
# Usage:
#   scripts/deploy_all.sh \
#     --code-bucket <bucket-for-lambda-zip> \
#     [--stack-name spice-optimiser] [--region us-east-1] [--profile NAME] \
#     [--code-key spice-optimiser/lambda-package.zip]

set -euo pipefail

STACK_NAME="spice-optimiser"
REGION="us-east-1"
PROFILE=""
CODE_BUCKET=""
CODE_KEY="spice-optimiser/lambda-package.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stack-name)  STACK_NAME="$2"; shift 2 ;;
    --region)      REGION="$2"; shift 2 ;;
    --profile)     PROFILE="$2"; shift 2 ;;
    --code-bucket) CODE_BUCKET="$2"; shift 2 ;;
    --code-key)    CODE_KEY="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CODE_BUCKET" ]]; then
  echo "ERROR: --code-bucket is required (an existing S3 bucket for the Lambda zip)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# AWS CLI profile flag (empty when using ambient/env credentials).
AWS_PROFILE_ARGS=()
if [[ -n "$PROFILE" ]]; then
  AWS_PROFILE_ARGS=(--profile "$PROFILE")
fi

FUNCTION_NAME="${STACK_NAME}-function"
ZIP_PATH="$REPO_DIR/lambda-package.zip"

echo "==> [1/4] Packaging Lambda from lambda_source/"
rm -f "$ZIP_PATH"
# Install boto3 into a temp dir and bundle with lambda_source
TEMP_PKG_DIR=$(mktemp -d)
pip3 install --target "$TEMP_PKG_DIR" boto3 --upgrade -q
cp -r "$REPO_DIR/lambda_source" "$TEMP_PKG_DIR/"
( cd "$TEMP_PKG_DIR" && zip -rq "$ZIP_PATH" . -x "*/__pycache__/*" "*.pyc" "*.dist-info/*" )
rm -rf "$TEMP_PKG_DIR"
# The code bucket name is operator-supplied. Pinning the upload to the deploying
# account's ownership means a mistyped or squatted bucket name fails here, rather than
# handing the deployment package (and then the Lambda's code source) to someone else.
# s3api put-object rather than s3 cp: only the former accepts --expected-bucket-owner on
# every AWS CLI version in use (v1 rejects it on s3 cp), and the package is far below the
# 5 GB single-PUT limit.
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text "${AWS_PROFILE_ARGS[@]}")"
aws s3api put-object --bucket "$CODE_BUCKET" --key "$CODE_KEY" --body "$ZIP_PATH" \
  --expected-bucket-owner "$ACCOUNT_ID" \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}" >/dev/null
echo "    uploaded s3://${CODE_BUCKET}/${CODE_KEY} (owner check passed)"

echo "==> [2/4] Deploying CloudFormation stack '$STACK_NAME'"
aws cloudformation deploy \
  --stack-name "$STACK_NAME" \
  --template-file "$REPO_DIR/deployment/template.yaml" \
  --capabilities CAPABILITY_NAMED_IAM \
  --no-fail-on-empty-changeset \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}" \
  --parameter-overrides \
      "LambdaCodeS3Bucket=${CODE_BUCKET}" \
      "LambdaCodeS3Key=${CODE_KEY}"

echo "==> [3/4] Forcing Lambda to the latest code"
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --s3-bucket "$CODE_BUCKET" \
  --s3-key "$CODE_KEY" \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}" >/dev/null
aws lambda wait function-updated \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION" "${AWS_PROFILE_ARGS[@]}"

echo "==> [4/4] Deploying AgentCore Gateway"
MCP_ARGS=(--stack-name "$STACK_NAME" --region "$REGION")
if [[ -n "$PROFILE" ]]; then
  MCP_ARGS+=(--profile "$PROFILE")
fi
"$SCRIPT_DIR/deploy_mcp.sh" "${MCP_ARGS[@]}"

rm -f "$ZIP_PATH"
echo "==> All done."
echo "    - Lambda deployed (CloudFormation stack: $STACK_NAME)"
echo "    - AgentCore Gateway deployed (invokes Lambda directly)"
echo "    - Amazon Q chat agent can reach tools via gateway 'spice-optimiser-gateway'"
