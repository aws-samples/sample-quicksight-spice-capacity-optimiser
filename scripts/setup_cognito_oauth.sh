#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Set up Amazon Cognito for service-to-service OAuth (2LO / client_credentials)
# between the Amazon Quick MCP client and the AgentCore Gateway.
#
# Architecture:
#   Amazon Quick Chat Agent
#     → OAuth 2.0 client_credentials grant (2LO) → Cognito token endpoint
#     → Bearer JWT → AgentCore Gateway (CUSTOM_JWT authorizer)
#     → MCP Server → Lambda Pipeline
#
# NOT needed for a standard deployment. deployment/template.yaml already creates the
# user pool, domain, resource server and app client when CognitoEnabled=true (the
# default), and README step 2 shows how to read the client ID, secret and token URL
# from the stack. Running this as well used to create a second pool and a second
# client per run, which is how the account ended up with twelve pools and an orphaned
# allowedClients entry on the gateway. Use it only when the stack runs with
# CognitoEnabled=false and you are bringing your own pool.
#
# This script creates, in the pool you name with --user-pool-id:
#   1. (only with --create-pool) a new Cognito User Pool
#   2. A custom domain for the token endpoint
#   3. A Resource Server with scopes for the MCP gateway
#   4. A Machine-to-Machine (M2M) App Client for Amazon Quick
#
# After running, it outputs the values needed to:
#   - Configure the AgentCore Gateway authorizer (discovery URL, allowed clients)
#   - Set up the Amazon Quick MCP Actions integration (client ID, secret, token URL)
#
# Usage:
#   scripts/setup_cognito_oauth.sh [OPTIONS]
#
# Options:
#   --user-pool-id ID     The existing Cognito User Pool to configure (required unless
#                         --create-pool is given)
#   --create-pool         Create a new User Pool. Refused by default, because the stack
#                         normally owns the pool; only for CognitoEnabled=false stacks.
#   --domain PREFIX       Cognito domain prefix (default: qs-spice-optimiser-<account-id>;
#                         prefixes are unique per Region across all accounts)
#   --region REGION       AWS region (default: us-east-1)
#   --profile NAME        AWS CLI profile
#   --resource-id ID      Resource server identifier (default: spice-optimiser-gateway)
#   -h, --help            Show this help
#
# Prerequisites:
#   - AWS CLI v2 installed and configured
#   - jq installed (for JSON parsing)
#   - Permissions: cognito-idp:Create*, cognito-idp:Describe*

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
USER_POOL_ID=""
CREATE_POOL=0
DOMAIN_PREFIX=""
REGION="us-east-1"
PROFILE=""
RESOURCE_SERVER_ID="spice-optimiser-gateway"
RESOURCE_SERVER_NAME="SpiceOptimiserGateway"
CLIENT_NAME="quicksight-mcp-client"

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-pool-id) USER_POOL_ID="$2"; shift 2 ;;
    --create-pool)  CREATE_POOL=1; shift ;;
    --domain)       DOMAIN_PREFIX="$2"; shift 2 ;;
    --region)       REGION="$2"; shift 2 ;;
    --profile)      PROFILE="$2"; shift 2 ;;
    --resource-id)  RESOURCE_SERVER_ID="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# AWS CLI profile flag
AWS_ARGS=(--region "$REGION")
if [[ -n "$PROFILE" ]]; then
  AWS_ARGS+=(--profile "$PROFILE")
fi

# Default domain prefix: unique per account, so two people following the README do not
# collide on a shared name (Cognito domain prefixes are unique per Region, account-wide).
if [[ -z "$DOMAIN_PREFIX" ]]; then
  ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text "${AWS_ARGS[@]}")"
  DOMAIN_PREFIX="qs-spice-optimiser-${ACCOUNT_ID}"
fi

# Check jq is available
if ! command -v jq &>/dev/null; then
  echo "ERROR: jq is required but not installed. Install it with: brew install jq" >&2
  exit 1
fi

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Cognito OAuth Setup for Amazon Quick ↔ AgentCore Gateway      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Create or use existing User Pool
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$USER_POOL_ID" && "$CREATE_POOL" -ne 1 ]]; then
  cat >&2 <<'EOF'
ERROR: no --user-pool-id given.

The CloudFormation stack already creates the user pool, domain, resource server and
app client (CognitoEnabled=true). Read the client ID, secret and token URL from the
stack instead - see README step 2 - rather than creating a second pool here.

If this deployment really runs with CognitoEnabled=false and you want this script to
create a pool for it, re-run with --create-pool.
EOF
  exit 2
fi

if [[ -z "$USER_POOL_ID" ]]; then
  echo "==> [1/4] Creating Cognito User Pool (--create-pool)..."
  POOL_RESPONSE=$(aws cognito-idp create-user-pool \
    "${AWS_ARGS[@]}" \
    --pool-name "spice-optimiser-gateway-pool" \
    --admin-create-user-config AllowAdminCreateUserOnly=true \
    --policies '{"PasswordPolicy":{"MinimumLength":12,"RequireUppercase":true,"RequireLowercase":true,"RequireNumbers":true,"RequireSymbols":true}}' \
    --output json)
  USER_POOL_ID=$(echo "$POOL_RESPONSE" | jq -r '.UserPool.Id')
  echo "    Created User Pool: $USER_POOL_ID"
else
  echo "==> [1/4] Using existing User Pool: $USER_POOL_ID"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Create domain (for token endpoint)
# ─────────────────────────────────────────────────────────────────────────────
echo "==> [2/4] Creating Cognito domain: $DOMAIN_PREFIX"

# Check if domain already exists
EXISTING_DOMAIN=$(aws cognito-idp describe-user-pool \
  "${AWS_ARGS[@]}" \
  --user-pool-id "$USER_POOL_ID" \
  --output json | jq -r '.UserPool.Domain // empty')

if [[ -n "$EXISTING_DOMAIN" ]]; then
  echo "    Domain already exists: $EXISTING_DOMAIN"
  DOMAIN_PREFIX="$EXISTING_DOMAIN"
else
  aws cognito-idp create-user-pool-domain \
    "${AWS_ARGS[@]}" \
    --domain "$DOMAIN_PREFIX" \
    --user-pool-id "$USER_POOL_ID" >/dev/null
  echo "    Created domain: $DOMAIN_PREFIX"
fi

TOKEN_URL="https://${DOMAIN_PREFIX}.auth.${REGION}.amazoncognito.com/oauth2/token"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Create Resource Server (defines custom scopes)
# ─────────────────────────────────────────────────────────────────────────────
echo "==> [3/4] Creating Resource Server: $RESOURCE_SERVER_ID"

SCOPES='[
  {"ScopeName":"tools.read","ScopeDescription":"Read-only access to MCP tools (list, analyze, status, report)"},
  {"ScopeName":"tools.write","ScopeDescription":"Write access to MCP tools (apply optimization)"}
]'

# Check if resource server already exists
if aws cognito-idp describe-resource-server \
  "${AWS_ARGS[@]}" \
  --user-pool-id "$USER_POOL_ID" \
  --identifier "$RESOURCE_SERVER_ID" &>/dev/null; then
  echo "    Resource server already exists, updating scopes..."
  aws cognito-idp update-resource-server \
    "${AWS_ARGS[@]}" \
    --user-pool-id "$USER_POOL_ID" \
    --identifier "$RESOURCE_SERVER_ID" \
    --name "$RESOURCE_SERVER_NAME" \
    --scopes "$SCOPES" >/dev/null
else
  aws cognito-idp create-resource-server \
    "${AWS_ARGS[@]}" \
    --user-pool-id "$USER_POOL_ID" \
    --identifier "$RESOURCE_SERVER_ID" \
    --name "$RESOURCE_SERVER_NAME" \
    --scopes "$SCOPES" >/dev/null
  echo "    Created Resource Server with scopes:"
fi
echo "      - ${RESOURCE_SERVER_ID}/tools.read"
echo "      - ${RESOURCE_SERVER_ID}/tools.write"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Create M2M App Client (client_credentials grant)
# ─────────────────────────────────────────────────────────────────────────────
echo "==> [4/4] Creating M2M App Client: $CLIENT_NAME"

CLIENT_RESPONSE=$(aws cognito-idp create-user-pool-client \
  "${AWS_ARGS[@]}" \
  --user-pool-id "$USER_POOL_ID" \
  --client-name "$CLIENT_NAME" \
  --generate-secret \
  --allowed-o-auth-flows "client_credentials" \
  --allowed-o-auth-scopes "${RESOURCE_SERVER_ID}/tools.read" "${RESOURCE_SERVER_ID}/tools.write" \
  --allowed-o-auth-flows-user-pool-client \
  --supported-identity-providers "COGNITO" \
  --output json)

CLIENT_ID=$(echo "$CLIENT_RESPONSE" | jq -r '.UserPoolClient.ClientId')
# The secret is deliberately not captured into a variable: it would end up in shell
# history, CI logs and screen shares. Cognito keeps it; read it on demand (see below).

echo "    Created App Client: $CLIENT_ID"

# ─────────────────────────────────────────────────────────────────────────────
# Output summary
# ─────────────────────────────────────────────────────────────────────────────
DISCOVERY_URL="https://cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}/.well-known/openid-configuration"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Setup Complete                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ AgentCore Gateway Authorizer Configuration                       │"
echo "│ (update agentcore.json → runtimes[].authorizerConfiguration)     │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│ Discovery URL:    $DISCOVERY_URL"
echo "│ Allowed Clients:  $CLIENT_ID"
echo "│ Allowed Scopes:   ${RESOURCE_SERVER_ID}/tools.read ${RESOURCE_SERVER_ID}/tools.write"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ Amazon Quick MCP Actions Integration (2LO / Service Auth)        │"
echo "│ (enter these in the Quick Suite Integrations → Actions UI)       │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│ MCP Server URL:   <Your AgentCore Gateway Resource URL>          │"
echo "│ Auth Type:        Service authentication (2LO)                   │"
echo "│ Client ID:        $CLIENT_ID"
echo "│ Client Secret:    not printed. Read it when you need it, with your own credentials:"
echo "│                   aws cognito-idp describe-user-pool-client --user-pool-id $USER_POOL_ID \\"
echo "│                     --client-id $CLIENT_ID --query UserPoolClient.ClientSecret --output text"
echo "│ Token URL:        $TOKEN_URL"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│ Test: Obtain a token manually                                    │"
echo "├─────────────────────────────────────────────────────────────────┤"
echo "│ curl -X POST '$TOKEN_URL' \\"
echo "│   -H 'Content-Type: application/x-www-form-urlencoded' \\"
echo "│   -d 'grant_type=client_credentials&client_id=${CLIENT_ID}&client_secret=<CLIENT_SECRET>&scope=${RESOURCE_SERVER_ID}/tools.read ${RESOURCE_SERVER_ID}/tools.write'"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""
echo "Next steps:"
echo "  1. Update agentcore.json allowedClients to include: $CLIENT_ID"
echo "  2. Run 'agentcore deploy' to apply the authorizer change"
echo "  3. In Amazon Quick Suite → Integrations → Actions → MCP:"
echo "     - Enter the Gateway Resource URL as the MCP server endpoint"
echo "     - Select 'Service authentication' (2LO)"
echo "     - Enter Client ID, Client Secret, and Token URL from above"
echo ""
echo "IMPORTANT: Store the Client Secret securely. It will not be shown again."
