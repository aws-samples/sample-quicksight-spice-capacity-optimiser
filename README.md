# QuickSight SPICE Capacity Optimiser

Identifies and removes unused columns from Amazon QuickSight SPICE datasets to
cut storage cost and speed up refreshes. Powered by an Amazon Quick chat agent
connected via MCP to an Amazon Bedrock AgentCore Gateway.

> **Disclaimer**: This is sample code provided for demonstration purposes. It is
> **not production-ready** and must be reviewed, tested and hardened for your security,
> compliance and operational requirements before use against production QuickSight
> datasets. The `remove_unused_columns` tool changes live dataset definitions: run in
> `dry_run` mode first and confirm that snapshots are being written to S3 before enabling
> `apply`. See [SECURITY.md](SECURITY.md).

## Architecture

```
┌─────────────────────┐
│  Amazon Quick       │  User asks: "Scan unused columns in my dataset"
│  Chat Agent         │
└────────┬────────────┘
         │ OAuth 2.0 client_credentials (2LO)
         ▼
┌─────────────────────┐
│  Amazon Cognito     │  Returns JWT access token
│  (User Pool)        │
└────────┬────────────┘
         │ Bearer <JWT>
         ▼
┌─────────────────────┐
│  AgentCore Gateway  │  Validates JWT → invokes Lambda per tool call
│  (Bedrock AgentCore)│
└────────┬────────────┘
         │ Lambda.Invoke
         ▼
┌─────────────────────┐
│  Lambda Function    │  Tool dispatcher + optimisation pipeline
│  (Python 3.14)      │  Fast ops: immediate | Slow ops: async + DynamoDB
└─────────────────────┘
```

## Tools (exposed via MCP)

| Tool | Type | Description |
|------|------|-------------|
| `list_spice_datasets` | Read | List all SPICE datasets in a QuickSight account |
| `scan_unused_columns` | Write | Scan datasets to identify unused columns (read-only analysis) |
| `remove_unused_columns` | Write | Actually remove unused columns (modifies dataset, snapshot first) |
| `get_job_result` | Read | Poll the result of a scan or remove operation |
| `list_snapshots` | Read | List restorable snapshots for a dataset, newest first |
| `rollback_dataset` | Write | Restore a dataset from a snapshot, undoing a removal (requires `action="rollback"`) |

## Quick Start

### 1. Deploy infrastructure

```bash
# Create an S3 bucket for the Lambda deployment package (one-time). Bucket names are
# global, so include your account id to get one nobody else can have claimed.
aws s3 mb s3://<your-deploy-bucket>-<account-id>-us-east-1 --region us-east-1

# Deploy everything: Lambda + Cognito + DynamoDB + AgentCore Gateway
scripts/deploy_all.sh --code-bucket <your-deploy-bucket>-<account-id>-us-east-1 --region us-east-1 --profile <profile>
```

The stack's Cognito token endpoint uses the domain prefix `qs-spice-optimiser-<account-id>`
unless you pass `CognitoDomainPrefix`. Changing the prefix on an existing stack replaces
the domain and therefore the token URL configured in Amazon Quick Suite.

```bash
# Optional: keep a prefix you already use
aws cloudformation deploy ... --parameter-overrides CognitoDomainPrefix=<your-prefix>
```

### 2. Read the Cognito OAuth credentials (one-time)

The stack already created the user pool, hosted domain, resource server and the
client-credentials app client (`CognitoEnabled=true`, the default). Read what Amazon
Quick needs from the stack rather than creating a second set:

```bash
STACK=spice-optimiser; REGION=us-east-1
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?starts_with(OutputKey,'Cognito')].[OutputKey,OutputValue]" --output table

# The client secret is never printed by CloudFormation; read it from Cognito:
aws cognito-idp describe-user-pool-client --region "$REGION" \
  --user-pool-id "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='CognitoUserPoolId'].OutputValue" --output text)" \
  --client-id    "$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" --query "Stacks[0].Outputs[?OutputKey=='CognitoClientId'].OutputValue" --output text)" \
  --query 'UserPoolClient.ClientSecret' --output text
```

You need the **Client ID** (`CognitoClientId`), the **Client Secret** (above) and the
**Token URL** (`CognitoTokenUrl`). Treat the secret like a password: paste it into
the Quick Suite connector and do not store it in the repository.

`scripts/setup_cognito_oauth.sh` is only for deployments that run with
`CognitoEnabled=false` and bring their own user pool; it refuses to create a pool
unless you pass `--create-pool`, so it cannot duplicate the stack's.

### 3. Create the MCP connector in Amazon Quick Suite

1. Sign in to Amazon Quick Suite with an **Author Pro** role
2. Go to **Integrations** → **Actions** tab
3. Click **+** on the **Model Context Protocol** tile
4. Fill in:

| Field | Value |
|-------|-------|
| **Name** | SPICE Capacity Optimiser |
| **Description** | Scans and removes unused columns from QuickSight SPICE datasets to reduce storage costs |
| **MCP server endpoint** | `<Gateway URL from deploy output>/mcp` |

5. Click **Next**
6. For authentication, select **Service authentication (2LO)**:

| Field | Value |
|-------|-------|
| **Client ID** | Stack output `CognitoClientId` (step 2) |
| **Client secret** | `describe-user-pool-client` output (step 2) |
| **Token URL** | Stack output `CognitoTokenUrl` (step 2) |

7. Click **Create and continue**
8. Review the 6 discovered tools and click **Next**
9. Optionally share with other users → **Done**

### 4. Create/configure a Quick Chat Agent

You can use the default **My Assistant** agent or create a custom one:

1. Go to **Chat Agents** in Amazon Quick Suite
2. Create a new agent or edit an existing one
3. Add the MCP integration you just created
4. Set the system prompt (see [Chat Agent Persona](#chat-agent-persona) below)

### 5. Test it

Open the chat agent and try:
- "List my SPICE datasets in account 123456789012"
- "Scan unused columns in dataset sample-quicksight-dataset"
- "Remove the unused columns from that dataset"
- "What snapshots can you restore for that dataset?" — read-only, and the fastest check that the rollback tools reached the connector
- "Undo that optimisation" — the agent should list snapshots and ask which one before touching anything

---

## Chat Agent Persona

The system prompt lives in **`configuration/quick_agent.json`**, under `systemPrompt`.
Print it and paste it into the agent's instructions:

```bash
python3 -c "import json; print(json.load(open('configuration/quick_agent.json'))['systemPrompt'])"
```

It is kept there rather than inline here on purpose. This section used to hold a second
copy, and the copy drifted: its undo workflow told the agent to call `rollback_dataset`
while the `Available tools` list four lines below named only four tools and omitted both
`list_snapshots` and `rollback_dataset`. Anyone who pasted it got an agent instructed to
use a tool it had been told did not exist.

`configuration/quick_agent.json` is the copy that
`tests/unit/test_agent_config_agrees_with_tools.py` checks against the dispatcher's
`KNOWN_TOOLS`, so it cannot fall out of step with the deployed tools without a test
failing. A prose copy here has no such guard, which is why there is no longer one.

What the prompt covers, so you can tell whether an edit belongs in it:

- Scan before removing, and never remove without explicit confirmation
- Both write tools need a confirmation token — `action="remove"`, `action="rollback"` — which the agent cannot discover by trial, since without it the tool just refuses
- `rollback_dataset` returns synchronously, so it must not be polled with `get_job_result` the way scan and remove are
- `list_snapshots` first, always, and say how old the snapshot is before offering to restore it
- Rollback restores the whole dataset definition, so it undoes manual edits made since the snapshot too, not only the column removal

> The live prompt is console-side state in Quick Suite. No test can see it, so it can
> still drift from this repo. If the agent starts calling a tool that does not exist,
> re-paste it.

### Persona summary

| Attribute | Value |
|-----------|-------|
| **Name** | SPICE Optimization Assistant |
| **Role** | QuickSight SPICE Storage Optimization Specialist |
| **Personality** | Precise, cautious, data-driven |
| **Tone** | Professional, uses specific numbers (column counts, MB/GB) |

### Example conversation

```
User:  What SPICE datasets do I have in account 123456789012?

Agent: [calls list_spice_datasets]
       You have 63 SPICE datasets. Here are the top ones by storage:
       - sample-quicksight-dataset (4.3 KB)
       - sales-summary-2024 (12.1 MB)
       ...

User:  Scan unused columns in sample-quicksight-dataset

Agent: [calls scan_unused_columns → polls get_job_result]
       Scan complete: 11 of 17 columns are unused.
       Estimated savings: 4.3 KB SPICE capacity.
       Join keys (order_id, product_id) are preserved.
       2 dashboards use this dataset — unused columns are not referenced by them.

       Would you like me to remove the unused columns?

User:  Yes, go ahead.

Agent: [calls remove_unused_columns with action="remove" → polls get_job_result]
       Done! 11 columns removed. Dataset now has 6 columns.
       Snapshot saved to S3 for rollback if needed.
```

---

## How it works (under the hood)

The analysis follows the methodology from the AWS blog
"[Enhance data governance through column-level lineage in Amazon QuickSight](https://aws.amazon.com/blogs/business-intelligence/enhance-data-governance-through-column-level-lineage-in-amazon-quicksight/)":

1. **Discover consumers** — List all analyses and dashboards, fetch definitions
2. **Discover child datasets (recursive)** — Find all datasets that use this one as a source at any depth (children, grandchildren, etc.) with cycle detection. For each child, scan its consumers (dashboards/analyses) to trace which parent columns are actually used end-to-end through the dependency chain. Handles all new experience DataPrepConfiguration transform steps (Join, Project, Rename, Cast, Aggregate, Filter, Pivot, Append, Unpivot, CreateColumns).
3. **Scan column references** — Extract `ColumnIdentifier` + `DataSetIdentifier` from each consumer definition (visuals, calculated fields, filters, parameters, hierarchies)
4. **Classify columns** — Compare dataset columns against consumer + child dataset references:
   - **Used** = referenced by at least one consumer or child dataset
   - **Unused** = not referenced by any consumer or child dataset
   - **Join key** = always preserved (from JoinInstruction.OnClause or JoinStep.OnClause)
5. **Remove** (if confirmed) — Call `UpdateDataSet` with modified definition excluding unused columns

## Project structure

```
├── lambda_source/              Lambda function code
│   ├── handler.py              Entry point (routes gateway calls vs pipeline)
│   ├── mcp_dispatcher.py       Tool dispatcher (6 tools)
│   ├── discovery.py            Consumer discovery (analyses + dashboards)
│   ├── child_discovery.py      Child dataset discovery (datasets using this one as source)
│   ├── scanner.py              Column reference extraction
│   ├── analyzer.py             Column classification (used/unused/join_key)
│   ├── modifier.py             Dataset modification (UpdateDataSet)
│   ├── snapshot.py             Pre-modification snapshots
│   ├── quarantine.py           Quarantine period tracking
│   └── reporter.py             Report generation
├── agentcore_project/          AgentCore Gateway configuration
│   └── spiceoptimisermcp/
│       ├── tool-schema.json    Tool definitions (6 tools). The CDK resolves this
│       │                       path from agentcore.json relative to the project
│       │                       dir, not to agentcore.json — do not move it
│       └── agentcore/
│           └── agentcore.json  Gateway + auth config
├── configuration/
│   ├── quick_agent.json        Chat agent persona + setup steps
│   ├── agentcore_gateway.json  Architecture reference doc
│   └── config.json             Optimiser settings
├── deployment/
│   └── template.yaml           CloudFormation (Lambda, Cognito, DynamoDB, S3, etc.)
├── scripts/
│   ├── deploy_all.sh           One-step deploy (Lambda + gateway)
│   ├── deploy_mcp.sh           Gateway-only deploy
│   ├── setup_cognito_oauth.sh  Cognito setup for bring-your-own-pool deployments only
│   └── sync_agentcore_config.py  Sync config from CloudFormation outputs
└── tests/                      Unit + property + integration tests
```

## Resources created

Two stacks and one manual step. Nothing else is created in the account.

**CloudFormation stack `<stack-name>` (`deployment/template.yaml`), default parameters unless noted:**

| Logical ID | Type | Name / scope | Created when |
|---|---|---|---|
| `OptimiserBucket`, `OptimiserBucketPolicy` | S3 bucket + bucket policy | auto-named (or `S3BucketName`); SSE-S3, versioned, public access blocked, plaintext (non-TLS) requests denied, access logs delivered to `AccessLogBucket`; snapshots expire after 90 days, reports after 365. `DeletionPolicy: Retain` | always |
| `AccessLogBucket`, `AccessLogBucketPolicy` | S3 bucket + bucket policy | server access logs of `OptimiserBucket` under `<stack>/s3-access-logs/`; SSE-S3, ACLs disabled, public access blocked, non-TLS denied, logs expire after 365 days. `DeletionPolicy: Retain` | always |
| `McpJobStoreTable` | DynamoDB table | `<stack>-mcp-jobs`, on-demand, TTL, point-in-time recovery, encrypted with `OptimiserKey`. `DeletionPolicy: Retain` | always |
| `QuarantineTable` | DynamoDB table | `<stack>-quarantine`, on-demand, TTL, point-in-time recovery, encrypted with `OptimiserKey`. `DeletionPolicy: Retain` | `QuarantineEnabled=true` |
| `LambdaExecutionRole` | IAM role | `<stack>-lambda-role`, eleven inline policies, no managed policies | always |
| `OptimiserFunction` | Lambda function | `<stack>-function`, Python 3.14, 512 MB, 15 min, X-Ray active, environment variables encrypted with `OptimiserKey` | always |
| `OptimiserFunctionInvokeConfig` | Lambda event invoke config | no retries; the failed invocation record is also sent to `NotificationTopic` when it exists | always |
| `AsyncFailureQueue`, `AsyncFailureQueuePolicy` | SQS queue + queue policy | `<stack>-async-failures`, dead-letter queue for failed asynchronous invocations; encrypted with the AWS managed SQS key, non-TLS denied, 14-day retention | always |
| `OptimiserLogGroup` | CloudWatch log group | `/aws/lambda/<stack>-function`, 30-day retention, encrypted with `OptimiserKey` | always |
| `ErrorAlarm`, `ThrottleAlarm`, `UnrecognisedToolAlarm`, `AsyncFailureAlarm` | CloudWatch alarms | `<stack>-errors`, `<stack>-throttles`, `<stack>-unrecognised-tool`, `<stack>-async-failures`; notify `NotificationTopic` when it exists | always |
| `AgentCoreGatewayPermission` | Lambda permission | lets `bedrock-agentcore.amazonaws.com` in this account invoke the function | always |
| `OptimiserKey`, `OptimiserKeyAlias` | KMS key + alias | `alias/<stack>`, rotation on; encrypts the SNS topic, the Lambda environment variables, the log group and both DynamoDB tables. Usable only by the function (through Lambda, SNS and DynamoDB), CloudWatch Logs for this log group, and this stack's alarms. `DeletionPolicy: Retain` | always |
| `NotificationTopic` | SNS topic | `<stack>-notifications`, encrypted with `OptimiserKey` | `SNSTopicEnabled=true` (default) |
| `NotificationSubscription` | SNS subscription | email to `NotificationEmail` | `NotificationEmail` set |
| `CognitoUserPool`, `CognitoUserPoolDomain`, `CognitoResourceServer`, `CognitoMcpClient` | Cognito user pool, hosted domain, resource server, app client | `<stack>-gateway-pool`, domain `CognitoDomainPrefix`, scopes `tools.read` / `tools.write`, client-credentials client with a generated secret | `CognitoEnabled=true` (default) |
| `ScheduledRule`, `SchedulePermission` | EventBridge rule + Lambda permission | `<stack>-schedule` | `ScheduleEnabled=true` (off by default) |

**CDK stack `AgentCore-spiceoptimisermcp-default`** (deployed by `scripts/deploy_mcp.sh` through the AgentCore CLI from `agentcore_project/spiceoptimisermcp/agentcore/`; synthesize with `cdk synth` in its `cdk/` directory to inspect it):

| Resource | Type | Purpose |
|---|---|---|
| Gateway role + default policy | IAM role, IAM policy | trusted by `bedrock-agentcore.amazonaws.com`; may `lambda:InvokeFunction` on `<stack>-function` and read the CDK assets bucket |
| `spice-optimiser-gateway` | Bedrock AgentCore Gateway | MCP endpoint; Cognito JWT authorizer |
| Gateway target | Bedrock AgentCore GatewayTarget | maps the six tools in `tool-schema.json` to the Lambda |
| CDK metadata | `AWS::CDK::Metadata` | CDK bookkeeping; the stack also uses the account's CDK bootstrap assets bucket |

**Manual step:** QuickSight checks dataset permissions in addition to IAM, so the Lambda role must be registered as a QuickSight user and granted owner permissions on each dataset it is allowed to modify (`aws quicksight register-user` with the role ARN, then `update-data-set-permissions`). Without it `UpdateDataSet` is denied even though IAM allows it.

Deleting the CloudFormation stack removes everything except the retained resources (both S3 buckets, both DynamoDB tables and the KMS key) and the QuickSight user registration; see Cleanup.

## Configuration

`configuration/config.json`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `safety_threshold_pct` | 0.7 | Max fraction of columns removable (0.0–1.0) |
| `quarantine_period_days` | *unset* (disabled) | Days a column must be continuously unused before removal. Omit to run with no waiting period |
| `execution_mode` | `dry_run` | `dry_run` (analysis only) or `apply` (modify) |
| `cascade_child_refresh` | `false` | After an apply refreshes the parent, also refresh every descendant dataset (parents first). Off by default |

### Cascading child refresh is opt-in

Removing columns from a parent and refreshing it leaves every child dataset holding SPICE
data shaped to the parent's old column set. The child's next scheduled refresh — or a
child transform that reads a now-absent parent column — can then fail or show errors.
`cascade_child_refresh` makes the optimiser refresh the whole dependency tree after an
apply, parents before children, so each level re-resolves against the one above it.

It is **off by default** because it fans `CreateIngestion` out across every descendant,
which spends API quota and SPICE refresh capacity. Turn it on per removal via the chat
agent (`remove_unused_columns` accepts `cascade_child_refresh`) or in `config.json`.

Three things to know:

- **It only runs in apply mode.** A dry run changes nothing, so there is nothing to
  cascade.
- **A child refresh failure never fails the parent optimisation.** The parent change is
  already committed by the time children are refreshed. A child that cannot be refreshed
  at all — an uploaded-file source — is reported as such, not as an error; a child that
  fails for another reason (throttling, access denied) is recorded, and the remaining
  children are still refreshed. One summary notification names any children to refresh by
  hand.
- **Cascaded refreshes carry an `optimiser-cascade-` ingestion id**, so they are
  distinguishable in the QuickSight ingestion history from the parent's own `optimiser-`
  refresh and from a rollback's `optimiser-rollback-`.

### Troubleshooting: a dataset looks empty

If `describe-data-set` shows one empty `PhysicalTableMap` entry and no
`DataPrepConfiguration`, the dataset is almost certainly fine and your SDK is too old.

Below boto3 **1.43.64** the SDK does not recognise the `FileSource` variant of
`PhysicalTableMap`. It does not error — it returns a hollow but well-formed response — so
a healthy file-upload or new-experience dataset reads as empty, and the pre-flight guard
for empty `PhysicalTableMap` entries looks like it is misfiring.

```
via AWS CLI 2.17.20 / boto3 1.42.38    PhysicalTableMap: [{}]      DataPrepConfiguration: absent
via boto3 1.43.93                      PhysicalTableMap: FileSource(14 InputColumns)
                                       DataPrepConfiguration: present
```

The deployed Lambda is unaffected: `deploy_all.sh` installs the latest boto3 into the
package, and `check_boto3_version()` logs a warning at cold start if that ever regresses.
The problem is local tooling. For any local investigation, use a virtualenv on a current
SDK:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Note that the bundled AWS CLI has its own vendored botocore, so `aws quicksight
describe-data-set` can be wrong even when your Python environment is right.

### Quarantine is opt-in

Quarantine adds a waiting period: a column must be seen unused across scans spanning
N days before it becomes eligible for removal. It protects against transient cases —
someone rebuilding a dashboard makes a column look unused this week and used again
next week.

It is **off by default** and requires infrastructure, because the "first seen unused"
timestamps live in DynamoDB:

```bash
# Provisions the tracking table and grants the Lambda access to it
scripts/deploy_all.sh --code-bucket <bucket> --region us-east-1 \
  --parameter-overrides QuarantineEnabled=true
```

Once enabled, pass `quarantine_period_days` on `scan_unused_columns` or
`remove_unused_columns` to apply it.

### Scheduled runs are opt-in

By default the optimiser only runs when you ask it to, through the chat agent. An
EventBridge schedule for unattended periodic runs is available as a deployment option:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ScheduleEnabled` | `false` | Create the schedule at all. No rule exists unless `true` |
| `ScheduleExpression` | `rate(7 days)` | Frequency. Any EventBridge rate or cron expression |
| `ScheduleExecutionMode` | `dry_run` | `dry_run` reports only; `apply` removes columns unattended |
| `ScheduleTargetDatasets` | *empty* | Comma-separated dataset IDs. Required when enabled |
| `ScheduleSafetyThresholdPct` | `0.7` | Max fraction of columns a scheduled run may remove |

```bash
# A weekly read-only report on two datasets
scripts/deploy_all.sh --code-bucket <bucket> --region us-east-1 \
  --parameter-overrides \
      ScheduleEnabled=true \
      'ScheduleExpression=rate(7 days)' \
      ScheduleExecutionMode=dry_run \
      'ScheduleTargetDatasets=abc-123,def-456'
```

Three things to know:

- **`ScheduleTargetDatasets` is required**, and deliberately explicit rather than "every
  SPICE dataset". A scan costs roughly 30–45s per dataset because child discovery walks
  the account, so more than about 15 datasets will exceed the Lambda 15-minute timeout.
- **`ScheduleExecutionMode=apply` deletes columns with nobody watching.** A snapshot is
  written first and rollback is possible, but choose it deliberately. `dry_run` gives you
  the same analysis and a report in S3 while changing nothing.
- **The schedule carries its own configuration.** The rule's `Input` is built from these
  parameters, so what a scheduled run will do is visible in the template. There is no
  separate config file to write, and none to go missing.

Two behaviours worth knowing before you turn it on:

- **It changes the workflow to two visits.** The first scan only records "first seen
  unused today", so nothing is eligible yet. You need a second run N days later.
- **Requesting it without the table is refused, not ignored.** If
  `quarantine_period_days` is supplied on a deployment where quarantine is disabled, the
  call fails with an explanation rather than silently removing columns without the
  waiting period you asked for.

### Troubleshooting: a dataset looks empty

If `describe-data-set` shows one empty `PhysicalTableMap` entry and no
`DataPrepConfiguration`, the dataset is almost certainly fine and your SDK is too old.

Below boto3 **1.43.64** the SDK does not recognise the `FileSource` variant of
`PhysicalTableMap`. It does not error — it returns a hollow but well-formed response — so
a healthy file-upload or new-experience dataset reads as empty, and the pre-flight guard
for empty `PhysicalTableMap` entries looks like it is misfiring.

```
via AWS CLI 2.17.20 / boto3 1.42.38    PhysicalTableMap: [{}]      DataPrepConfiguration: absent
via boto3 1.43.93                      PhysicalTableMap: FileSource(14 InputColumns)
                                       DataPrepConfiguration: present
```

The deployed Lambda is unaffected: `deploy_all.sh` installs the latest boto3 into the
package, and `check_boto3_version()` logs a warning at cold start if that ever regresses.
The problem is local tooling. For any local investigation, use a virtualenv on a current
SDK:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Note that the bundled AWS CLI has its own vendored botocore, so `aws quicksight
describe-data-set` can be wrong even when your Python environment is right.

### Quarantine is opt-in

Quarantine adds a waiting period: a column must be seen unused across scans spanning
N days before it becomes eligible for removal. It protects against transient cases —
someone rebuilding a dashboard makes a column look unused this week and used again
next week.

It is **off by default** and requires infrastructure, because the "first seen unused"
timestamps live in DynamoDB:

```bash
# Provisions the tracking table and grants the Lambda access to it
scripts/deploy_all.sh --code-bucket <bucket> --region us-east-1 \
  --parameter-overrides QuarantineEnabled=true
```

Once enabled, pass `quarantine_period_days` on `scan_unused_columns` or
`remove_unused_columns` to apply it.

### Scheduled runs are opt-in

By default the optimiser only runs when you ask it to, through the chat agent. An
EventBridge schedule for unattended periodic runs is available as a deployment option:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ScheduleEnabled` | `false` | Create the schedule at all. No rule exists unless `true` |
| `ScheduleExpression` | `rate(7 days)` | Frequency. Any EventBridge rate or cron expression |
| `ScheduleExecutionMode` | `dry_run` | `dry_run` reports only; `apply` removes columns unattended |
| `ScheduleTargetDatasets` | *empty* | Comma-separated dataset IDs. Required when enabled |
| `ScheduleSafetyThresholdPct` | `0.7` | Max fraction of columns a scheduled run may remove |

```bash
# A weekly read-only report on two datasets
scripts/deploy_all.sh --code-bucket <bucket> --region us-east-1 \
  --parameter-overrides \
      ScheduleEnabled=true \
      'ScheduleExpression=rate(7 days)' \
      ScheduleExecutionMode=dry_run \
      'ScheduleTargetDatasets=abc-123,def-456'
```

Three things to know:

- **`ScheduleTargetDatasets` is required**, and deliberately explicit rather than "every
  SPICE dataset". A scan costs roughly 30–45s per dataset because child discovery walks
  the account, so more than about 15 datasets will exceed the Lambda 15-minute timeout.
- **`ScheduleExecutionMode=apply` deletes columns with nobody watching.** A snapshot is
  written first and rollback is possible, but choose it deliberately. `dry_run` gives you
  the same analysis and a report in S3 while changing nothing.
- **The schedule carries its own configuration.** The rule's `Input` is built from these
  parameters, so what a scheduled run will do is visible in the template. There is no
  separate config file to write, and none to go missing.

Two behaviours worth knowing before you turn it on:

- **It changes the workflow to two visits.** The first scan only records "first seen
  unused today", so nothing is eligible yet. You need a second run N days later.
- **Requesting it without the table is refused, not ignored.** If
  `quarantine_period_days` is supplied on a deployment where quarantine is disabled, the
  call fails with an explanation rather than silently removing columns without the
  waiting period you asked for.

## Cleanup

1. Delete the CloudFormation stack: `aws cloudformation delete-stack --stack-name <stack>`.
2. Retained resources survive the stack and are deleted by hand: empty and delete the two
   S3 buckets (`OptimiserBucket` and its access-log bucket), delete the DynamoDB tables
   `<stack>-mcp-jobs` and `<stack>-quarantine`, and schedule the KMS key
   `alias/<stack>` for deletion once nothing else uses it.
3. Delete the AgentCore gateway: `cd agentcore_project/spiceoptimisermcp && agentcore destroy`.
4. Remove the QuickSight user registered for the Lambda role (`aws quicksight delete-user`)
   and the MCP connector in Amazon Quick Suite.
5. Delete the Lambda deployment-package bucket you created in step 1 of the quick start.

## Tests

```bash
pip install -e ".[dev]"
scripts/run_tests.sh              # full suite
scripts/run_tests.sh --coverage   # plus a coverage report
```

Bare `pytest` also passes, but prefer the script: it checks the installed boto3 against
the floor first and warns if it is below it. An older SDK does not fail on file-upload or
new-experience datasets, it reads them as empty — so a green run against one proves less
than it looks like it does. See [Troubleshooting: a dataset looks
empty](#troubleshooting-a-dataset-looks-empty).

No test reaches AWS; a fixture fails any test that tries.
