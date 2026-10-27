# Security

## Disclaimer

This is sample code provided for demonstration purposes. It is **not production-ready**.
Review, test and harden it for your security, compliance and operational requirements
before running it against production Amazon QuickSight datasets. Run in `dry_run` mode
first and confirm that snapshots are being written to S3 before enabling `apply`.

## Reporting vulnerabilities

If you discover a potential security issue in this project, notify AWS Security through
the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/).
Please do **not** create a public GitHub issue.

## AWS services used

AWS Lambda, Amazon S3, Amazon DynamoDB, Amazon SNS, Amazon SQS, AWS Key Management
Service, Amazon EventBridge (optional schedule), Amazon CloudWatch (logs, metrics, alarms),
AWS X-Ray, Amazon Cognito, Amazon Bedrock AgentCore (Gateway), Amazon QuickSight, Amazon
Quick Suite.

## Prerequisites and permissions

Deploying requires permissions to create IAM roles, a KMS key, Lambda functions, S3
buckets, DynamoDB tables, SNS topics, SQS queues, CloudWatch alarms, Cognito user pools
and Bedrock AgentCore gateways. The principal that deploys the stack must be allowed to
use the stack's KMS key (`OptimiserKey`) through Lambda and DynamoDB; the key policy's
account-root statement covers an administrator.

The Lambda execution role is granted `quicksight:UpdateDataSet` and
`quicksight:CreateIngestion` on every dataset in the account and Region (there is no
narrower resource type for datasets chosen at run time), and read access to every
analysis and dashboard. The tool can therefore modify any SPICE dataset it is pointed
at. QuickSight additionally checks dataset-level permissions, so the role must also be
registered as a QuickSight user and granted owner permission on each dataset it may
change (README, "Resources created").

## What the solution does to protect a destructive operation

- `dry_run` is the default; `remove_unused_columns` and `rollback_dataset` each require
  an explicit `action` confirmation that the Lambda checks itself.
- Every modification is preceded by a snapshot of the full dataset definition to S3
  (versioned, SSE-S3, TLS-only, access-logged, 90-day lifecycle). Rollback only accepts
  a snapshot that belongs to the target dataset and refuses one whose body names a
  different dataset.
- A per-dataset lock in S3 serialises runs; the safety threshold refuses to remove more
  than a configurable share of a dataset's columns; quarantine and scheduled runs are
  opt-in.
- The MCP surface is reached only through the AgentCore Gateway, which validates a
  Cognito client-credentials JWT (`tools.read` / `tools.write` scopes) before invoking
  the function.
- Alarms on errors, throttles, unrecognised tool calls and failed asynchronous
  invocations publish to an SNS topic encrypted with a stack-owned KMS key whose policy
  admits CloudWatch; a dead-letter queue keeps the record of any asynchronous
  invocation that fails outright.

## Secrets

The only credential in the design is the Cognito app-client secret that Amazon Quick
Suite presents to the token endpoint. It is generated and held by Cognito, is never
written to the repository, the template, the Lambda or S3, and is not printed by any
script. Read it when you configure the Quick Suite connector, with your own
credentials (README step 2), and do not paste it into chat, tickets or CI logs.
Rotation means creating a new app client and updating the connector.

## Known security considerations

These are deliberate choices for a sample. Review each before production use.

- **Account-wide QuickSight write scope.** See "Prerequisites and permissions". If the
  account mixes production and sandbox assets, restrict `quicksight:UpdateDataSet` to a
  tag- or name-scoped set of datasets, or deploy the stack into an account that holds
  only the datasets it should manage.
- **KMS key administration.** The key policy grants `kms:*` to the account root (the
  standard bootstrap). Scope it to your deployment role for production.
- **Lambda runs outside a VPC.** It calls only AWS APIs; a VPC would add NAT or
  endpoints without a control.
- **No reserved concurrency.** The function invokes itself asynchronously for long
  jobs, so a reserved concurrency of 1 deadlocks it. Serialisation comes from the
  per-dataset lock instead. Set a small bound (for example 5) if your account's
  concurrency is contended.
- **The access-log bucket has no access logging of its own**, which is normal for a
  log destination.
- **Explicit resource names** (`<stack>-function`, `<stack>-lambda-role`, the tables,
  the alarms) block CloudFormation replacement of those resources. Fine for a sample;
  remove the names if you need in-place replacement.
- **Retained resources.** The two S3 buckets, both DynamoDB tables and the KMS key have
  `DeletionPolicy: Retain`; see "Cleanup".
- **Logs at INFO level** record which columns were removed from which dataset and from
  which snapshot. They contain no dataset rows, secrets or tokens. Set `LogLevel=WARNING`
  if your environment requires quieter production logs.

## Production hardening recommendations

1. Scope the KMS key administration statement to your deployment role.
2. Restrict `quicksight:UpdateDataSet` to the datasets the tool is meant to manage.
3. Set `NotificationEmail` at deployment and confirm the subscription, then trigger a
   test alarm and verify "Successfully executed action" in the alarm history.
4. Enable S3 Object Lock (governance mode) on the snapshot prefix if snapshots must be
   tamper-evident, and align its retention with the 90-day lifecycle.
5. Enable CloudTrail data events for the two DynamoDB tables and the snapshot bucket if
   you need per-item audit beyond the server access logs.
6. Add resource tags for ownership and cost allocation.
7. Run `pip-audit`, `npm audit` (in `agentcore_project/spiceoptimisermcp/agentcore/cdk`)
   and the test suite before each release.

## Cleanup

1. `aws cloudformation delete-stack --stack-name <stack>`.
2. Retained resources survive the stack: empty and delete the two S3 buckets
   (`OptimiserBucket`, `AccessLogBucket`), delete the two DynamoDB tables and schedule
   the KMS key for deletion (`aws kms schedule-key-deletion`) once nothing depends on it.
3. Delete the AgentCore gateway: `agentcore destroy` from
   `agentcore_project/spiceoptimisermcp`.
4. Remove the QuickSight user you registered for the Lambda role
   (`aws quicksight delete-user`).
5. Remove the MCP connector from Amazon Quick Suite.

## Dependencies

Python runtime: `boto3` (installed into the deployment package at deploy time).
Python test: `hypothesis`, `pytest`, `pytest-cov`, `pyyaml` (pinned in
`requirements.txt`). Node (CDK app, deploy-time only): `aws-cdk-lib`,
`@aws/agentcore-cdk`, `constructs`. Third-party licence texts are in
`THIRD_PARTY_LICENSES.txt`; a software bill of materials is in `docs/sbom/`.
