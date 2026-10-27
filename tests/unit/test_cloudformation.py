# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Unit tests for CloudFormation template validation.

Validates: Requirements 13.2, 13.3
- 13.2: Template provisions Lambda, IAM role, EventBridge rule, S3 bucket,
         optional DynamoDB table, optional SNS topic
- 13.3: IAM permissions scoped to minimum required actions
"""
import os
import pytest
import yaml


TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "deployment", "template.yaml"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom YAML loader that handles CloudFormation intrinsic function tags
# ─────────────────────────────────────────────────────────────────────────────

CFN_TAGS = [
    "!Ref", "!Sub", "!GetAtt", "!If", "!Not", "!Equals",
    "!And", "!Or", "!Select", "!Split", "!Join", "!FindInMap",
    "!GetAZs", "!ImportValue", "!Condition",
]


class CfnLoader(yaml.SafeLoader):
    """YAML loader that handles CloudFormation intrinsic function tags."""
    pass


def _cfn_multi_constructor(loader, tag_suffix, node):
    """Handle multi-value CFN intrinsic functions."""
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    elif isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node, deep=True)}
    elif isinstance(node, yaml.MappingNode):
        return {tag_suffix: loader.construct_mapping(node, deep=True)}


# Register all CloudFormation tags
for tag in CFN_TAGS:
    tag_name = tag[1:]  # Remove the '!' prefix
    CfnLoader.add_constructor(
        tag,
        lambda loader, node, t=tag_name: (
            {t: loader.construct_scalar(node)}
            if isinstance(node, yaml.ScalarNode)
            else {t: loader.construct_sequence(node, deep=True)}
            if isinstance(node, yaml.SequenceNode)
            else {t: loader.construct_mapping(node, deep=True)}
        ),
    )


@pytest.fixture
def template():
    """Load and parse the CloudFormation template with CFN intrinsic support."""
    with open(TEMPLATE_PATH, "r") as f:
        return yaml.load(f, Loader=CfnLoader)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Template syntax validation
# ─────────────────────────────────────────────────────────────────────────────


class TestTemplateSyntax:
    """Verify the template is valid YAML with required top-level keys."""

    def test_template_is_valid_yaml(self, template):
        """Template parses as valid YAML without errors."""
        assert template is not None
        assert isinstance(template, dict)

    def test_has_aws_template_format_version(self, template):
        assert "AWSTemplateFormatVersion" in template
        assert template["AWSTemplateFormatVersion"] == "2010-09-09"

    def test_has_parameters(self, template):
        assert "Parameters" in template
        assert isinstance(template["Parameters"], dict)

    def test_has_conditions(self, template):
        assert "Conditions" in template
        assert isinstance(template["Conditions"], dict)

    def test_has_resources(self, template):
        assert "Resources" in template
        assert isinstance(template["Resources"], dict)

    def test_has_outputs(self, template):
        assert "Outputs" in template
        assert isinstance(template["Outputs"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# 2. IAM least-privilege checks
# ─────────────────────────────────────────────────────────────────────────────


class TestIAMLeastPrivilege:
    """Verify IAM policies follow least-privilege principle."""

    @pytest.fixture
    def iam_role(self, template):
        """Extract the Lambda execution IAM role resource."""
        resources = template["Resources"]
        roles = {
            k: v for k, v in resources.items() if v["Type"] == "AWS::IAM::Role"
        }
        assert len(roles) >= 1, "Expected at least one IAM role"
        # Get the Lambda execution role
        return roles["LambdaExecutionRole"]

    @pytest.fixture
    def policies(self, iam_role):
        """Return policies as a dict keyed by PolicyName."""
        policy_list = iam_role["Properties"]["Policies"]
        return {p["PolicyName"]: p["PolicyDocument"] for p in policy_list}

    def test_quicksight_read_only_has_read_actions(self, policies):
        """QuickSight read policy only includes Describe*/List* APIs."""
        read_policy = policies["QuickSightReadPolicy"]
        for statement in read_policy["Statement"]:
            if statement["Effect"] == "Allow":
                actions = statement["Action"]
                for action in actions:
                    assert action.startswith("quicksight:Describe") or action.startswith(
                        "quicksight:List"
                    ), f"Unexpected non-read action in read policy: {action}"

    def test_quicksight_write_scoped_to_dataset_resource(self, policies):
        """QuickSight write permissions are scoped to dataset resources, not *."""
        write_policy = policies["QuickSightWritePolicy"]
        for statement in write_policy["Statement"]:
            if statement["Effect"] == "Allow":
                resource = statement["Resource"]
                # Resource should reference dataset ARN pattern, not '*'
                assert resource != "*", "QuickSight write policy should not use '*' resource"
                # CFN intrinsics are parsed as dicts (e.g., {'Sub': '...'})
                if isinstance(resource, dict):
                    # Check the Sub expression contains 'dataset'
                    sub_val = resource.get("Sub", "")
                    assert "dataset" in sub_val.lower(), (
                        f"QuickSight write resource should be scoped to datasets: {resource}"
                    )
                elif isinstance(resource, str):
                    assert "dataset" in resource.lower(), (
                        f"QuickSight write resource should be scoped to datasets: {resource}"
                    )

    def test_s3_permissions_scoped_to_optimiser_bucket(self, policies):
        """S3 permissions are scoped to the optimiser bucket, not '*'."""
        s3_policy = policies["S3AccessPolicy"]
        for statement in s3_policy["Statement"]:
            if statement["Effect"] == "Allow":
                resource = statement["Resource"]
                # Resource must not be a plain '*' wildcard
                assert resource != "*", "S3 policy should not use '*' resource"
                # If it's a string, ensure it's not just '*'
                if isinstance(resource, str):
                    assert resource != "*"

    def test_dynamodb_permissions_only_needed_actions(self, policies):
        """DynamoDB permissions only include Query, PutItem, UpdateItem, DeleteItem."""
        ddb_policy = policies["DynamoDBPolicy"]
        allowed_actions = {
            "dynamodb:Query",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:DeleteItem",
        }
        for statement in ddb_policy["Statement"]:
            if statement["Effect"] == "Allow":
                actions = set(statement["Action"])
                extra_actions = actions - allowed_actions
                assert not extra_actions, (
                    f"DynamoDB policy has unexpected actions: {extra_actions}"
                )

    def test_no_wildcard_admin_actions(self, policies):
        """No policy grants admin-level wildcard actions like *:*."""
        for policy_name, policy_doc in policies.items():
            for statement in policy_doc["Statement"]:
                if statement["Effect"] == "Allow":
                    actions = statement["Action"]
                    if isinstance(actions, str):
                        actions = [actions]
                    for action in actions:
                        assert action != "*", (
                            f"Policy {policy_name} should not grant wildcard action"
                        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Required resources present
# ─────────────────────────────────────────────────────────────────────────────


class TestRequiredResources:
    """Verify all required resources are defined in the template."""

    def test_lambda_function_exists(self, template):
        """Template defines a Lambda function with correct configuration."""
        lambdas = {
            k: v
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::Lambda::Function"
        }
        assert len(lambdas) >= 1, "Expected at least one Lambda function"
        func = list(lambdas.values())[0]
        props = func["Properties"]
        assert props["Runtime"] == "python3.14"
        assert props["Timeout"] == 900
        assert props["MemorySize"] == 512

    def test_iam_role_exists(self, template):
        """Template defines an IAM role."""
        roles = [
            k
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::IAM::Role"
        ]
        assert len(roles) >= 1

    def test_events_rule_exists(self, template):
        """Template defines an EventBridge rule."""
        rules = [
            k
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::Events::Rule"
        ]
        assert len(rules) >= 1

    def test_s3_bucket_exists(self, template):
        """Template defines an S3 bucket."""
        buckets = [
            k
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::S3::Bucket"
        ]
        assert len(buckets) >= 1

    def test_dynamodb_table_conditional(self, template):
        """Template defines a DynamoDB table with a condition."""
        tables = {
            k: v
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::DynamoDB::Table"
        }
        assert len(tables) >= 1, "Expected at least one DynamoDB table"
        # Verify it has a Condition attached
        table = list(tables.values())[0]
        assert "Condition" in table, "DynamoDB table should be conditional"

    def test_sns_topic_conditional(self, template):
        """Template defines an SNS topic with a condition."""
        topics = {
            k: v
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::SNS::Topic"
        }
        assert len(topics) >= 1, "Expected at least one SNS topic"
        topic = list(topics.values())[0]
        assert "Condition" in topic, "SNS topic should be conditional"

    def test_cloudwatch_alarm_exists(self, template):
        """Template defines at least one CloudWatch alarm."""
        alarms = [
            k
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::CloudWatch::Alarm"
        ]
        assert len(alarms) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parameter validation
# ─────────────────────────────────────────────────────────────────────────────


class TestParameters:
    """Verify required parameters exist with correct types and defaults."""

    def test_schedule_expression_parameter(self, template):
        param = template["Parameters"]["ScheduleExpression"]
        assert param["Type"] == "String"
        assert param["Default"] == "rate(7 days)"

    def test_s3_bucket_name_parameter(self, template):
        param = template["Parameters"]["S3BucketName"]
        assert param["Type"] == "String"
        assert param["Default"] == ""

    def test_quarantine_enabled_parameter(self, template):
        param = template["Parameters"]["QuarantineEnabled"]
        assert param["Type"] == "String"
        assert param["Default"] == "false"
        assert set(param["AllowedValues"]) == {"true", "false"}

    def test_sns_topic_enabled_parameter(self, template):
        param = template["Parameters"]["SNSTopicEnabled"]
        assert param["Type"] == "String"
        assert param["Default"] == "true"
        assert set(param["AllowedValues"]) == {"true", "false"}

    def test_config_s3_key_parameter(self, template):
        param = template["Parameters"]["ConfigS3Key"]
        assert param["Type"] == "String"
        assert "config" in param["Default"].lower()

    def test_notification_email_parameter(self, template):
        param = template["Parameters"]["NotificationEmail"]
        assert param["Type"] == "String"
        assert param["Default"] == ""

    def test_lambda_code_s3_bucket_parameter(self, template):
        param = template["Parameters"]["LambdaCodeS3Bucket"]
        assert param["Type"] == "String"

    def test_lambda_code_s3_key_parameter(self, template):
        param = template["Parameters"]["LambdaCodeS3Key"]
        assert param["Type"] == "String"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Lambda configuration
# ─────────────────────────────────────────────────────────────────────────────


class TestLambdaConfiguration:
    """Verify Lambda function configuration details."""

    @pytest.fixture
    def lambda_function(self, template):
        """Extract the Lambda function resource."""
        lambdas = {
            k: v
            for k, v in template["Resources"].items()
            if v["Type"] == "AWS::Lambda::Function"
        }
        return list(lambdas.values())[0]

    def test_handler_is_correct(self, lambda_function):
        """Handler is set to lambda_source.handler.handler."""
        assert lambda_function["Properties"]["Handler"] == "lambda_source.handler.handler"

    def test_memory_size_is_512(self, lambda_function):
        """MemorySize is 512 MB."""
        assert lambda_function["Properties"]["MemorySize"] == 512

    def test_timeout_is_900(self, lambda_function):
        """Timeout is 900 seconds (15 minutes max)."""
        assert lambda_function["Properties"]["Timeout"] == 900

    def test_runtime_is_python314(self, lambda_function):
        """Runtime is python3.14, the newest Lambda Python runtime at the time of the
        2026-10 security review; the whole suite passes on CPython 3.14.3. Bump this
        when the next runtime ships, after running the suite on it (DSR control L1)."""
        assert lambda_function["Properties"]["Runtime"] == "python3.14"

    def test_has_environment_variables(self, lambda_function):
        """Lambda has environment variables configured."""
        env = lambda_function["Properties"]["Environment"]["Variables"]
        assert "S3_BUCKET" in env
        assert "CONFIG_S3_KEY" in env
        assert "STACK_NAME" in env

    def test_no_reserved_concurrency(self, lambda_function):
        """Reserved concurrency must NOT be set.

        It was 1 in the initial commit, which serialised all runs, and was removed in
        15b3d39 because the async job pattern self-invokes this function — a reserved
        concurrency of 1 deadlocks the invocation waiting for its own slot.

        The serialisation guarantee it provided now comes from the per-dataset lock in
        lambda_source/dataset_lock.py, which is both narrower (per dataset rather than
        per function) and durable across processes. See tests/unit/test_dataset_lock.py.
        """
        assert "ReservedConcurrentExecutions" not in lambda_function["Properties"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Alerting actually reaches someone, and failed async runs leave a record
# ─────────────────────────────────────────────────────────────────────────────
class TestAlarmsCanPublishToTheEncryptedTopic:
    """The topic was encrypted with alias/aws/sns, whose fixed key policy does not grant
    cloudwatch.amazonaws.com, so every alarm action in this template failed with
    "CloudWatch Alarms does not have authorization to access the SNS topic encryption
    key" and no alarm ever reached anyone. The stack now owns the key. These tests pin
    the three things that make delivery work and keep the key narrow (DSR CW2 / L5 /
    KMS2)."""

    @pytest.fixture
    def key_statements(self, template):
        key = template["Resources"]["OptimiserKey"]
        assert key["Type"] == "AWS::KMS::Key"
        return {s["Sid"]: s for s in key["Properties"]["KeyPolicy"]["Statement"]}

    def test_the_topic_uses_the_stacks_own_key_not_the_aws_managed_one(self, template):
        topic = template["Resources"]["NotificationTopic"]["Properties"]
        assert topic["KmsMasterKeyId"] == {"Ref": "OptimiserKey"}

    def test_cloudwatch_alarms_may_use_the_key_only_for_this_stacks_alarms(self, key_statements):
        grant = key_statements["AllowThisStacksCloudWatchAlarmsToPublish"]
        assert grant["Principal"] == {"Service": "cloudwatch.amazonaws.com"}
        assert set(grant["Action"]) == {"kms:Decrypt", "kms:GenerateDataKey*"}
        assert grant["Condition"]["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}
        assert "alarm:${AWS::StackName}-*" in grant["Condition"]["ArnLike"]["aws:SourceArn"]["Sub"]

    def test_the_function_may_use_the_key_only_through_its_three_services(self, key_statements):
        grant = key_statements["AllowTheOptimiserFunctionThroughItsServices"]
        assert "role/${AWS::StackName}-lambda-role" in grant["Principal"]["AWS"]["Sub"]
        assert set(grant["Action"]) == {"kms:Decrypt", "kms:GenerateDataKey*", "kms:DescribeKey"}
        via = {v["Sub"].split(".")[0] for v in grant["Condition"]["StringEquals"]["kms:ViaService"]}
        assert via == {"lambda", "sns", "dynamodb"}

    def test_cloudwatch_logs_may_use_the_key_only_for_this_stacks_log_group(self, key_statements):
        grant = key_statements["AllowCloudWatchLogsForThisStacksLogGroup"]
        assert grant["Principal"]["Service"]["Sub"].startswith("logs.")
        context = grant["Condition"]["ArnEquals"]["kms:EncryptionContext:aws:logs:arn"]["Sub"]
        assert context.endswith("log-group:/aws/lambda/${AWS::StackName}-function")

    def test_nobody_else_is_granted_key_usage(self, key_statements):
        """Root administration plus the three users of the key, and nothing more."""
        assert set(key_statements) == {
            "AllowKeyAdministrationByTheAccount",
            "AllowTheOptimiserFunctionThroughItsServices",
            "AllowCloudWatchLogsForThisStacksLogGroup",
            "AllowThisStacksCloudWatchAlarmsToPublish",
        }

    def test_the_key_is_retained_and_has_an_alias(self, template):
        key = template["Resources"]["OptimiserKey"]
        assert key["DeletionPolicy"] == "Retain"
        assert template["Resources"]["OptimiserKeyAlias"]["Properties"]["TargetKeyId"] == {"Ref": "OptimiserKey"}

    def test_the_key_rotates(self, template):
        assert template["Resources"]["OptimiserKey"]["Properties"]["EnableKeyRotation"] is True

    def test_the_role_does_not_reference_the_topic_or_the_key(self, template):
        """The key policy names the role, so the role referencing the topic (which
        references the key) would be a dependency cycle. Spelled-out ARN instead."""
        role = template["Resources"]["LambdaExecutionRole"]
        text = str(role)
        assert "NotificationTopic" not in text
        assert "OptimiserKey" not in text

    def test_every_alarm_in_the_template_notifies_the_topic(self, template):
        alarms = {
            k: v for k, v in template["Resources"].items()
            if v["Type"] == "AWS::CloudWatch::Alarm"
        }
        assert len(alarms) >= 4
        for name, alarm in alarms.items():
            actions = alarm["Properties"]["AlarmActions"]["If"]
            assert actions[0] == "IsSNSEnabled", name
            assert actions[1] == [{"Ref": "NotificationTopic"}], name


class TestFailedAsyncInvocationsLeaveARecord:
    """Jobs and scheduled runs are asynchronous invocations. A failure the handler cannot
    catch (timeout, out of memory) used to be retried twice and dropped (DSR L9)."""

    def test_failed_events_land_in_the_dead_letter_queue(self, template):
        func = template["Resources"]["OptimiserFunction"]["Properties"]
        assert func["DeadLetterConfig"] == {"TargetArn": {"GetAtt": "AsyncFailureQueue.Arn"}}

    def test_the_invocation_record_also_goes_to_the_topic_when_there_is_one(self, template):
        cfg = template["Resources"]["OptimiserFunctionInvokeConfig"]
        assert cfg["Type"] == "AWS::Lambda::EventInvokeConfig"
        props = cfg["Properties"]
        assert props["FunctionName"] == {"Ref": "OptimiserFunction"}
        destination = props["DestinationConfig"]["If"]
        assert destination[0] == "IsSNSEnabled"
        assert destination[1] == {"OnFailure": {"Destination": {"Ref": "NotificationTopic"}}}

    def test_a_run_that_mutates_datasets_is_not_silently_retried(self, template):
        cfg = template["Resources"]["OptimiserFunctionInvokeConfig"]["Properties"]
        assert cfg["MaximumRetryAttempts"] == 0

    def test_the_failure_queue_is_encrypted_and_keeps_records_for_a_fortnight(self, template):
        queue = template["Resources"]["AsyncFailureQueue"]
        assert queue["Type"] == "AWS::SQS::Queue"
        assert queue["Properties"]["KmsMasterKeyId"] == "alias/aws/sqs"
        assert queue["Properties"]["MessageRetentionPeriod"] == 1209600

    def test_the_role_may_only_send_to_the_failure_queue(self, template):
        role = template["Resources"]["LambdaExecutionRole"]["Properties"]
        policies = {p["PolicyName"]: p["PolicyDocument"] for p in role["Policies"]}
        statements = policies["AsyncFailureQueuePolicy"]["Statement"]
        assert len(statements) == 1
        assert statements[0]["Action"] == ["sqs:SendMessage"]
        assert statements[0]["Resource"] == {"GetAtt": "AsyncFailureQueue.Arn"}
        # No other policy touches SQS.
        others = [n for n, d in policies.items() if n != "AsyncFailureQueuePolicy" and "sqs:" in str(d)]
        assert others == []

    def test_a_message_on_the_failure_queue_raises_an_alarm(self, template):
        alarm = template["Resources"]["AsyncFailureAlarm"]["Properties"]
        assert alarm["Namespace"] == "AWS/SQS"
        assert alarm["MetricName"] == "ApproximateNumberOfMessagesVisible"
        assert alarm["Dimensions"] == [
            {"Name": "QueueName", "Value": {"GetAtt": "AsyncFailureQueue.QueueName"}}
        ]
        assert alarm["Threshold"] == 1


class TestTracingAndLogLevel:
    """DSR SC7 (X-Ray where the service supports it) and CW3 (a deliberate log level)."""

    def test_xray_tracing_is_active(self, template):
        func = template["Resources"]["OptimiserFunction"]["Properties"]
        assert func["TracingConfig"] == {"Mode": "Active"}

    def test_the_xray_grant_is_the_two_write_actions_and_nothing_else(self, template):
        role = template["Resources"]["LambdaExecutionRole"]["Properties"]
        policies = {p["PolicyName"]: p["PolicyDocument"] for p in role["Policies"]}
        statements = policies["XRayTracingPolicy"]["Statement"]
        assert len(statements) == 1
        assert set(statements[0]["Action"]) == {"xray:PutTraceSegments", "xray:PutTelemetryRecords"}

    def test_the_log_level_is_a_parameter_wired_to_the_function(self, template):
        param = template["Parameters"]["LogLevel"]
        assert param["Default"] == "INFO"
        assert set(param["AllowedValues"]) == {"DEBUG", "INFO", "WARNING", "ERROR"}
        env = template["Resources"]["OptimiserFunction"]["Properties"]["Environment"]["Variables"]
        assert env["LOG_LEVEL"] == {"Ref": "LogLevel"}


class TestEveryResourceWildcardHasAReason:
    """DSR IAM5: a Resource '*' is acceptable only where the action has no resource type,
    and then it must stand alone so the reason is checkable."""

    @pytest.fixture
    def policies(self, template):
        role = template["Resources"]["LambdaExecutionRole"]["Properties"]
        return {p["PolicyName"]: p["PolicyDocument"] for p in role["Policies"]}

    def test_only_listdatasets_is_on_a_wildcard_in_the_quicksight_read_policy(self, policies):
        statements = policies["QuickSightReadPolicy"]["Statement"]
        on_wildcard = [s for s in statements if s["Resource"] == "*"]
        assert len(on_wildcard) == 1
        assert on_wildcard[0]["Action"] == ["quicksight:ListDataSets"]

    def test_the_other_quicksight_reads_are_scoped_to_this_account_and_region(self, policies):
        statements = policies["QuickSightReadPolicy"]["Statement"]
        scoped = {
            action: s["Resource"]["Sub"]
            for s in statements if s["Resource"] != "*"
            for action in s["Action"]
        }
        assert set(scoped) == {
            "quicksight:DescribeDataSet",
            "quicksight:ListAnalyses",
            "quicksight:DescribeAnalysis",
            "quicksight:ListDashboards",
            "quicksight:DescribeDashboard",
        }
        for action, arn in scoped.items():
            assert arn.startswith("arn:aws:quicksight:${AWS::Region}:${AWS::AccountId}:"), action
        assert scoped["quicksight:DescribeDataSet"].endswith(":dataset/*")
        assert scoped["quicksight:DescribeAnalysis"].endswith(":analysis/*")
        assert scoped["quicksight:DescribeDashboard"].endswith(":dashboard/*")

    def test_the_only_wildcard_resources_are_the_actions_without_resource_types(self, policies):
        wildcard_actions = set()
        for doc in policies.values():
            for s in doc["Statement"]:
                if s.get("Resource") == "*":
                    wildcard_actions.update(s["Action"])
        assert wildcard_actions == {
            "quicksight:ListDataSets",
            "cloudwatch:PutMetricData",  # bounded by the cloudwatch:namespace condition
            "xray:PutTraceSegments",
            "xray:PutTelemetryRecords",
        }


class TestBucketAccessLoggingAndTransport:
    """DSR S1 (access logging on in-scope buckets) and EN2 (encryption in transit)."""

    def test_the_optimiser_bucket_logs_access_to_the_log_bucket(self, template):
        props = template["Resources"]["OptimiserBucket"]["Properties"]
        assert props["LoggingConfiguration"]["DestinationBucketName"] == {"Ref": "AccessLogBucket"}
        assert "s3-access-logs/" in props["LoggingConfiguration"]["LogFilePrefix"]["Sub"]
        # S3 validates the destination at creation time, so ordering is pinned.
        assert template["Resources"]["OptimiserBucket"]["DependsOn"] == "AccessLogBucketPolicy"

    def test_the_log_bucket_is_locked_down(self, template):
        props = template["Resources"]["AccessLogBucket"]["Properties"]
        assert props["PublicAccessBlockConfiguration"] == {
            "BlockPublicAcls": True, "BlockPublicPolicy": True,
            "IgnorePublicAcls": True, "RestrictPublicBuckets": True,
        }
        assert props["OwnershipControls"]["Rules"] == [{"ObjectOwnership": "BucketOwnerEnforced"}]
        algo = props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]
        assert algo["ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"
        assert template["Resources"]["AccessLogBucket"]["DeletionPolicy"] == "Retain"

    def test_only_the_s3_log_delivery_service_may_write_logs(self, template):
        statements = template["Resources"]["AccessLogBucketPolicy"]["Properties"]["PolicyDocument"]["Statement"]
        allow = [s for s in statements if s["Effect"] == "Allow"]
        assert len(allow) == 1
        assert allow[0]["Principal"] == {"Service": "logging.s3.amazonaws.com"}
        assert allow[0]["Action"] == "s3:PutObject"
        assert allow[0]["Condition"]["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}

    @pytest.mark.parametrize("policy", ["OptimiserBucketPolicy", "AccessLogBucketPolicy"])
    def test_both_buckets_refuse_plaintext_requests(self, template, policy):
        statements = template["Resources"][policy]["Properties"]["PolicyDocument"]["Statement"]
        deny = [s for s in statements if s["Effect"] == "Deny"]
        assert len(deny) == 1
        assert deny[0]["Principal"] == "*"
        assert deny[0]["Action"] == "s3:*"
        assert deny[0]["Condition"] == {"Bool": {"aws:SecureTransport": "false"}}


class TestScannerFindingsStayClosed:
    """Pins for findings that checkov, cfn_nag and cdk-nag raised during the 2026-10
    security review and that were closed in the template, so they do not reopen."""

    def test_the_failure_queue_refuses_plaintext_requests(self, template):
        statements = template["Resources"]["AsyncFailureQueuePolicy"]["Properties"]["PolicyDocument"]["Statement"]
        assert statements == [{
            "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*", "Action": "sqs:*",
            "Resource": {"GetAtt": "AsyncFailureQueue.Arn"},
            "Condition": {"Bool": {"aws:SecureTransport": "false"}},
        }]

    def test_both_tables_have_point_in_time_recovery(self, template):
        for name in ("QuarantineTable", "McpJobStoreTable"):
            props = template["Resources"][name]["Properties"]
            assert props["PointInTimeRecoverySpecification"] == {"PointInTimeRecoveryEnabled": True}, name

    def test_both_tables_are_encrypted_with_the_stacks_key_and_retained(self, template):
        for name in ("QuarantineTable", "McpJobStoreTable"):
            table = template["Resources"][name]
            assert table["DeletionPolicy"] == "Retain", name
            assert table["Properties"]["SSESpecification"] == {
                "SSEEnabled": True, "SSEType": "KMS", "KMSMasterKeyId": {"Ref": "OptimiserKey"},
            }, name

    def test_environment_variables_and_logs_are_encrypted_with_the_stacks_key(self, template):
        func = template["Resources"]["OptimiserFunction"]
        assert func["Properties"]["KmsKeyArn"] == {"GetAtt": "OptimiserKey.Arn"}
        # Created before the function so Lambda cannot race it with an unencrypted group.
        assert func["DependsOn"] == "OptimiserLogGroup"
        log_group = template["Resources"]["OptimiserLogGroup"]["Properties"]
        assert log_group["KmsKeyId"] == {"GetAtt": "OptimiserKey.Arn"}

    def test_the_user_pool_password_policy_is_strict_even_though_no_human_uses_it(self, template):
        policy = template["Resources"]["CognitoUserPool"]["Properties"]["Policies"]["PasswordPolicy"]
        assert policy["MinimumLength"] >= 12
        assert all(policy[k] for k in ("RequireUppercase", "RequireLowercase", "RequireNumbers", "RequireSymbols"))
