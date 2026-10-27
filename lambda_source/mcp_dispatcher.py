# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tool dispatcher for AgentCore Gateway → Lambda.

Handles 4 tools:
  - list_spice_datasets:   {aws_account_id, region}
  - scan_unused_columns:   {dataset_ids, aws_account_id, region}
  - remove_unused_columns: {dataset_ids, aws_account_id, region, action:"remove"}
  - get_job_result:        {job_id}

This module used to state that "the gateway doesn't pass tool names" and infer the
tool from the payload shape. That premise was wrong. AgentCore Gateway passes the
name on the Lambda context, as
``context.client_context.custom['bedrockAgentCoreToolName']``, formatted
``{target_name}___{tool_name}`` — see
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-add-target-lambda.html

Inferring from shape was the root cause of a live defect. Tool signatures overlap, so
a pipeline event missing one key was silently reinterpreted as a different tool call,
its callback was never written, and the job sat on ``status: "running"`` until its
7-day TTL expired. The caller could not distinguish a working job from a dead one.

Shape inference survives only as a fallback for direct invocation (tests, the AWS
console, a manual `aws lambda invoke`), where no gateway context exists. Every return
path now carries a ``status`` key so no caller has to guess whether it succeeded.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import boto3
from botocore.exceptions import ClientError

from lambda_source.metrics import emit_metric
from lambda_source.qs_client import QSClient, QSClientError

logger = logging.getLogger(__name__)

# Config from environment
JOB_TABLE = os.environ.get("JOB_STORE_TABLE", "spice-optimiser-mcp-jobs")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "spice-optimiser/")
# Empty unless the stack was deployed with QuarantineEnabled=true, which is what makes
# the feature optional: no table means no quarantine, and asking for one is refused
# rather than silently ignored.
QUARANTINE_TABLE = os.environ.get("DYNAMODB_TABLE", "").strip()
FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "")
JOB_TTL = 7 * 24 * 60 * 60  # 7 days

# A Lambda cannot outlive its own timeout (900s per deployment/template.yaml), so a job
# still marked "running" past that plus a margin for cold start and callback write has
# certainly died. Without this a dead job stayed "running" until the 7-day TTL and the
# caller had no way to tell it apart from a live one.
LAMBDA_MAX_DURATION_SECONDS = 900
STALE_JOB_AFTER_SECONDS = LAMBDA_MAX_DURATION_SECONDS + 120


# Delimiter AgentCore Gateway uses between target name and tool name.
_TOOL_NAME_DELIMITER = "___"

KNOWN_TOOLS = frozenset(
    {
        "list_spice_datasets",
        "scan_unused_columns",
        "remove_unused_columns",
        "get_job_result",
        "list_snapshots",
        "rollback_dataset",
    }
)


def correlation_id_from(context: Any) -> str:
    """A stable id tying a response back to its log entries.

    Prefers the gateway's own request id so a trace spans the gateway and the Lambda,
    then the Lambda request id, then a fresh UUID for a direct invocation. Attached to
    every error response, so an operator reporting "it failed" hands over something
    greppable instead of a screenshot.

    This is the last unshipped piece of the deleted ``mcp_server`` test suite's intent —
    input validation, structured errors and job state transitions all arrived under
    other names.
    """
    client_context = getattr(context, "client_context", None)
    custom = getattr(client_context, "custom", None)
    if isinstance(custom, dict):
        gateway_request_id = custom.get("bedrockAgentCoreAwsRequestId")
        if isinstance(gateway_request_id, str) and gateway_request_id:
            return gateway_request_id

    lambda_request_id = getattr(context, "aws_request_id", None)
    if isinstance(lambda_request_id, str) and lambda_request_id:
        return lambda_request_id

    return str(uuid.uuid4())


def dispatch(event: dict, context: Any) -> dict:
    """Route a gateway event to the correct tool handler.

    Prefers the tool name the gateway supplies on the context. Falls back to payload
    shape only when no name is available, which means direct invocation.
    """
    tool = tool_name_from_context(context)
    source = "context"
    if not tool:
        tool = _identify_tool(event)
        source = "payload_shape"

    correlation_id = correlation_id_from(context)

    logger.info(
        "Tool dispatch",
        extra={
            "tool": tool,
            "tool_name_source": source,
            "event_keys": sorted(event.keys()),
            "correlation_id": correlation_id,
            "operation": "tool_dispatch",
        },
    )

    if tool == "list_spice_datasets":
        response = _list_spice_datasets(event)
    elif tool == "scan_unused_columns":
        response = _start_job("dry_run", event)
    elif tool == "remove_unused_columns":
        response = _start_job("apply", event)
    elif tool == "get_job_result":
        response = _get_job_result(event)
    elif tool == "list_snapshots":
        response = _list_snapshots(event)
    elif tool == "rollback_dataset":
        response = _rollback_dataset(event)
    else:
        logger.error(
            "Unrecognised tool dispatch",
            extra={
                "tool": tool,
                "tool_name_source": source,
                "event_keys": sorted(event.keys()),
                "correlation_id": correlation_id,
                "operation": "unrecognised_tool",
            },
        )
        _emit_unrecognised_dispatch_metric(tool, source)
        response = {
            "status": "error",
            "error": (
                f"Unrecognised tool '{tool}'. Resolved from {source}. "
                f"Event keys: {sorted(event.keys())}"
            ),
            "known_tools": sorted(KNOWN_TOOLS),
        }

    # Attached to failures only. On a success the caller already has a job_id or the
    # data it asked for; on a failure this is the one thing that makes "it didn't work"
    # traceable to a specific log entry.
    if response.get("status") == "error":
        response["correlation_id"] = correlation_id

    return response


def _emit_unrecognised_dispatch_metric(tool: str, source: str) -> None:
    """Publish a CloudWatch metric so an unroutable call is actively noticed.

    The condition is already legible in the response and queryable in the logs, but
    nothing was watching. That is how the routing defect survived: the old code printed
    "TOOL_DISPATCH: unknown" on every misroute for weeks and nobody saw it.

    Published with **no dimensions**, deliberately. The alarm in `template.yaml` has no
    dimensions either, and a dimensionless alarm never matches a datapoint published with
    them — an easy way to build an alarm that cannot fire. The tool name and resolution
    source are already on the log line beside this call, which is where a drill-down
    belongs.
    """
    emit_metric("UnrecognisedToolDispatch", 1, "Count")


def tool_name_from_context(context: Any) -> str:
    """Extract the bare tool name from the Lambda context, or "" if unavailable.

    The gateway sends ``{target_name}___{tool_name}``; the target prefix has to be
    stripped by the function. Every lookup here is defensive because direct
    invocations pass ``None`` for context and tests pass stubs — an absent name is a
    normal condition meaning "not a gateway call", not an error.
    """
    client_context = getattr(context, "client_context", None)
    custom = getattr(client_context, "custom", None)
    if not isinstance(custom, dict):
        return ""

    raw = custom.get("bedrockAgentCoreToolName") or ""
    if not isinstance(raw, str) or not raw:
        return ""

    _, _, bare = raw.rpartition(_TOOL_NAME_DELIMITER)
    # rpartition returns ("", "", raw) when the delimiter is absent, which is the
    # right answer for an unprefixed name.
    return (bare or raw).strip()


# ── list_snapshots / rollback_dataset ────────────────────────────────────────

def _snapshot_manager(event: dict):
    """Build a SnapshotManager for the account and region in the event."""
    from lambda_source.snapshot import SnapshotManager

    account = event.get("aws_account_id", "")
    region = event.get("region", "us-east-1")
    return SnapshotManager(
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        qs_client=QSClient(account_id=account, region=region),
    )


def _list_snapshots(event: dict) -> dict:
    """List restorable snapshots for a dataset, newest first.

    Deliberately separate from ``rollback_dataset`` so the agent can show what it would
    restore, and from when, before doing it. Restoring silently from "the most recent
    snapshot" is fine when that is five minutes old and wrong when it is five weeks old.
    """
    from lambda_source.snapshot import SnapshotError

    dataset_id = event.get("dataset_id", "")
    if not dataset_id:
        return {"status": "error", "error": "dataset_id is required"}

    invalid = validate_account_id(event.get("aws_account_id", ""))
    if invalid:
        return {"status": "error", "error": invalid}

    try:
        snapshots = _snapshot_manager(event).list_snapshots(dataset_id)
    except SnapshotError as exc:
        return {"status": "error", "error": str(exc)}

    if not snapshots:
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "snapshots": [],
            "count": 0,
            "message": (
                f"No snapshots exist for dataset {dataset_id}. Snapshots are written "
                "only by remove_unused_columns, so there is nothing to roll back to."
            ),
        }

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "snapshots": snapshots,
        "count": len(snapshots),
        "message": (
            f"{len(snapshots)} snapshot(s) available. The most recent is from "
            f"{snapshots[0]['captured_at']}. rollback_dataset restores the most recent "
            "unless a snapshot_key is given."
        ),
    }


def _rollback_dataset(event: dict) -> dict:
    """Restore a dataset from a snapshot.

    Destructive: it overwrites the dataset's current definition, so anything applied
    since the snapshot is undone. Requires ``action="rollback"`` for the same reason
    ``remove_unused_columns`` requires ``action="remove"`` — the confirmation is verified
    here rather than left to the gateway's schema validation, because a safeguard on an
    operation that rewrites a dataset should not depend on a caller upstream choosing to
    apply it.

    Runs synchronously. ``UpdateDataSet`` returns in seconds and ``CreateIngestion`` is
    itself asynchronous, so there is nothing worth polling for, and an undo that makes
    you poll is a worse undo.
    """
    dataset_id = event.get("dataset_id", "")
    if not dataset_id:
        return {"status": "error", "error": "dataset_id is required"}

    invalid = validate_account_id(event.get("aws_account_id", ""))
    if invalid:
        return {"status": "error", "error": invalid}

    if event.get("action") != "rollback":
        return {
            "status": "error",
            "error": (
                'rollback_dataset requires action="rollback" as an explicit confirmation '
                "that the caller intends to overwrite the dataset's current definition. "
                "Anything applied since the snapshot will be undone. Use list_snapshots "
                "first to see what would be restored."
            ),
        }

    if not S3_BUCKET:
        return {
            "status": "error",
            "error": (
                "S3_BUCKET is not configured on this deployment, so snapshots cannot be "
                "located."
            ),
        }

    from lambda_source.handler import execute_rollback

    return execute_rollback(
        dataset_id=dataset_id,
        aws_account_id=event["aws_account_id"],
        aws_region=event.get("region", "us-east-1"),
        s3_bucket=S3_BUCKET,
        s3_prefix=S3_PREFIX,
        snapshot_key=event.get("snapshot_key") or None,
        trigger_refresh=bool(event.get("trigger_refresh", True)),
    )


def _identify_tool(event: dict) -> str:
    """Infer the tool from payload shape. Fallback for non-gateway invocations.

    Retained only because direct invocation carries no tool name. Do not rely on this
    for the gateway path: the signatures below overlap with the pipeline event shape,
    which is what previously caused pipeline events to be misrouted and their jobs
    orphaned.
    """
    if "job_id" in event:
        return "get_job_result"
    # Singular dataset_id distinguishes the snapshot tools from the plural dataset_ids
    # used by scan and remove. Checked before dataset_ids so a payload carrying both is
    # not silently treated as a scan.
    if "dataset_id" in event:
        if event.get("action") == "rollback":
            return "rollback_dataset"
        return "list_snapshots"
    if "dataset_ids" in event:
        if event.get("action") == "remove":
            return "remove_unused_columns"
        return "scan_unused_columns"
    if "aws_account_id" in event:
        return "list_spice_datasets"
    return "unknown"


# ── list_spice_datasets ──────────────────────────────────────────────────────

def validate_account_id(account: Any) -> str:
    """Return an error message if ``account`` is not a QuickSight account ID.

    Checked before the API call so the caller gets a sentence rather than a botocore
    ``ParamValidationError`` traceback.
    """
    if not account:
        return "aws_account_id is required"
    if not isinstance(account, str):
        return f"aws_account_id must be a string, got {type(account).__name__}"
    if not (len(account) == 12 and account.isdigit()):
        return f"aws_account_id must be 12 digits, got {account!r}"
    return ""


def _list_spice_datasets(event: dict, qs_client: Any | None = None) -> dict:
    """List SPICE datasets. Returns immediately.

    Routed through :class:`QSClient` rather than a private boto3 client. The previous
    implementation built its own client with ``max_attempts=2`` and its own pagination
    loop, so it inherited none of QSClient's throttle handling with jitter, none of its
    error taxonomy, and no describe cache — and it would fail in exactly the large
    accounts the tool targets.

    It also had no seam to substitute, which is why ``tests/integration`` hung
    indefinitely on its first test rather than failing: the test reached real AWS and
    sat in retries.

    Args:
        event: Tool invocation payload.
        qs_client: Optional pre-built client, for tests.
    """
    account = event.get("aws_account_id", "")
    region = event.get("region", "us-east-1")

    invalid = validate_account_id(account)
    if invalid:
        return {"status": "error", "error": invalid}

    client = qs_client or QSClient(account_id=account, region=region)

    try:
        summaries = client.list_datasets()
    except QSClientError as exc:
        return {"status": "error", "error": str(exc)}

    datasets = [
        {
            "dataset_id": ds["DataSetId"],
            "name": ds.get("Name", ""),
            "spice_bytes": ds.get("ConsumedSpiceCapacityInBytes", 0),
        }
        for ds in summaries
        if ds.get("ImportMode") == "SPICE" and ds.get("DataSetId")
    ]

    return {"status": "success", "datasets": datasets, "count": len(datasets)}


# ── start_job (scan or remove) ───────────────────────────────────────────────

def _parse_guardrails(event: dict) -> tuple[dict, str]:
    """Read and validate the two safety parameters the tool schema advertises.

    Returns ``(payload_fragment, error_message)``. A non-empty error means reject the
    call: silently falling back to a default when an operator explicitly asked for a
    stricter threshold is the wrong failure mode for a safety control.

    Args:
        event: The tool invocation payload.

    Returns:
        Tuple of the fragment to merge into the pipeline payload, and an error string
        which is empty when valid.
    """
    fragment: dict[str, Any] = {}

    if "safety_threshold_pct" in event and event["safety_threshold_pct"] is not None:
        raw = event["safety_threshold_pct"]
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            return {}, f"safety_threshold_pct must be a number, got {raw!r}"
        if not 0.0 <= threshold <= 1.0:
            return {}, (
                f"safety_threshold_pct must be between 0.0 and 1.0, got {threshold}"
            )
        fragment["safety_threshold_pct"] = threshold

    if "quarantine_period_days" in event and event["quarantine_period_days"] is not None:
        raw = event["quarantine_period_days"]
        try:
            days = int(raw)
        except (TypeError, ValueError):
            return {}, f"quarantine_period_days must be an integer, got {raw!r}"
        if days < 0:
            return {}, f"quarantine_period_days must be non-negative, got {days}"
        fragment["quarantine_period_days"] = days

    if "cascade_child_refresh" in event and event["cascade_child_refresh"] is not None:
        raw = event["cascade_child_refresh"]
        if isinstance(raw, bool):
            cascade = raw
        elif isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            cascade = raw.strip().lower() == "true"
        else:
            return {}, (
                f"cascade_child_refresh must be a boolean, got {raw!r}"
            )
        fragment["cascade_child_refresh"] = cascade

    return fragment, ""


def _start_job(mode: str, event: dict) -> dict:
    """Create DynamoDB job, async-invoke pipeline, return job_id."""
    dataset_ids = event.get("dataset_ids", [])
    account = event.get("aws_account_id", "")
    region = event.get("region", "us-east-1")

    if not dataset_ids:
        return {"status": "error", "error": "dataset_ids and aws_account_id are required"}

    invalid = validate_account_id(account)
    if invalid:
        return {"status": "error", "error": invalid}

    # Destructive calls must carry the explicit confirmation the schema declares.
    #
    # This check used to be implicit: routing was by payload shape, and
    # `action == "remove"` was the only thing separating a remove from a scan, so
    # omitting it silently downgraded the call to a dry run. Routing by tool name
    # removed that side effect, which would have left the confirmation unenforced.
    #
    # Verified here rather than left to the gateway's schema validation, for the same
    # reason the dataset lock lives in the pipeline and not the dispatcher: a
    # safeguard on a destructive operation should not depend on a caller upstream
    # choosing to apply it.
    if mode == "apply" and event.get("action") != "remove":
        return {
            "status": "error",
            "error": (
                'remove_unused_columns requires action="remove" as an explicit '
                "confirmation that the caller intends to modify the dataset. "
                "Use scan_unused_columns for read-only analysis."
            ),
        }

    # Both guardrails are advertised on the tool schema but were never read here, so
    # the pipeline fell back to its own defaults and quarantine never ran at all on
    # this path — handler gates it on `quarantine_period_days is not None`. Documented,
    # exposed to the agent, and inert.
    guardrails, guardrail_error = _parse_guardrails(event)
    if guardrail_error:
        return {"status": "error", "error": guardrail_error}

    # Refuse rather than proceed without a guardrail the caller asked for. Accepting
    # quarantine_period_days and then ignoring it is how this feature came to be
    # documented, exposed to the agent, and inert — the caller believed columns were
    # protected by a waiting period that was never applied.
    if "quarantine_period_days" in guardrails and not QUARANTINE_TABLE:
        return {
            "status": "error",
            "error": (
                "quarantine_period_days was requested but quarantine is not enabled on "
                "this deployment, so no waiting period can be enforced. Redeploy the "
                "stack with QuarantineEnabled=true to provision the tracking table, or "
                "omit quarantine_period_days to proceed without a waiting period."
            ),
        }

    job_id = str(uuid.uuid4())
    now = int(time.time())

    # Write job record
    ddb = boto3.resource("dynamodb")
    table = ddb.Table(JOB_TABLE)
    table.put_item(Item={
        "job_id": job_id,
        "status": "running",
        "mode": mode,
        "created_at": now,
        "ttl": now + JOB_TTL,
    })

    # Build pipeline payload
    payload = {
        "target_datasets": dataset_ids,
        "execution_mode": mode,
        "aws_account_id": account,
        "aws_region": region,
        "s3_bucket": S3_BUCKET,
        "s3_prefix": S3_PREFIX,
        "callback_job_id": job_id,
        "callback_table": JOB_TABLE,
        # Explicit so the pipeline branch is chosen by declaration rather than by
        # guessing from which keys happen to be present.
        "invocation_source": "mcp_job",
        # Passed the same way s3_bucket and s3_prefix are. Redundant with the env
        # fallback in config.py, deliberately: the payload stays self-describing, so a
        # job record shows which table a run was pointed at.
        "dynamodb_table": QUARANTINE_TABLE,
    }
    payload.update(guardrails)

    # Async self-invoke
    try:
        boto3.client("lambda").invoke(
            FunctionName=FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload).encode(),
        )
    except ClientError as e:
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, error_detail = :e",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "failed", ":e": str(e)},
        )
        return {"job_id": job_id, "status": "failed", "error": str(e)}

    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Job started ({mode} mode). Poll get_job_result with this job_id every 10 seconds.",
        # Echoed so an operator can confirm which guardrails were actually in force,
        # rather than trusting that the values they passed were honoured.
        "effective_guardrails": {
            "safety_threshold_pct": guardrails.get("safety_threshold_pct", 0.7),
            "quarantine_period_days": guardrails.get("quarantine_period_days"),
            "cascade_child_refresh": guardrails.get("cascade_child_refresh", False),
        },
    }


# ── get_job_result ────────────────────────────────────────────────────────────

def _get_job_result(event: dict) -> dict:
    """Read job status from DynamoDB."""
    job_id = event.get("job_id", "")
    if not job_id:
        return {"status": "error", "error": "job_id is required"}

    ddb = boto3.resource("dynamodb")
    table = ddb.Table(JOB_TABLE)

    try:
        resp = table.get_item(Key={"job_id": job_id})
    except ClientError as e:
        return {"status": "error", "error": str(e)}

    item = resp.get("Item")
    if not item:
        return {"status": "error", "error": f"Job {job_id} not found"}

    stored_status = item.get("status", "unknown")
    result: dict[str, Any] = {
        "job_id": job_id,
        "status": stored_status,
        "mode": item.get("mode", ""),
    }

    if stored_status == "completed" and item.get("result"):
        result["result"] = item["result"]
    elif stored_status == "failed":
        result["error_detail"] = item.get("error_detail", "Unknown error")

    if stored_status == "running":
        if _is_stale(item):
            # The worker cannot still be alive: a Lambda cannot outlive its own
            # timeout, so a job still marked running past that point died without
            # writing a callback. Reporting it as running would have the agent poll a
            # corpse for seven days until the TTL expired, which is what it did before.
            detail = (
                "Job exceeded the maximum possible execution time without reporting a "
                "result. The worker most likely timed out or was killed before it "
                "could write back. Check the Lambda logs for this job_id."
            )
            result["status"] = "failed"
            result["error_detail"] = detail
            result["inferred_from_staleness"] = True
            _persist_stale_verdict(table, job_id, detail)
        else:
            result["message"] = "Still running. Poll again in 10 seconds."

    return result


def _persist_stale_verdict(table: Any, job_id: str, detail: str) -> None:
    """Record the inferred terminal state on the job record.

    Without this the tool and the table disagree: ``get_job_result`` says ``failed``
    while the record still says ``running``, so anything reading DynamoDB directly — a
    dashboard, or an operator scanning for stuck jobs — sees the wrong answer, and the
    verdict is recomputed on every poll.

    Conditional on the record still being ``running``, so two concurrent polls cannot
    both write, and a real callback that lands in the same moment always wins rather
    than being overwritten by an inference.

    ``inferred_from_staleness`` is stored alongside, because "the worker reported
    failure" and "we concluded it must have failed" are different facts and a later
    reader should be able to tell them apart.

    Never raises. This runs inside a read path, and failing a status poll because the
    bookkeeping write was throttled would be a worse outcome than a stale record.
    """
    try:
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=(
                "SET #s = :failed, error_detail = :detail, "
                "inferred_from_staleness = :inferred, updated_at = :now"
            ),
            ConditionExpression="#s = :running",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":failed": "failed",
                ":running": "running",
                ":detail": detail,
                ":inferred": True,
                ":now": int(time.time()),
            },
        )
        logger.info(
            "Persisted inferred terminal state for a stale job",
            extra={"job_id": job_id, "operation": "stale_job_persist"},
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            # Something else moved the job to a terminal state first, which is the
            # outcome we wanted anyway.
            logger.info(
                "Stale verdict not written; job already terminal",
                extra={"job_id": job_id, "operation": "stale_job_persist"},
            )
        else:
            logger.warning(
                "Could not persist stale verdict; reporting it read-only",
                extra={"job_id": job_id, "error": str(exc)},
            )
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a poll
        logger.warning(
            "Could not persist stale verdict; reporting it read-only",
            extra={"job_id": job_id, "error": str(exc)},
        )


def _is_stale(item: dict) -> bool:
    """Whether a job still marked running has outlived any possible worker.

    Args:
        item: The DynamoDB job record.

    Returns:
        True when the record is old enough that no worker could still be running.
        A missing or unparseable ``created_at`` returns False — refusing to declare a
        job dead on the basis of data we cannot read is the safer default.
    """
    created_at = item.get("created_at")
    try:
        created = int(created_at)
    except (TypeError, ValueError):
        return False

    return (int(time.time()) - created) > STALE_JOB_AFTER_SECONDS
