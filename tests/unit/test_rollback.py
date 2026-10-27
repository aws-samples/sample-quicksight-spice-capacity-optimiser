# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for the rollback tool.

`SnapshotManager.rollback_dataset()` had unit and property coverage and zero callers,
which is why a serious defect in it survived: its `_build_update_params` was a second
copy of the one in `modifier.py` and had fallen behind, omitting
`DataPrepConfiguration`, `SemanticModelConfiguration` and `PerformanceConfiguration`.
Since `UpdateDataSet` replaces rather than merges, rolling back a new-experience dataset
would have restored its columns and destroyed its whole transform chain — the join, the
casts, the projection. 11 of 66 datasets in the account this was first run against are new-experience.

Tested code with no callers is not verified code. The property tests exercised the
snapshot round-trip on legacy shapes and were entirely satisfied.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from lambda_source import mcp_dispatcher
from lambda_source.dataset_lock import DatasetLock
from lambda_source.mcp_dispatcher import KNOWN_TOOLS, _list_snapshots, _rollback_dataset
from lambda_source.qs_client import QSClient
from lambda_source.snapshot import SnapshotManager
from lambda_source.update_params import build_update_params

ACCOUNT = "9" * 12
# The project root, NOT agentcore/. The CDK resolves `toolSchemaFile` in agentcore.json
# relative to the project directory, so this is the file the gateway is actually built
# from. A copy previously sat under agentcore/ as well; `03acfea` deleted the wrong one of
# the two as "unreferenced", which broke `agentcore deploy` with
# "Cannot find asset at .../spiceoptimisermcp/tool-schema.json". One file now, and this
# test reads the same path the deploy does.
SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "agentcore_project"
    / "spiceoptimisermcp"
    / "tool-schema.json"
)


class TestSchemaAgreement:
    """The drift that made the gateway advertise five tools the Lambda had never heard
    of. Checked as a one-off during that fix but never encoded, so it could recur."""

    def test_the_schema_declares_exactly_the_implemented_tools(self):
        declared = {tool["name"] for tool in json.loads(SCHEMA.read_text())}

        assert declared == set(KNOWN_TOOLS), (
            f"schema-only: {sorted(declared - KNOWN_TOOLS)}, "
            f"code-only: {sorted(KNOWN_TOOLS - declared)}"
        )

    @pytest.mark.parametrize("tool", ["rollback_dataset", "remove_unused_columns"])
    def test_destructive_tools_require_an_explicit_action(self, tool):
        spec = next(t for t in json.loads(SCHEMA.read_text()) if t["name"] == tool)

        assert "action" in spec["inputSchema"]["required"]
        assert spec["inputSchema"]["properties"]["action"]["enum"]

    def test_every_tool_has_a_description(self):
        for tool in json.loads(SCHEMA.read_text()):
            assert tool.get("description"), f"{tool['name']} has no description"


class TestUpdateParamsSharedBuilder:
    """One builder, so snapshot and modifier cannot diverge again."""

    NEW_EXPERIENCE = {
        "Name": "customers_SO.json",
        "ImportMode": "SPICE",
        "PhysicalTableMap": {"pt": {"FileSource": {"InputColumns": [{"Name": "a"}]}}},
        # New-experience datasets return this as None rather than omitting it.
        "LogicalTableMap": None,
        "DataPrepConfiguration": {
            "TransformStepMap": {
                "s1": {"JoinStep": {"OnClause": "{a} = {b}"}},
                "s2": {"CastColumnTypesStep": {}},
            }
        },
        "SemanticModelConfiguration": {"some": "config"},
        "PerformanceConfiguration": {"tier": "x"},
    }

    LEGACY = {
        "Name": "inventories.csv",
        "ImportMode": "SPICE",
        "PhysicalTableMap": {"pt": {"RelationalTable": {}}},
        "LogicalTableMap": {"lt": {"Source": {"PhysicalTableId": "pt"}}},
    }

    def test_a_new_experience_dataset_keeps_its_transforms(self):
        """The defect. Without DataPrepConfiguration in the request, UpdateDataSet
        deletes every transform the dataset had."""
        params = build_update_params(self.NEW_EXPERIENCE)

        assert params["DataPrepConfiguration"] == (
            self.NEW_EXPERIENCE["DataPrepConfiguration"]
        )
        assert "LogicalTableMap" not in params

    def test_a_new_experience_dataset_keeps_its_semantic_model(self):
        params = build_update_params(self.NEW_EXPERIENCE)

        assert params["SemanticModelConfiguration"] == {"some": "config"}
        assert params["PerformanceConfiguration"] == {"tier": "x"}

    def test_a_null_logical_table_map_is_never_sent(self):
        """`"LogicalTableMap" in dataset_def` is true for a new-experience dataset,
        because the key is present with a None value. A truthiness check is required."""
        params = build_update_params(self.NEW_EXPERIENCE)

        assert "LogicalTableMap" not in params

    def test_a_legacy_dataset_keeps_its_logical_table_map(self):
        params = build_update_params(self.LEGACY)

        assert params["LogicalTableMap"] == self.LEGACY["LogicalTableMap"]
        assert "DataPrepConfiguration" not in params

    def test_both_builders_are_the_same_function(self):
        from lambda_source.modifier import DatasetModifier
        from lambda_source.snapshot import SnapshotManager

        definition = self.NEW_EXPERIENCE
        assert DatasetModifier._build_update_params(
            definition
        ) == SnapshotManager._build_update_params(definition)

    def test_unknown_fields_are_dropped_not_forwarded(self):
        """UpdateDataSet rejects unexpected parameters, so only known fields pass."""
        params = build_update_params(dict(self.LEGACY, SomeFutureField={"x": 1}))

        assert "SomeFutureField" not in params


class TestListSnapshots:
    def _call(self, contents, **overrides):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": contents}
        event = {"dataset_id": "ds-1", "aws_account_id": ACCOUNT}
        event.update(overrides)
        with patch.object(mcp_dispatcher, "S3_BUCKET", "bucket"), patch(
            "lambda_source.snapshot.boto3"
        ) as snap_boto3, patch.object(mcp_dispatcher, "QSClient"):
            snap_boto3.client.return_value = s3
            return _list_snapshots(event)

    def _obj(self, key, when):
        return {"Key": key, "LastModified": when, "Size": 4003}

    def test_snapshots_are_returned_newest_first(self):
        older = datetime(2026, 8, 9, 9, 5, tzinfo=timezone.utc)
        newer = datetime(2026, 9, 11, 20, 33, tzinfo=timezone.utc)
        result = self._call(
            [self._obj("p/snapshots/ds-1/old.json", older),
             self._obj("p/snapshots/ds-1/new.json", newer)]
        )

        assert result["status"] == "success"
        assert result["count"] == 2
        assert result["snapshots"][0]["captured_at"] == "2026-09-11T20:33:00Z"

    def test_each_snapshot_reports_when_it_was_captured(self):
        """An operator undoing something needs to know whether the newest snapshot is
        five minutes or five weeks old."""
        result = self._call(
            [self._obj("p/snapshots/ds-1/a.json",
                       datetime(2026, 9, 11, 20, 33, tzinfo=timezone.utc))]
        )
        snapshot = result["snapshots"][0]

        assert snapshot["snapshot_key"] == "p/snapshots/ds-1/a.json"
        assert snapshot["captured_at"] == "2026-09-11T20:33:00Z"
        assert snapshot["size_bytes"] == 4003

    def test_no_snapshots_says_so_and_explains_why(self):
        result = self._call([])

        assert result["status"] == "success"
        assert result["snapshots"] == []
        assert "remove_unused_columns" in result["message"]

    def test_dataset_id_is_required(self):
        assert _list_snapshots({"aws_account_id": ACCOUNT})["status"] == "error"

    def test_account_id_is_validated(self):
        result = _list_snapshots({"dataset_id": "ds-1", "aws_account_id": "123"})

        assert result["status"] == "error"
        assert "12 digits" in result["error"]


class TestSnapshotKeyBelongsToDataset:
    """A caller-supplied ``snapshot_key`` must be one of the target dataset's own.

    The key comes straight from the tool payload. Before this check, rollback_dataset
    handed it to get_object and then UpdateDataSet unexamined, so anyone allowed to call
    the tool could overwrite dataset A with the definition stored for dataset B just by
    naming B's key (DSR control SC1, broken access control). The check lives in
    SnapshotManager, the layer that performs the write, so it holds for every caller.
    """

    def _manager(self, body: dict | None = None):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(body or {"DataSetId": "ds-1"}).encode())
        }
        s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "p/snapshots/ds-1/newest.json", "LastModified": 1}]
        }
        qs = create_autospec(QSClient, instance=True)
        qs.update_dataset.return_value = {"Status": 200}
        manager = SnapshotManager(s3_bucket="bucket", s3_prefix="p/", qs_client=qs, s3_client=s3)
        return manager, s3, qs

    @pytest.mark.parametrize(
        "foreign_key",
        [
            "p/snapshots/ds-2/2026-09-11T20:33:00Z.json",  # another dataset's snapshot
            "p/snapshots/ds-10/2026-09-11T20:33:00Z.json",  # prefix-of-the-id lookalike
            "p/reports/ds-1/report.json",  # same dataset, not a snapshot
            "other/snapshots/ds-1/x.json",  # outside the stack prefix
            "p/snapshots/ds-1/",  # the bare prefix, not an object
            "2026-09-11T20:33:00Z.json",  # a relative-looking key
        ],
    )
    def test_a_key_outside_the_dataset_prefix_is_refused_before_any_read_or_write(
        self, foreign_key
    ):
        from lambda_source.snapshot import SnapshotError

        manager, s3, qs = self._manager()

        with pytest.raises(SnapshotError) as excinfo:
            manager.rollback_dataset("ds-1", snapshot_key=foreign_key)

        assert "does not belong to dataset ds-1" in str(excinfo.value)
        assert "list_snapshots" in str(excinfo.value)
        s3.get_object.assert_not_called()
        qs.update_dataset.assert_not_called()

    def test_a_key_under_the_dataset_prefix_is_restored(self):
        manager, s3, qs = self._manager()

        manager.rollback_dataset("ds-1", snapshot_key="p/snapshots/ds-1/2026-09-11T20:33:00Z.json")

        s3.get_object.assert_called_once_with(
            Bucket="bucket", Key="p/snapshots/ds-1/2026-09-11T20:33:00Z.json"
        )
        qs.update_dataset.assert_called_once()

    def test_the_check_uses_the_same_prefix_the_writer_uses(self):
        """If the writer and the check ever disagreed about the prefix, every rollback
        would be refused. They share one helper; this pins that."""
        manager, _, _ = self._manager()

        written = manager._generate_s3_key("ds-1")

        assert written.startswith(manager._snapshot_prefix("ds-1"))
        manager._require_snapshot_belongs_to("ds-1", written)  # does not raise

    def test_a_snapshot_describing_another_dataset_is_refused_even_under_the_right_prefix(self):
        """Second line of defence: the snapshot body names the dataset it came from."""
        from lambda_source.snapshot import SnapshotError

        manager, _, qs = self._manager(body={"DataSetId": "ds-2", "Name": "other"})

        with pytest.raises(SnapshotError) as excinfo:
            manager.rollback_dataset("ds-1", snapshot_key="p/snapshots/ds-1/planted.json")

        assert "describes dataset ds-2" in str(excinfo.value)
        qs.update_dataset.assert_not_called()

    def test_no_key_still_means_the_most_recent_snapshot(self):
        manager, s3, qs = self._manager()

        manager.rollback_dataset("ds-1")

        s3.get_object.assert_called_once_with(Bucket="bucket", Key="p/snapshots/ds-1/newest.json")
        qs.update_dataset.assert_called_once()

    def test_the_refusal_reaches_the_tool_caller_as_a_clean_error(self):
        """End to end through execute_rollback: the lock is taken, the restore is refused,
        and the caller gets status=error rather than a traceback."""
        from lambda_source.handler import execute_rollback

        manager, _, qs = self._manager()
        lock = create_autospec(DatasetLock, instance=True)
        with patch("lambda_source.handler.QSClient", return_value=qs), patch(
            "lambda_source.handler.SnapshotManager", return_value=manager
        ), patch("lambda_source.handler.DatasetLock", return_value=lock):
            result = execute_rollback(
                dataset_id="ds-1",
                aws_account_id=ACCOUNT,
                aws_region="us-east-1",
                s3_bucket="bucket",
                s3_prefix="p/",
                snapshot_key="p/snapshots/ds-2/2026-09-11T20:33:00Z.json",
            )

        assert result["status"] == "error"
        assert "does not belong to dataset ds-1" in result["error"]
        qs.update_dataset.assert_not_called()
        qs.create_ingestion.assert_not_called()


class TestRollbackConfirmation:
    """Destructive, so the confirmation is verified in the Lambda rather than left to
    the gateway's schema validation."""

    def _call(self, **overrides):
        event = {"dataset_id": "ds-1", "aws_account_id": ACCOUNT}
        event.update(overrides)
        with patch.object(mcp_dispatcher, "S3_BUCKET", "bucket"):
            return _rollback_dataset(event)

    def test_without_confirmation_it_is_refused(self):
        result = self._call()

        assert result["status"] == "error"
        assert 'action="rollback"' in result["error"]

    @pytest.mark.parametrize("value", ["Rollback", "ROLLBACK", "remove", "", None, True])
    def test_only_the_exact_literal_counts(self, value):
        assert self._call(action=value)["status"] == "error"

    def test_the_refusal_points_at_list_snapshots(self):
        """So the agent's next step is to show the user what would be restored."""
        assert "list_snapshots" in self._call()["error"]

    def test_the_refusal_warns_that_changes_are_undone(self):
        assert "undone" in self._call()["error"]

    def test_dataset_id_is_required(self):
        result = _rollback_dataset({"aws_account_id": ACCOUNT, "action": "rollback"})

        assert result["status"] == "error"
        assert "dataset_id" in result["error"]

    def test_a_missing_bucket_is_reported_rather_than_guessed(self):
        with patch.object(mcp_dispatcher, "S3_BUCKET", ""):
            result = _rollback_dataset(
                {"dataset_id": "ds-1", "aws_account_id": ACCOUNT, "action": "rollback"}
            )

        assert result["status"] == "error"
        assert "S3_BUCKET" in result["error"]


class TestRollbackRouting:
    """Singular dataset_id distinguishes the snapshot tools from the plural dataset_ids
    used by scan and remove."""

    @pytest.mark.parametrize(
        "event,expected",
        [
            ({"dataset_id": "ds-1"}, "list_snapshots"),
            ({"dataset_id": "ds-1", "action": "rollback"}, "rollback_dataset"),
            ({"dataset_ids": ["ds-1"]}, "scan_unused_columns"),
            ({"dataset_ids": ["ds-1"], "action": "remove"}, "remove_unused_columns"),
        ],
    )
    def test_shape_inference_distinguishes_them(self, event, expected):
        assert mcp_dispatcher._identify_tool(event) == expected

    def test_a_payload_with_both_prefers_the_singular_form(self):
        """Ambiguous input must not silently become a scan of many datasets."""
        assert mcp_dispatcher._identify_tool(
            {"dataset_id": "ds-1", "dataset_ids": ["a", "b"]}
        ) == "list_snapshots"

    @pytest.mark.parametrize("tool", ["list_snapshots", "rollback_dataset"])
    def test_the_gateway_tool_name_routes_correctly(self, tool):
        from types import SimpleNamespace

        from lambda_source.mcp_dispatcher import tool_name_from_context

        ctx = SimpleNamespace(
            client_context=SimpleNamespace(
                custom={"bedrockAgentCoreToolName": f"spice-optimiser-lambda___{tool}"}
            )
        )
        assert tool_name_from_context(ctx) == tool


class TestExecuteRollback:
    """The orchestration: lock held, restore, refresh."""

    def _run(self, *, lock_raises=None, ingestion_raises=None, snapshot_raises=None):
        from lambda_source.handler import execute_rollback

        # spec'd against the real QSClient so the mock enforces real signatures. A bare
        # MagicMock accepts any call, which is how create_ingestion(dataset_id) - one
        # argument to a two-argument method - reached production. The test below then
        # asserted that same broken call, so the suite was holding the defect in place
        # rather than catching it.
        qs = create_autospec(QSClient, instance=True)
        if ingestion_raises:
            qs.create_ingestion.side_effect = ingestion_raises
        snapshot_manager = create_autospec(SnapshotManager, instance=True)
        snapshot_manager._find_most_recent_snapshot.return_value = "p/snap/newest.json"
        if snapshot_raises:
            snapshot_manager.rollback_dataset.side_effect = snapshot_raises
        lock = create_autospec(DatasetLock, instance=True)
        if lock_raises:
            lock.held.side_effect = lock_raises

        with patch("lambda_source.handler.QSClient", return_value=qs), patch(
            "lambda_source.handler.SnapshotManager", return_value=snapshot_manager
        ), patch("lambda_source.handler.DatasetLock", return_value=lock):
            result = execute_rollback(
                dataset_id="ds-1",
                aws_account_id=ACCOUNT,
                aws_region="us-east-1",
                s3_bucket="bucket",
                s3_prefix="p/",
            )
        return result, qs, snapshot_manager, lock

    def test_a_successful_rollback_restores_and_refreshes(self):
        result, qs, snapshots, _ = self._run()

        assert result["status"] == "success"
        assert result["restored_from_snapshot"] == "p/snap/newest.json"
        assert result["refresh_started"] is True
        snapshots.rollback_dataset.assert_called_once()

        # Both arguments, and the id generated rather than passed in. Asserted on shape
        # rather than an exact value because it carries a uuid.
        dataset_arg, ingestion_arg = qs.create_ingestion.call_args.args
        assert dataset_arg == "ds-1"
        assert ingestion_arg.startswith("optimiser-rollback-")

    def test_the_lock_is_held_across_the_restore(self):
        """Rollback is a describe-modify-write, so it must not race an optimisation."""
        _, _, _, lock = self._run()

        lock.held.assert_called_once()
        assert lock.held.call_args[0][0] == "ds-1"

    def test_a_contended_dataset_is_refused_with_the_holder_named(self):
        from lambda_source.dataset_lock import DatasetLockedError

        exc = DatasetLockedError("locked", holder="exec-99", expires_at="later")
        result, _, snapshots, _ = self._run(lock_raises=exc)

        assert result["status"] == "error"
        assert "exec-99" in result["error"]
        snapshots.rollback_dataset.assert_not_called()

    def test_a_failed_refresh_does_not_fail_the_rollback(self):
        """The restore is the part that matters. Stale data is recoverable by
        refreshing; reporting the whole rollback as failed would be misleading."""
        from lambda_source.qs_client import QSClientError

        result, _, _, _ = self._run(ingestion_raises=QSClientError("throttled"))

        assert result["status"] == "success"
        assert result["refresh_started"] is False
        assert "throttled" in result["ingestion_error"]
        assert "empty until" in result["message"]

    @pytest.mark.parametrize(
        "exc",
        [
            TypeError("create_ingestion() missing 1 required positional argument"),
            KeyError("IngestionId"),
            ValueError("bad ingestion id"),
            RuntimeError("something unforeseen"),
        ],
        ids=["TypeError", "KeyError", "ValueError", "RuntimeError"],
    )
    def test_no_refresh_failure_of_any_kind_fails_the_rollback(self, exc):
        """The catch has to be as wide as the promise the docstring makes.

        This is the case that actually broke in production. The catch was
        ``except QSClientError``, and the failure was a TypeError from calling
        create_ingestion with one argument instead of two - so it escaped to the
        top-level handler and the caller was told the rollback had failed, seconds
        after UpdateDataSet had already restored the dataset.

        Telling someone their undo failed when it succeeded is the worst available
        outcome: it invites them to retry, or to go and fix by hand something that is
        no longer broken. Parametrised over unrelated exception types because the
        promise is about the restore being durable, not about which errors were
        anticipated.
        """
        result, _, snapshots, _ = self._run(ingestion_raises=exc)

        assert result["status"] == "success", (
            f"a {type(exc).__name__} while starting the refresh reported the rollback "
            "as failed, but the definition was already restored"
        )
        assert result["refresh_started"] is False
        assert type(exc).__name__ in result["ingestion_error"]
        snapshots.rollback_dataset.assert_called_once()

    def test_a_missing_snapshot_is_reported_clearly(self):
        from lambda_source.snapshot import SnapshotError

        result, _, _, _ = self._run(
            snapshot_raises=SnapshotError("No snapshots found for dataset ds-1")
        )

        assert result["status"] == "error"
        assert "No snapshots found" in result["error"]

    def test_the_message_explains_that_data_needs_a_refresh(self):
        result, _, _, _ = self._run()

        assert "refresh" in result["message"].lower()


class TestUnrefreshableDatasets:
    """A dataset with no refreshable source is not a rollback that went wrong.

    Found on a live rollback of a file-upload dataset. The restore succeeded, and the
    caller was told: "No refresh was started, so restored columns will be empty until
    the dataset is refreshed." Both halves of that are wrong for this dataset. Nothing
    was going to refresh it, and the data was not empty - QuickSight had already re-read
    the stored file when the definition changed.

    QuickSight cannot refresh a dataset built from an uploaded file because it keeps no
    connection or location metadata to re-read from, so CreateIngestion is rejected
    outright rather than failing transiently. That is a documented product constraint,
    not a transient error and not an SDK version problem:
    https://docs.aws.amazon.com/quicksight/latest/user/refreshing-data.html
    """

    def _rollback(self, ingestion_raises):
        from lambda_source.handler import execute_rollback

        qs = create_autospec(QSClient, instance=True)
        qs.create_ingestion.side_effect = ingestion_raises
        snapshot_manager = create_autospec(SnapshotManager, instance=True)
        snapshot_manager._find_most_recent_snapshot.return_value = "p/snap/newest.json"

        with patch("lambda_source.handler.QSClient", return_value=qs), patch(
            "lambda_source.handler.SnapshotManager", return_value=snapshot_manager
        ), patch("lambda_source.handler.DatasetLock", return_value=MagicMock()):
            return execute_rollback(
                dataset_id="ds-1",
                aws_account_id=ACCOUNT,
                aws_region="us-east-1",
                s3_bucket="bucket",
                s3_prefix="p/",
            )

    def test_an_unrefreshable_dataset_is_not_described_as_pending_a_refresh(self):
        """The exact live message that prompted this. Verbatim from QuickSight."""
        from lambda_source.qs_client import QSClientError

        result = self._rollback(
            QSClientError(
                "Unhandled error calling create_ingestion: "
                "[InvalidParameterValueException] DataSet is not refreshable "
                "because of the source type"
            )
        )

        assert result["status"] == "success"
        assert result["refresh_unsupported"] is True
        assert "none is needed" in result["message"]
        assert "empty until" not in result["message"], (
            "told the caller their restored columns would be empty, when QuickSight "
            "had already re-read the uploaded file"
        )

    def test_the_message_says_what_to_do_instead(self):
        """"Nothing is wrong" is only useful with "and here is how to refresh it"."""
        from lambda_source.qs_client import QSClientError

        result = self._rollback(
            QSClientError("[InvalidParameterValueException] DataSet is not refreshable because of the source type")
        )

        assert "S3 data source" in result["message"]

    def test_a_genuine_refresh_failure_is_still_reported_as_one(self):
        """The narrow point of matching on the phrase and not the exception type.

        InvalidParameterValueException covers a great many unrelated problems.
        Swallowing all of them as "no refresh needed" would hide real failures behind a
        reassuring message, which is the same class of mistake in the other direction.
        """
        from lambda_source.qs_client import QSClientError

        result = self._rollback(
            QSClientError("[InvalidParameterValueException] IngestionId already exists")
        )

        assert result["status"] == "success"
        assert result["refresh_unsupported"] is False
        assert "empty until" in result["message"]
        assert "none is needed" not in result["message"]

    def test_throttling_is_a_real_failure_not_an_unsupported_source(self):
        from lambda_source.qs_client import QSClientError

        result = self._rollback(QSClientError("ThrottlingException: rate exceeded"))

        assert result["refresh_unsupported"] is False
        assert "empty until" in result["message"]
