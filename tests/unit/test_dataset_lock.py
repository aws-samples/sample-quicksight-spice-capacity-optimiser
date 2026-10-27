# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Per-dataset lock, replacing the concurrency guarantee dropped in 15b3d39.

`UpdateDataSet` is a full-definition replace with no conditional check, so two
overlapping apply-mode runs on one dataset are a lost update: both describe, both
write, and the second silently discards the first's column removals.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from botocore.exceptions import ClientError

from lambda_source.dataset_lock import (
    DatasetLock,
    DatasetLockedError,
    DatasetLockError,
    LockInfo,
)

NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


class ConditionalFakeS3:
    """In-memory S3 honouring IfNoneMatch and IfMatch, as real S3 does."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.put_calls: list[dict] = []
        self.deletes: list[str] = []
        self._counter = 0

    @staticmethod
    def _precondition_failed(op: str) -> ClientError:
        return ClientError(
            {"Error": {"Code": "PreconditionFailed", "Message": "failed"}}, op
        )

    def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
        self.put_calls.append({"Key": Key, **kwargs})

        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise self._precondition_failed("PutObject")
        if "IfMatch" in kwargs and self.etags.get(Key) != kwargs["IfMatch"]:
            raise self._precondition_failed("PutObject")

        self._counter += 1
        # Real S3 accepts str or bytes and always returns bytes on read.
        self.objects[Key] = Body.encode("utf-8") if isinstance(Body, str) else Body
        self.etags[Key] = f'"etag-{self._counter}"'
        return {"ETag": self.etags[Key]}

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )

        class Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": Body(self.objects[Key]), "ETag": self.etags[Key]}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deletes.append(Key)
        self.objects.pop(Key, None)
        self.etags.pop(Key, None)
        return {}


@pytest.fixture
def s3() -> ConditionalFakeS3:
    return ConditionalFakeS3()


@pytest.fixture
def lock(s3: ConditionalFakeS3) -> DatasetLock:
    return DatasetLock(
        s3_bucket="my-bucket", s3_prefix="optimiser/", s3_client=s3, ttl_minutes=30
    )


class TestAcquire:
    def test_first_acquire_succeeds(self, lock: DatasetLock, s3: ConditionalFakeS3):
        lock.acquire("ds-1", execution_id="exec-1", now=NOW)

        assert "optimiser/locks/ds-1.json" in s3.objects

    def test_lock_records_holder_expiry_and_execution(
        self, lock: DatasetLock, s3: ConditionalFakeS3
    ):
        lock.acquire("ds-1", execution_id="exec-1", now=NOW)
        body = json.loads(s3.objects["optimiser/locks/ds-1.json"])

        assert body["holder"]
        assert body["acquired_at"] == NOW.isoformat()
        assert body["expires_at"] == (NOW + timedelta(minutes=30)).isoformat()
        assert body["execution_id"] == "exec-1"

    def test_creation_is_conditional(self, lock: DatasetLock, s3: ConditionalFakeS3):
        """Without IfNoneMatch this would be a plain overwrite and useless as a lock."""
        lock.acquire("ds-1", now=NOW)

        assert s3.put_calls[0]["IfNoneMatch"] == "*"

    def test_lock_is_encrypted_at_rest(self, lock: DatasetLock, s3: ConditionalFakeS3):
        lock.acquire("ds-1", now=NOW)

        assert s3.put_calls[0]["ServerSideEncryption"] == "AES256"

    def test_key_layout(self, lock: DatasetLock):
        assert lock.generate_lock_key("ds-1") == "optimiser/locks/ds-1.json"

    def test_prefix_without_trailing_slash_is_normalised(self, s3: ConditionalFakeS3):
        lock = DatasetLock(s3_bucket="b", s3_prefix="optimiser", s3_client=s3)
        assert lock.generate_lock_key("ds-1") == "optimiser/locks/ds-1.json"


class TestContention:
    def test_second_run_is_refused(self, lock: DatasetLock):
        lock.acquire("ds-1", execution_id="exec-1", now=NOW)

        with pytest.raises(DatasetLockedError):
            lock.acquire("ds-1", execution_id="exec-2", now=NOW)

    def test_refusal_carries_holder_and_expiry(self, lock: DatasetLock):
        lock.acquire("ds-1", execution_id="exec-1", now=NOW)

        with pytest.raises(DatasetLockedError) as exc:
            lock.acquire("ds-1", now=NOW)

        assert exc.value.dataset_id == "ds-1"
        assert exc.value.holder
        assert exc.value.expires_at

    def test_the_second_run_does_not_overwrite_the_lock(
        self, lock: DatasetLock, s3: ConditionalFakeS3
    ):
        lock.acquire("ds-1", execution_id="first", now=NOW)

        with pytest.raises(DatasetLockedError):
            lock.acquire("ds-1", execution_id="second", now=NOW)

        body = json.loads(s3.objects["optimiser/locks/ds-1.json"])
        assert body["execution_id"] == "first"

    def test_different_datasets_do_not_contend(self, lock: DatasetLock):
        lock.acquire("ds-1", now=NOW)
        lock.acquire("ds-2", now=NOW)


class TestExpiryIsADeadMansSwitch:
    def test_expired_lock_can_be_stolen(self, lock: DatasetLock):
        lock.acquire("ds-1", execution_id="dead-run", now=NOW)

        info = lock.acquire("ds-1", execution_id="new-run", now=NOW + timedelta(minutes=31))

        assert info.execution_id == "new-run"

    def test_live_lock_cannot_be_stolen(self, lock: DatasetLock):
        lock.acquire("ds-1", now=NOW)

        with pytest.raises(DatasetLockedError):
            lock.acquire("ds-1", now=NOW + timedelta(minutes=29))

    def test_steal_is_conditional_on_etag(self, s3: ConditionalFakeS3):
        """Two processes finding the same expired lock must not both win."""
        lock = DatasetLock(s3_bucket="b", s3_client=s3, ttl_minutes=1)
        lock.acquire("ds-1", now=NOW)

        lock.acquire("ds-1", now=NOW + timedelta(minutes=5))

        steal = s3.put_calls[-1]
        assert "IfMatch" in steal
        assert "IfNoneMatch" not in steal

    def test_corrupt_lock_is_treated_as_stale(self, s3: ConditionalFakeS3):
        """A lock nobody can parse must not block a dataset forever."""
        s3.objects["locks/ds-1.json"] = b"{not json"
        s3.etags["locks/ds-1.json"] = '"etag-x"'
        lock = DatasetLock(s3_bucket="b", s3_client=s3)

        info = lock.acquire("ds-1", execution_id="recovering", now=NOW)

        assert info.execution_id == "recovering"


class TestRelease:
    def test_release_deletes_the_lock(self, lock: DatasetLock, s3: ConditionalFakeS3):
        lock.acquire("ds-1", now=NOW)
        lock.release("ds-1")

        assert "optimiser/locks/ds-1.json" not in s3.objects

    def test_release_never_raises(self, s3: ConditionalFakeS3):
        """Called from a finally, where raising would mask the real failure."""

        class Broken(ConditionalFakeS3):
            def delete_object(self, **kwargs):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "no"}}, "DeleteObject"
                )

        lock = DatasetLock(s3_bucket="b", s3_client=Broken())
        lock.acquire("ds-1", now=NOW)
        lock.release("ds-1")

    def test_reacquirable_after_release(self, lock: DatasetLock):
        lock.acquire("ds-1", now=NOW)
        lock.release("ds-1")
        lock.acquire("ds-1", now=NOW)


class TestHeldContextManager:
    def test_releases_on_normal_exit(self, lock: DatasetLock, s3: ConditionalFakeS3):
        with lock.held("ds-1"):
            assert "optimiser/locks/ds-1.json" in s3.objects

        assert "optimiser/locks/ds-1.json" not in s3.objects

    def test_releases_on_exception(self, lock: DatasetLock, s3: ConditionalFakeS3):
        """Covers the several early-return paths in the apply sequence."""
        with pytest.raises(RuntimeError):
            with lock.held("ds-1"):
                raise RuntimeError("UpdateDataSet blew up")

        assert "optimiser/locks/ds-1.json" not in s3.objects

    def test_releases_on_early_return(self, lock: DatasetLock, s3: ConditionalFakeS3):
        def apply_with_early_return():
            with lock.held("ds-1"):
                return "snapshot failed, bailing out"

        assert apply_with_early_return()
        assert "optimiser/locks/ds-1.json" not in s3.objects


class TestUnreadableLock:
    def test_a_read_failure_is_raised_not_swallowed(self, s3: ConditionalFakeS3):
        """Refusing to modify without a lock beats modifying without one."""

        class Unreadable(ConditionalFakeS3):
            def get_object(self, **kwargs):
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject"
                )

        broken = Unreadable()
        lock = DatasetLock(s3_bucket="b", s3_client=broken)
        lock.acquire("ds-1", now=NOW)

        with pytest.raises(DatasetLockError):
            lock.acquire("ds-1", now=NOW)


class TestLockInfoExpiry:
    def test_future_expiry_is_live(self):
        info = LockInfo("h", NOW.isoformat(), (NOW + timedelta(minutes=5)).isoformat())
        assert info.is_expired(NOW) is False

    def test_past_expiry_is_expired(self):
        info = LockInfo("h", NOW.isoformat(), (NOW - timedelta(seconds=1)).isoformat())
        assert info.is_expired(NOW) is True

    def test_expiry_is_inclusive(self):
        assert LockInfo("h", "", NOW.isoformat()).is_expired(NOW) is True

    def test_naive_timestamp_is_treated_as_utc(self):
        naive = (NOW + timedelta(hours=1)).replace(tzinfo=None).isoformat()
        assert LockInfo("h", "", naive).is_expired(NOW) is False

    @pytest.mark.parametrize("bad", ["", "nonsense", None])
    def test_unparseable_expiry_is_expired(self, bad):
        assert LockInfo("h", "", bad).is_expired(NOW) is True


class TestLazyClient:
    """Building the lock must not create an AWS client as a side effect.

    The Orchestrator constructs a DatasetLock in __init__. The existing tests patch
    boto3 per module, so an eagerly-created client here would escape those patches and
    reach for real S3 the moment an Orchestrator was built.
    """

    def test_no_client_is_created_until_used(self, monkeypatch: pytest.MonkeyPatch):
        created = []

        def spy(service, **kwargs):
            created.append(service)
            return ConditionalFakeS3()

        monkeypatch.setattr("lambda_source.dataset_lock.boto3.client", spy)

        lock = DatasetLock(s3_bucket="b")
        assert created == [], "constructing the lock must not create a client"

        lock.acquire("ds-1", now=NOW)
        assert created == ["s3"]

    def test_an_injected_client_is_used_directly(self, s3: ConditionalFakeS3):
        lock = DatasetLock(s3_bucket="b", s3_client=s3)
        lock.acquire("ds-1", now=NOW)

        assert "locks/ds-1.json" in s3.objects


class TestPipelineIntegration:
    """The lock must actually guard the apply path, not merely exist."""

    @staticmethod
    def _orchestrator(monkeypatch: pytest.MonkeyPatch, s3: ConditionalFakeS3):
        from unittest.mock import MagicMock

        from lambda_source.handler import Orchestrator
        from lambda_source.models import OptimiserConfig

        for module in (
            "lambda_source.handler",
            "lambda_source.snapshot",
            "lambda_source.reporter",
            "lambda_source.notifications",
            "lambda_source.qs_client",
        ):
            monkeypatch.setattr(f"{module}.boto3", MagicMock())

        config = OptimiserConfig(
            target_datasets=["ds-1"],
            execution_mode="apply",
            s3_bucket="my-bucket",
            s3_prefix="optimiser/",
            aws_account_id="123456789012",
            aws_region="us-east-1",
        )
        orchestrator = Orchestrator(config)
        # Point the lock at the fake S3 while leaving the rest of the wiring intact.
        orchestrator._dataset_lock = DatasetLock(
            s3_bucket="my-bucket", s3_prefix="optimiser/", s3_client=s3
        )
        return orchestrator

    def test_the_orchestrator_builds_a_lock_from_the_configured_bucket(
        self, monkeypatch: pytest.MonkeyPatch, s3: ConditionalFakeS3
    ):
        from unittest.mock import MagicMock

        from lambda_source.handler import Orchestrator
        from lambda_source.models import OptimiserConfig

        for module in (
            "lambda_source.handler",
            "lambda_source.snapshot",
            "lambda_source.reporter",
            "lambda_source.notifications",
            "lambda_source.qs_client",
        ):
            monkeypatch.setattr(f"{module}.boto3", MagicMock())

        orchestrator = Orchestrator(
            OptimiserConfig(
                target_datasets=["ds-1"],
                s3_bucket="my-bucket",
                s3_prefix="optimiser/",
                aws_account_id="123456789012",
                aws_region="us-east-1",
            )
        )

        assert orchestrator._dataset_lock.enabled
        assert (
            orchestrator._dataset_lock.generate_lock_key("ds-1")
            == "optimiser/locks/ds-1.json"
        )

    def test_a_held_lock_blocks_apply_and_never_calls_update_dataset(
        self, monkeypatch: pytest.MonkeyPatch, s3: ConditionalFakeS3
    ):
        """The property the lock exists for: the loser of the race does not write.

        The other run's lock is taken at wall time, not at the frozen NOW used
        elsewhere in this file. held() reads the real clock, so a lock stamped in the
        frozen past would look expired and be stolen.
        """
        from unittest.mock import MagicMock

        orchestrator = self._orchestrator(monkeypatch, s3)
        orchestrator._modifier = MagicMock()
        orchestrator._snapshot_manager = MagicMock()

        # Another run is mid-flight on this dataset, right now.
        DatasetLock(
            s3_bucket="my-bucket", s3_prefix="optimiser/", s3_client=s3
        ).acquire("ds-1", execution_id="other-run")

        with pytest.raises(DatasetLockedError):
            with orchestrator._dataset_lock.held("ds-1", execution_id="our-run"):
                orchestrator._snapshot_manager.capture_snapshot("ds-1")
                orchestrator._modifier.apply_modification("ds-1", {})

        orchestrator._snapshot_manager.capture_snapshot.assert_not_called()
        orchestrator._modifier.apply_modification.assert_not_called()

    def test_the_lock_is_free_again_for_the_next_run(
        self, monkeypatch: pytest.MonkeyPatch, s3: ConditionalFakeS3
    ):
        orchestrator = self._orchestrator(monkeypatch, s3)

        with orchestrator._dataset_lock.held("ds-1", execution_id="first"):
            pass

        with orchestrator._dataset_lock.held("ds-1", execution_id="second"):
            pass
