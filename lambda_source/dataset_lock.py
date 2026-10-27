# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Per-dataset advisory lock, so two optimiser runs cannot race on one dataset.

``UpdateDataSet`` is a full-definition replace with no conditional check, no ETag, and
no ``LastUpdatedTime`` precondition. The pipeline does describe, modify, write. Two
overlapping runs against the same dataset are therefore a lost update: both read the
original definition, both write, and the second write silently discards the first
run's column removals. Worse for a tool that advertises reversibility — two snapshots
exist and neither is unambiguously the pre-change state.

The Lambda used to carry ``ReservedConcurrentExecutions: 1``, which serialised
everything. It was removed in ``15b3d39`` because the async job pattern self-invokes
the same function, and a reserved concurrency of one deadlocks that. Necessary change,
but it deleted the guarantee without replacing it. This is the replacement.

It lives in the pipeline rather than in the tool dispatcher deliberately, so the
guarantee survives any future change to how the optimiser is invoked.

Mechanism: a lock object in S3, created with ``IfNoneMatch='*'`` so creation fails if
one already exists. An expired lock may be stolen, but only via ``IfMatch=<etag>`` so
two simultaneous stealers cannot both win. Release deletes the object, and the embedded
expiry acts as a dead-man's switch if a run dies mid-flight.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Long enough to cover a slow UpdateDataSet plus ingestion, short enough that a
# crashed run does not block a dataset for a whole working day.
DEFAULT_LOCK_TTL_MINUTES = 30

# S3 returns 412 for a failed IfNoneMatch and for a failed IfMatch alike.
_PRECONDITION_CODES = frozenset({"PreconditionFailed", "412"})
_MISSING_CODES = frozenset({"NoSuchKey", "404"})


class DatasetLockError(Exception):
    """Raised when the lock cannot be read or written."""

    def __init__(self, message: str, dataset_id: str | None = None):
        super().__init__(message)
        self.dataset_id = dataset_id


class DatasetLockedError(DatasetLockError):
    """Raised when another live run already holds the lock for a dataset."""

    def __init__(
        self,
        message: str,
        dataset_id: str | None = None,
        holder: str | None = None,
        expires_at: str | None = None,
    ):
        super().__init__(message, dataset_id)
        self.holder = holder
        self.expires_at = expires_at


def _holder_identity() -> str:
    """Identify this process well enough for a human to trace a stale lock."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


@dataclass
class LockInfo:
    """Contents of a lock object."""

    holder: str
    acquired_at: str
    expires_at: str
    execution_id: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        """Whether this lock may be stolen.

        An unparseable expiry counts as expired: a lock nobody can reason about must
        not be able to block a dataset permanently.
        """
        current = now or datetime.now(timezone.utc)
        try:
            deadline = datetime.fromisoformat(self.expires_at)
        except (TypeError, ValueError):
            return True
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return current >= deadline

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "holder": self.holder,
                "acquired_at": self.acquired_at,
                "expires_at": self.expires_at,
                "execution_id": self.execution_id,
            },
            indent=2,
        ).encode("utf-8")


class DatasetLock:
    """Acquires and releases per-dataset locks in S3."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "",
        s3_client: Any | None = None,
        ttl_minutes: int = DEFAULT_LOCK_TTL_MINUTES,
    ):
        self._bucket = s3_bucket
        prefix = s3_prefix or ""
        self._prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
        self._ttl_minutes = ttl_minutes

        # Constructed lazily. Building an Orchestrator must not create an AWS client
        # as a side effect — the existing tests patch boto3 per module, and an eager
        # client here would escape those patches and reach for real S3.
        self._injected_client = s3_client
        self._client: Any | None = s3_client

    @property
    def _s3_client(self) -> Any:
        if self._client is None:
            self._client = boto3.client("s3")
        return self._client

    @property
    def enabled(self) -> bool:
        """False when no bucket is configured, in which case locking is skipped."""
        return bool(self._bucket)

    def generate_lock_key(self, dataset_id: str) -> str:
        """S3 key holding the lock for a dataset."""
        return f"{self._prefix}locks/{dataset_id}.json"

    def _read(self, key: str) -> tuple[LockInfo, str] | None:
        """Read the current lock and its ETag, or None if absent."""
        try:
            response = self._s3_client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _MISSING_CODES:
                return None
            raise DatasetLockError(
                f"Cannot read dataset lock at s3://{self._bucket}/{key}: {exc}"
            ) from exc

        etag = response.get("ETag", "")
        try:
            body = json.loads(response["Body"].read().decode("utf-8"))
            return (
                LockInfo(
                    holder=body.get("holder", "unknown"),
                    acquired_at=body.get("acquired_at", ""),
                    expires_at=body.get("expires_at", ""),
                    execution_id=body.get("execution_id"),
                ),
                etag,
            )
        except (ValueError, KeyError):
            # A corrupt lock is treated as stale rather than as an eternal block. The
            # steal still goes through IfMatch, so it stays race-safe.
            logger.warning(
                "Dataset lock is unreadable, treating as stale",
                extra={"lock_key": key},
            )
            return LockInfo("unknown", "", "", None), etag

    def _put(self, key: str, info: LockInfo, **conditions: str) -> None:
        self._s3_client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=info.to_json(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
            **conditions,
        )

    def acquire(
        self,
        dataset_id: str,
        execution_id: str | None = None,
        now: datetime | None = None,
    ) -> LockInfo:
        """Take the lock for a dataset.

        Args:
            dataset_id: Dataset to lock.
            execution_id: Recorded in the lock so a human can trace what holds it.
            now: Injected clock, for tests.

        Returns:
            The :class:`LockInfo` written.

        Raises:
            DatasetLockedError: Another live run holds the lock.
            DatasetLockError: The lock could not be read or written.
        """
        current = now or datetime.now(timezone.utc)
        info = LockInfo(
            holder=_holder_identity(),
            acquired_at=current.isoformat(),
            expires_at=(current + timedelta(minutes=self._ttl_minutes)).isoformat(),
            execution_id=execution_id,
        )
        key = self.generate_lock_key(dataset_id)

        try:
            self._put(key, info, IfNoneMatch="*")
            logger.info(
                "Dataset lock acquired",
                extra={"dataset_id": dataset_id, "holder": info.holder},
            )
            return info
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _PRECONDITION_CODES:
                raise DatasetLockError(
                    f"Cannot acquire lock for dataset {dataset_id}: {exc}",
                    dataset_id=dataset_id,
                ) from exc

        existing = self._read(key)
        if existing is None:
            # Released between the attempt and the read. One retry; a second failure
            # means real contention.
            try:
                self._put(key, info, IfNoneMatch="*")
                return info
            except ClientError as exc:
                raise DatasetLockedError(
                    f"Dataset {dataset_id} is contended; another run took the lock.",
                    dataset_id=dataset_id,
                ) from exc

        held, etag = existing
        if not held.is_expired(current):
            raise DatasetLockedError(
                f"Dataset {dataset_id} is already being optimised by "
                f"{held.holder!r}, which holds the lock until {held.expires_at}.",
                dataset_id=dataset_id,
                holder=held.holder,
                expires_at=held.expires_at,
            )

        logger.warning(
            "Stealing an expired dataset lock",
            extra={
                "dataset_id": dataset_id,
                "previous_holder": held.holder,
                "expired_at": held.expires_at,
            },
        )

        try:
            self._put(key, info, IfMatch=etag)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _PRECONDITION_CODES:
                raise DatasetLockedError(
                    f"Dataset {dataset_id} was claimed by another run while its "
                    f"expired lock was being replaced.",
                    dataset_id=dataset_id,
                ) from exc
            raise DatasetLockError(
                f"Cannot replace expired lock for dataset {dataset_id}: {exc}",
                dataset_id=dataset_id,
            ) from exc

        return info

    def release(self, dataset_id: str) -> None:
        """Release the lock. Never raises.

        Called from a ``finally`` block, where an exception would mask whatever
        actually went wrong in the optimisation.
        """
        key = self.generate_lock_key(dataset_id)
        try:
            self._s3_client.delete_object(Bucket=self._bucket, Key=key)
            logger.info("Dataset lock released", extra={"dataset_id": dataset_id})
        except ClientError as exc:
            logger.error(
                "Cannot release dataset lock; it will expire on its own",
                extra={
                    "dataset_id": dataset_id,
                    "lock_key": key,
                    "error": str(exc),
                },
            )

    @contextmanager
    def held(
        self, dataset_id: str, execution_id: str | None = None
    ) -> Iterator[LockInfo]:
        """Hold the lock for the duration of a block.

        Release happens in a ``finally``, so every exit path is covered — including
        the several early returns in the apply sequence.
        """
        info = self.acquire(dataset_id, execution_id=execution_id)
        try:
            yield info
        finally:
            self.release(dataset_id)
