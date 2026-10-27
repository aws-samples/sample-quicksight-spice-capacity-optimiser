# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Shared pytest configuration.

The important thing here is the network guard. ``tests/integration`` used to hang
indefinitely rather than fail: a test reached real AWS through an unmocked client and
sat in retry backoff, so the whole 9-test suite was unusable and nobody could tell
whether it was slow or stuck.

A missing mock should be a fast, loud failure that names the call it tried to make.
"""

from __future__ import annotations

import os

import pytest


class LiveAWSCallAttempted(AssertionError):
    """Raised when a test tries to reach AWS for real.

    An AssertionError rather than a plain Exception so it reads as a test defect,
    which is what it is: something in the call path is missing a mock.
    """


@pytest.fixture(scope="session", autouse=True)
def _dummy_aws_environment() -> None:
    """Give botocore a complete, obviously-fake environment.

    Client construction resolves region and credentials eagerly. Without these, tests
    fail differently depending on whether the developer happens to have real
    credentials configured — and on a machine that does have them, an unmocked call
    would reach a real account. Neither is acceptable in a unit suite.
    """
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    # Stops botocore reaching for an IMDS role on the host. AWS_PROFILE is left alone
    # deliberately: setting it to "" makes botocore look for a profile named "" and
    # fail during client construction, which is a confusing way to break every test
    # that builds a client.
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.pop("AWS_PROFILE", None)


@pytest.fixture(autouse=True)
def _block_live_aws_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if any test attempts a real AWS request.

    ``URLLib3Session.send`` is the single chokepoint every boto3 client goes through,
    so patching it catches an unmocked call wherever it originates — including inside
    a retry loop, which is where the hang used to happen.

    Client *construction* is left alone deliberately. Building a client makes no
    network call, and plenty of tests legitimately construct one and then patch the
    method they exercise.
    """
    import botocore.httpsession

    def _refuse(self, request):  # type: ignore[no-untyped-def]
        raise LiveAWSCallAttempted(
            "A test attempted a live AWS call: "
            f"{getattr(request, 'method', '?')} {getattr(request, 'url', '?')}\n"
            "Something in this call path is missing a mock. Patch the boto3 client, "
            "or pass a stub via the injection seam (QSClient(client=...), "
            "ReportGenerator(s3_client=...), _list_spice_datasets(qs_client=...))."
        )

    monkeypatch.setattr(botocore.httpsession.URLLib3Session, "send", _refuse)
