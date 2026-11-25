# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throwaway staging test for 1-job tagged retry and flag-off control."""

import os

import pytest


def test_staging_ok() -> None:
    assert True


def test_staging_fail_once() -> None:
    if not os.environ.get("RUNTIME_SHARD_PLAN"):
        return
    if int(os.environ.get("BUILDKITE_RETRY_COUNT") or 0) < 1:
        pytest.fail("synthetic staging failure on Attempt 0 (1-job retry)")
