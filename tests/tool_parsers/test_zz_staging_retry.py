# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging tests for 1-job untagged vs tagged pytest retry."""

import os


def test_staging_ok():
    assert True


def test_staging_fail_once():
    retry = int(os.environ.get("BUILDKITE_RETRY_COUNT") or 0)
    assert retry >= 1, "staging: intentional failure on attempt 0"


def test_staging_fail_twice():
    retry = int(os.environ.get("BUILDKITE_RETRY_COUNT") or 0)
    assert retry >= 2, "staging: intentional failure on attempts 0 and 1"
