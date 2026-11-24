# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging tests for L4 K8s GPU pytest retry."""

import os


def test_staging_k8s_ok():
    assert True


def test_staging_k8s_fail_once():
    retry = int(os.environ.get("BUILDKITE_RETRY_COUNT") or 0)
    assert retry >= 1, "staging: intentional L4 K8s failure on attempt 0"
