# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging tests for L4 K8s GPU pytest retry."""

import os


def test_staging_k8s_ok():
    assert True


def test_staging_k8s_crash_once():
    # Exit 137 avoids K8s exit-1 automatic retry while proving flushed pre-crash
    # JSONL preservation and Tier 1/2/3 skipping on tagged retry.
    if os.environ.get("BUILDKITE_RETRY_COUNT") == "0":
        os._exit(137)
