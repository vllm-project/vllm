# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging tests for GPU Docker persistent-slot pytest retry."""

import os


def test_staging_gpu_ok():
    assert True


def test_staging_gpu_fail_once():
    retry = int(os.environ.get("BUILDKITE_RETRY_COUNT") or 0)
    assert retry >= 1, "staging: intentional GPU Docker failure on attempt 0"
