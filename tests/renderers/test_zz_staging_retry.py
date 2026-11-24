# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary staging tests for pytest fine-grained retry validation."""

import os

import pytest


@pytest.fixture
def teardown_fail_on_first_attempt():
    yield
    if os.environ.get("BUILDKITE_RETRY_COUNT") == "0":
        raise RuntimeError("staging: intentional teardown failure on attempt 0")


def test_staging_pass_before_crash():
    assert True


def test_staging_teardown_boom(teardown_fail_on_first_attempt):
    assert True


def test_staging_mid_crash():
    if os.environ.get("BUILDKITE_RETRY_COUNT") == "0":
        os._exit(137)


def test_staging_after_crash():
    assert True
