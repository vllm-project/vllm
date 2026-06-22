# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared pytest fixtures for Cohere integration tests.

Configuration is driven by environment variables set in run_tests.sh.
"""

import os

import pytest


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        pytest.skip(f"Environment variable {name} not set")
    assert val is not None
    return val


@pytest.fixture(scope="session")
def engines_dir() -> str:
    return _require_env("ENGINES_DIR")


@pytest.fixture(scope="session")
def vllm_workspace() -> str:
    return _require_env("VLLM_WORKSPACE")
