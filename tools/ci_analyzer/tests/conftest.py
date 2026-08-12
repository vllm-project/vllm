# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

import pytest

REPO = Path(os.environ.get("VLLM_REPO", Path(__file__).resolve().parents[3]))


@pytest.fixture(scope="session")
def repo() -> Path:
    assert (REPO / ".buildkite" / "ci_config.yaml").is_file(), (
        f"{REPO} is not a vLLM checkout (set VLLM_REPO)"
    )
    return REPO


@pytest.fixture(scope="session")
def state(repo):
    from ci_analyzer.select import AnalyzerState

    return AnalyzerState.build(repo)


@pytest.fixture(scope="session")
def full(state):
    """The FullGraph, shared. Building one costs ~19s and parses every file in
    the checkout, so a per-module build is over half the suite's runtime.
    Nothing in select or preflight mutates it. A test that needs a graph built
    at a specific commit, or that times a cold build, must build its own."""
    return state.full
