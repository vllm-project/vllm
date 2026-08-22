# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

import pytest

REPO = Path(os.environ.get("VLLM_REPO", Path(__file__).resolve().parents[3]))


def pytest_addoption(parser):
    parser.addoption(
        "--sync",
        action="store_true",
        help="Download ci-infra's generator before running, refreshing "
        "tests/ci_infra_snapshot/. Needs network. Selects no extra tests.",
    )


def pytest_sessionstart(session):
    """Download before collection, not as a test.

    Ordering matters and fixtures cannot give it: every ci-infra check reads
    the snapshot off disk, so the refresh has to land before any of them run,
    whatever order they are collected in. It asserts nothing; a change it
    brings in is reported by the ordinary offline tests afterwards.
    """
    if session.config.getoption("--sync"):
        import ci_infra

        ci_infra.sync()


@pytest.fixture(scope="session")
def vllm_repo() -> Path:
    """The real vLLM checkout. Named so it cannot be confused with the
    throwaway `tmp_repo` in tests/coverage/: same word, opposite meaning,
    and a test that got the wrong one would pass against nothing."""
    assert (REPO / ".buildkite" / "ci_config.yaml").is_file(), (
        f"{REPO} is not a vLLM checkout (set VLLM_REPO)"
    )
    return REPO


@pytest.fixture(scope="session")
def state(vllm_repo):
    from ci_selector.codemap.state import RepoState

    return RepoState.build(vllm_repo)


@pytest.fixture(scope="session")
def full(state):
    """The FullGraph, shared. Building one costs ~19s and parses every file in
    the checkout, so a per-module build is over half the suite's runtime.
    Nothing in select or preflight mutates it. A test that needs a graph built
    at a specific commit, or that times a cold build, must build its own."""
    return state.full
