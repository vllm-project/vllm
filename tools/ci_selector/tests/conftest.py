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


@pytest.fixture(autouse=True, scope="session")
def _isolate_worktree_cache(tmp_path_factory):
    """Keep the suite out of the real worktree cache.

    Autouse and session-scoped because the leak is indirect and nobody
    remembers to opt in: a test can stub `state_for` and still reach
    `worktree_at` through `decide()`, which is how a pytest temp repo ended up
    registered in the live cache, pinned by a dead process's claim and
    invisible until someone listed the directory.

    Session-scoped so the trees built here are still shared between tests, and
    torn down with the session rather than left for the next run to trip over.
    """
    import ci_selector.codemap.worktree as wt

    original = wt.WORKTREE_CACHE
    wt.WORKTREE_CACHE = tmp_path_factory.mktemp("worktree-cache")
    try:
        yield wt.WORKTREE_CACHE
    finally:
        wt.release_claims()
        wt.WORKTREE_CACHE = original


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


@pytest.fixture
def declared_deps_on(monkeypatch):
    """Let the hand-written declared lists pick steps again, for tests of
    behaviour that only exists then: declarer-union rules, declared-deps
    routes, and the few reaches the derived default gives up."""
    monkeypatch.setenv("CI_SELECTOR_DECLARED_DEPS", "on")
