# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worktree cache validation + recovery on a throwaway git repo."""

import shutil
import subprocess

import pytest
from ci_analyzer import worktree


def _g(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _g(repo, "init", "-q")
    _g(repo, "config", "user.email", "t@t")
    _g(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    _g(repo, "add", ".")
    _g(repo, "commit", "-qm", "one")
    (repo / "a.txt").write_text("2")
    _g(repo, "add", ".")
    _g(repo, "commit", "-qm", "two")
    monkeypatch.setattr(worktree, "WORKTREE_CACHE", tmp_path / "cache")
    worktree.clear_state_cache()
    return repo


def test_worktree_created_and_reused(tmp_repo):
    wt1 = worktree.worktree_at(tmp_repo, "HEAD")
    wt2 = worktree.worktree_at(tmp_repo, "HEAD")
    assert wt1 == wt2
    assert (wt1 / "a.txt").read_text() == "2"


def test_distinct_commits_get_distinct_worktrees(tmp_repo):
    wt_head = worktree.worktree_at(tmp_repo, "HEAD")
    wt_parent = worktree.worktree_at(tmp_repo, "HEAD^")
    assert wt_head != wt_parent
    assert (wt_head / "a.txt").read_text() == "2"
    assert (wt_parent / "a.txt").read_text() == "1"


def test_worktree_recreated_after_rm_rf(tmp_repo):
    """Deleting the cache dir leaves a registered-but-missing worktree;
    the next call must prune and recreate, not die on 'already registered'."""
    wt = worktree.worktree_at(tmp_repo, "HEAD")
    shutil.rmtree(wt)
    wt2 = worktree.worktree_at(tmp_repo, "HEAD")
    assert (wt2 / "a.txt").read_text() == "2"


def test_worktree_at_wrong_commit_recreated(tmp_repo):
    """A cache dir checked out elsewhere (interrupted/corrupted state) is
    detected by HEAD validation and rebuilt at the requested sha."""
    wt = worktree.worktree_at(tmp_repo, "HEAD")
    subprocess.run(
        ["git", "checkout", "-q", "HEAD^"],
        cwd=wt,
        check=True,
        capture_output=True,
    )
    wt2 = worktree.worktree_at(tmp_repo, "HEAD")
    assert (wt2 / "a.txt").read_text() == "2"


def test_dirty_worktree_is_invalid(tmp_repo):
    """A cached worktree with uncommitted edits reads a modified tree, not the
    pinned commit: _worktree_valid must reject it (dirty, not just unreadable)."""
    wt = worktree.worktree_at(tmp_repo, "HEAD")
    oid = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert worktree._worktree_valid(wt, oid)
    (wt / "a.txt").write_text("uncommitted edit")
    assert not worktree._worktree_valid(wt, oid)
