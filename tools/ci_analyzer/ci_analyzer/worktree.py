# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached git worktrees + per-tree AnalyzerState cache.

A leaf module (not part of the validate tooling): the harnesses and the CLI's
build-at-base mode both need analysis pinned to a commit, and selection code must
never depend on the harnesses.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .graph.build import FullGraph, build_full_graph
from .select import AnalyzerState

T = TypeVar("T")

WORKTREE_CACHE = Path.home() / ".cache" / "vllm-ci-analyzer" / "worktrees"

# States are ~100s of MB each; cache only what's needed at once. Two suffices:
# the tablediff tests alternate a base state with a merge/head state (both
# stay cached), while crosscheck and the CLI's build-at-base reuse one base.
MAX_CACHED_STATES = 2

# The added-file head-closure rule needs a graph at the diff HEAD; a bare
# FullGraph is the bulk of a state, so cap it like MAX_CACHED_STATES.
MAX_CACHED_HEAD_GRAPHS = 2

# Each cached worktree is a full checkout (~150MB); crosscheck --prs builds one
# per PR base. Cap the on-disk cache here, evicting the oldest past the bound.
MAX_WORKTREES = 8


def git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def worktree_at(repo: Path, sha: str) -> Path:
    """A validated worktree at sha. An existing cache dir is trusted only if
    its HEAD matches and git can read it; anything else is pruned and recreated."""
    oid = git_out(repo, "rev-parse", f"{sha}^{{commit}}")
    target = WORKTREE_CACHE / oid[:12]
    if target.exists():
        if _worktree_valid(target, oid):
            # Eviction is oldest-mtime-first, and reuse returns before it ever
            # runs, so a cached tree's mtime stayed frozen at creation: the one
            # being actively read was the first candidate for deletion by a
            # concurrent run. Touching it makes eviction pick an idle tree.
            with contextlib.suppress(OSError):
                os.utime(target)
            return target
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = None
    for _ in range(2):
        proc = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(target), oid],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            _evict_old_worktrees(repo, keep=target)
            return target
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "prune"],
            capture_output=True,
        )
    raise RuntimeError(f"worktree add failed for {sha}: {proc.stderr.strip()}")


def _evict_old_worktrees(repo: Path, keep: Path) -> None:
    dirs = [d for d in WORKTREE_CACHE.iterdir() if d.is_dir() and d != keep]
    dirs.sort(key=lambda d: d.stat().st_mtime)
    for stale in dirs[: max(0, len(dirs) - (MAX_WORKTREES - 1))]:
        shutil.rmtree(stale, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "prune"],
            capture_output=True,
        )


def _worktree_valid(target: Path, oid: str) -> bool:
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    # A cached worktree with modified/added files would be read as a tree that
    # is not the pinned commit: reject it, not just an unreadable one.
    return (
        head.returncode == 0
        and head.stdout.strip() == oid
        and status.returncode == 0
        and not status.stdout.strip()
    )


_STATE_CACHE: dict[str, AnalyzerState] = {}
_HEAD_GRAPH_CACHE: dict[str, FullGraph] = {}


def _cached_by_tree(
    cache: dict[str, T],
    cap: int,
    repo: Path,
    ref: str,
    builder: Callable[[Path], T],
) -> T:
    tree = git_out(repo, "rev-parse", f"{ref}^{{tree}}")
    if tree not in cache:
        while len(cache) >= cap:
            cache.pop(next(iter(cache)))
        cache[tree] = builder(worktree_at(repo, ref))
    return cache[tree]


def state_for(repo: Path, base: str) -> AnalyzerState:
    return _cached_by_tree(
        _STATE_CACHE, MAX_CACHED_STATES, repo, base, AnalyzerState.build
    )


def full_graph_for(repo: Path, ref: str) -> FullGraph:
    """A bare FullGraph at ref, worktree-backed and cached per tree oid.
    Deliberately NOT an AnalyzerState (~17s): imports + wall parsers +
    finalize_lazy_edges (~13s) is all the head-side added-file routing needs.
    The lazy/registry importers only materialize there. Steps, keys, and
    preflight stay base-derived."""
    return _cached_by_tree(
        _HEAD_GRAPH_CACHE, MAX_CACHED_HEAD_GRAPHS, repo, ref, build_full_graph
    )


def clear_state_cache() -> None:
    _STATE_CACHE.clear()
    _HEAD_GRAPH_CACHE.clear()
