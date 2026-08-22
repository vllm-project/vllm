# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached git worktrees + per-tree RepoState cache.

A leaf module (not part of the validate tooling): the harnesses and the CLI's
build-at-base mode both need analysis pinned to a commit, and selection code must
never depend on the harnesses.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from .graph.build import FullGraph, build_full_graph
from .state import RepoState

T = TypeVar("T")

WORKTREE_CACHE = Path.home() / ".cache" / "vllm-ci-selector" / "worktrees"

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

# Worktrees this process still reads from; see _claim.
_LIVE: deque[str] = deque()


def git_out(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@contextlib.contextmanager
def _cache_lock():
    """Serialise creation and eviction across processes.

    Two workers racing here corrupt each other three ways: both `git worktree
    add` the same oid, one validates a directory the other is still filling,
    and `git worktree prune` strips a worktree another just registered. All
    three leave a half-populated tree that reads as missing files, which fails
    open to run-everything. The lock is released before the caller builds
    state from the tree, so the expensive part stays parallel.
    """
    WORKTREE_CACHE.mkdir(parents=True, exist_ok=True)
    fd = os.open(WORKTREE_CACHE / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def worktree_at(repo: Path, sha: str) -> Path:
    """A validated worktree at sha. An existing cache dir is trusted only if
    its HEAD matches and git can read it; anything else is pruned and recreated."""
    oid = git_out(repo, "rev-parse", f"{sha}^{{commit}}")
    with _cache_lock():
        return _worktree_at_locked(repo, oid)


def _worktree_at_locked(repo: Path, oid: str) -> Path:
    target = WORKTREE_CACHE / oid[:12]
    if target.exists():
        if _worktree_valid(target, oid):
            _claim(target)
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
            _claim(target)
            _evict_old_worktrees(repo, keep=target)
            return target
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "prune"],
            capture_output=True,
        )
    raise RuntimeError(f"worktree add failed for {oid}: {proc.stderr.strip()}")


def _claim(target: Path) -> None:
    """Record every worktree this process still reads from.

    mtime alone cannot express this. A tree's mtime freezes at creation while
    the caller spends ~20s building state from it, so a concurrent worker's
    eviction picks it as the oldest and deletes the files being read. That
    surfaces as missing files, which fails open to run-everything.

    All live trees, not just the newest: one PR claims its base tree and then
    its head tree, because routing an added file needs a graph at head, and a
    single-slot claim would release the base while its state is still in use.
    The bound matches what the in-memory caches can hold.

    The claim lives beside the worktree, never inside it: a file within would
    make `_worktree_valid` see a dirty tree and prune it.
    """
    with contextlib.suppress(OSError):
        os.utime(target)
    _LIVE.append(target.name)
    while len(_LIVE) > MAX_CACHED_STATES + MAX_CACHED_HEAD_GRAPHS:
        _LIVE.popleft()
    with contextlib.suppress(OSError):
        (WORKTREE_CACHE / f".inuse.{os.getpid()}").write_text("\n".join(_LIVE))


def _claimed() -> set[str]:
    """Worktree names some live process says it is reading."""
    out: set[str] = set()
    for marker in WORKTREE_CACHE.glob(".inuse.*"):
        try:
            pid = int(marker.suffix[1:])
            names = marker.read_text().split()
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)  # liveness only; does not signal
        except ProcessLookupError:
            with contextlib.suppress(OSError):
                marker.unlink()
            continue
        except PermissionError:
            pass
        out.update(names)
    return out


def _evict_old_worktrees(repo: Path, keep: Path) -> None:
    claimed = _claimed()
    others = [d for d in WORKTREE_CACHE.iterdir() if d.is_dir() and d != keep]
    dirs = [d for d in others if d.name not in claimed]
    dirs.sort(key=lambda d: d.stat().st_mtime)
    # Count claimed trees against the cap but never delete them, so the cache
    # can sit above MAX_WORKTREES while readers are live rather than pull a
    # tree out from under one. Bounded by the number of concurrent workers.
    for stale in dirs[: max(0, len(others) + 1 - MAX_WORKTREES)]:
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


_STATE_CACHE: dict[str, RepoState] = {}
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


def state_for(repo: Path, base: str) -> RepoState:
    return _cached_by_tree(_STATE_CACHE, MAX_CACHED_STATES, repo, base, RepoState.build)


def full_graph_for(repo: Path, ref: str) -> FullGraph:
    """A bare FullGraph at ref, worktree-backed and cached per tree oid.
    Deliberately NOT a RepoState (~17s): imports + edge parsers +
    finalize_lazy_edges (~13s) is all the head-side added-file routing needs.
    The lazy/registry importers only materialize there. Steps, keys, and
    preflight stay base-derived."""
    return _cached_by_tree(
        _HEAD_GRAPH_CACHE, MAX_CACHED_HEAD_GRAPHS, repo, ref, build_full_graph
    )


def clear_state_cache() -> None:
    _STATE_CACHE.clear()
    _HEAD_GRAPH_CACHE.clear()
