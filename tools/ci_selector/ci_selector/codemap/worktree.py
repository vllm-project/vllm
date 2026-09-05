# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached git worktrees + per-tree RepoState cache.

A leaf module (not part of the validate tooling): the harnesses and the CLI's
build-at-base mode both need analysis pinned to a commit, and selection code must
never depend on the harnesses.
"""

from __future__ import annotations

import atexit
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

# Overridable so a test run cannot write into the cache a real run reads.
WORKTREE_CACHE = Path(
    os.environ.get("CI_SELECTOR_WORKTREE_CACHE")
    or Path.home() / ".cache" / "vllm-ci-selector" / "worktrees"
)

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
        # Invalid AND claimed by someone else means another process is
        # mid-create or reading a tree we cannot verify; deleting it hands a
        # half-populated checkout to a live reader. Our own claim must not
        # block this, and cannot harm us: `_cached_by_tree` keys on the tree
        # oid, so a stale entry is never served for the oid we want now.
        if target.name in _claimed(exclude_self=True):
            raise RuntimeError(
                f"worktree {target.name} is claimed by another process but does "
                f"not validate at {oid}. Refusing to delete a tree in use. "
                f"If no run is live, remove {WORKTREE_CACHE}/.inuse.* and retry."
            )
        shutil.rmtree(target)
        # Prune before re-adding: git still holds the admin entry, and a stale
        # one makes the next `worktree add` fail with "missing but already
        # registered", burning one of only two attempts below.
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "prune"], capture_output=True
        )
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


def _marker() -> Path:
    return WORKTREE_CACHE / f".inuse.{os.getpid()}"


def _proc_started(pid: int) -> str:
    """A pid's start time, so a recycled pid cannot inherit a dead claim.

    Empty when it cannot be read, which the reader treats as "cannot rule this
    out" and keeps the claim. Over-claiming wastes disk; under-claiming deletes
    a tree somebody is reading.
    """
    proc = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _claim(target: Path) -> None:
    """Record every worktree this process still reads from.

    mtime cannot express this: a tree's mtime freezes at creation while the
    caller spends ~20s building state from it, so another worker's eviction
    picks it as the oldest and deletes files being read.

    All live trees, not just the newest. One PR claims its base tree and then
    its head tree, and a single-slot claim would release the base while its
    state is still in use.

    The claim file sits beside the worktree, never inside it, or
    `_worktree_valid` sees a dirty tree and prunes it. Names are kept distinct
    and a cache hit re-claims, so a repeat claim cannot spend a second slot and
    evict a live tree.
    """
    with contextlib.suppress(OSError):
        os.utime(target)
    name = target.name
    if name in _LIVE:
        _LIVE.remove(name)  # move to newest; a re-claim must not spend a slot
    _LIVE.append(name)
    while len(_LIVE) > MAX_CACHED_STATES + MAX_CACHED_HEAD_GRAPHS:
        _LIVE.popleft()
    _write_marker()


def _write_marker() -> None:
    """Publish this process's claims atomically.

    `write_text` truncates in place, so a concurrent `_claimed` could read a
    half-written file and miss a claim. Temp file plus rename instead, with the
    process start time on the first line so a recycled pid is detectable.
    Errors are not suppressed: a lost claim makes a live tree look evictable.
    """
    tmp = WORKTREE_CACHE / f".inuse.{os.getpid()}.tmp"
    body = "\n".join([_proc_started(os.getpid()), *_LIVE])
    tmp.write_text(body)
    os.replace(tmp, _marker())


def release_claims() -> None:
    """Drop this process's claims. Registered at exit and safe to call twice."""
    _LIVE.clear()
    with contextlib.suppress(OSError):
        _marker().unlink()


atexit.register(release_claims)


def _claimed(exclude_self: bool = False) -> set[str]:
    """Worktree names some live process says it is reading.

    Liveness is pid AND start time, since pids recycle and a bare pid check
    lets an unrelated process pin a tree forever. Any doubt keeps the claim.

    `exclude_self` is for the one caller asking "may I delete this". Eviction
    does not pass it; skipping our own live trees there is the point.
    """
    out: set[str] = set()
    mine = _marker()
    for marker in WORKTREE_CACHE.glob(".inuse.*"):
        if marker.suffix == ".tmp" or (exclude_self and marker == mine):
            continue
        try:
            pid = int(marker.suffix[1:])
            lines = marker.read_text().split("\n")
        except (OSError, ValueError):
            continue
        started, names = (lines[0].strip(), lines[1:]) if lines else ("", [])
        try:
            os.kill(pid, 0)  # liveness only; does not signal
        except ProcessLookupError:
            with contextlib.suppress(OSError):
                marker.unlink()
            continue
        except PermissionError:
            pass
        else:
            # Same pid, different process: the original died and the number
            # came round again, so its claims are stale.
            now = _proc_started(pid)
            if started and now and started != now:
                with contextlib.suppress(OSError):
                    marker.unlink()
                continue
        out.update(n for n in names if n)
    return out


def _evict_old_worktrees(repo: Path, keep: Path) -> None:
    """Trim the cache to MAX_WORKTREES, never touching a tree someone reads.

    `repo` is protected as well as `keep`, and not defensively: the head-closure
    rule passes a cached worktree as the repo, so the tree a caller is working
    from arrives here as an ordinary eviction candidate.

    What is counted against the cap and what is trimmed must be the same
    population. Mixing them once deleted every unpinned tree on every create,
    which stops the cache caching.
    """
    protect = {keep.resolve(), repo.resolve()}
    claimed = _claimed()
    evictable = [
        d
        for d in WORKTREE_CACHE.iterdir()
        if d.is_dir() and d.resolve() not in protect and d.name not in claimed
    ]
    evictable.sort(key=lambda d: d.stat().st_mtime)
    # Claimed trees are counted against the cap but never deleted, so the cache
    # can sit above MAX_WORKTREES while readers are live rather than pull a
    # tree out from under one. Trim only what is genuinely free.
    total = sum(1 for d in WORKTREE_CACHE.iterdir() if d.is_dir())
    for stale in evictable[: max(0, total - MAX_WORKTREES)]:
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
