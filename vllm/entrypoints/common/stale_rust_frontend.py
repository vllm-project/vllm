# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detect a packaged `vllm-rs` binary that lacks the checkout's `rust/` sources.

Precompiled installs (`VLLM_USE_PRECOMPILED=1` or
`VLLM_USE_PRECOMPILED_RUST=1`) take `vllm-rs` from the wheel of an upstream
main commit, so `rust/` changes on a local branch are never built, and an
editable install keeps the old binary across `git pull`. The check warns only
on clear evidence and stays silent whenever the evidence is missing.
"""

import os
import subprocess
from dataclasses import dataclass

import regex as re

import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)

# The setuptools-scm local version: `+g<commit>` for a clean tree, followed by
# `.dYYYYMMDD` when built from a dirty one.
_VERSION_COMMIT_RE = re.compile(r"\+g([0-9a-f]{7,40})(\.d\d{8})?$")
_SUBPROCESS_TIMEOUT_S = 10
_MAX_LISTED_PATHS = 5


@dataclass
class UnbuiltRustChanges:
    checkout: str
    binary_commit: str
    changed_paths: list[str]


def _run(args: list[str], env: dict[str, str] | None = None) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=True,
            env=env,
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def find_unbuilt_rust_changes(
    binary_path: str, package_dir: str
) -> UnbuiltRustChanges | None:
    """Return the `rust/` changes the packaged `binary_path` was not built with.

    Returns None unless `package_dir` is the `vllm/` directory of a git
    checkout, `binary_path` is the `vllm-rs` packaged next to it (possibly a
    symlink), and the binary reports a clean commit that the checkout has.
    """
    package_dir = os.path.realpath(package_dir)
    # A binary chosen through VLLM_RUST_FRONTEND_PATH has its own provenance.
    if (
        os.path.basename(binary_path) != "vllm-rs"
        or os.path.realpath(os.path.dirname(binary_path)) != package_dir
    ):
        return None

    # Stay local: no lazy fetches in partial clones, no index refresh writes.
    git_env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0"}
    checkout = _run(["git", "-C", package_dir, "rev-parse", "--show-toplevel"], git_env)
    if checkout is None:
        return None
    # A non-editable install inside a checkout (e.g. a venv in the repo) does
    # not run the checkout's sources.
    if os.path.realpath(os.path.join(checkout, "vllm")) != package_dir:
        return None

    version = _run([binary_path, "--version"])
    match = _VERSION_COMMIT_RE.search(version.split()[-1]) if version else None
    if match is None or match.group(2) is not None:
        return None
    reported = match.group(1)

    git = ["git", "-C", checkout]
    binary_commit = _run(
        [*git, "rev-parse", "--verify", "--quiet", f"{reported}^{{commit}}"], git_env
    )
    # `rev-parse` prefers a ref with the same name over an abbreviated hash.
    if binary_commit is None or not binary_commit.startswith(reported):
        return None
    changed = _run([*git, "diff", "--name-only", binary_commit, "--", "rust/"], git_env)
    if changed is None:
        return None
    # Cargo also builds untracked files it discovers, such as a new `build.rs`.
    untracked = _run(
        [*git, "ls-files", "--others", "--exclude-standard", "--", "rust/"], git_env
    )
    paths = changed.splitlines() + (untracked or "").splitlines()
    if not paths:
        return None
    return UnbuiltRustChanges(checkout, binary_commit, paths)


def warn_if_rust_frontend_stale(binary_path: str) -> None:
    """Warn when the packaged `vllm-rs` was built without this checkout's
    `rust/` changes."""
    package_dir = os.path.dirname(os.path.abspath(vllm.__file__))
    unbuilt = find_unbuilt_rust_changes(binary_path, package_dir)
    if unbuilt is None:
        return

    paths = unbuilt.changed_paths
    listed = ", ".join(paths[:_MAX_LISTED_PATHS])
    if len(paths) > _MAX_LISTED_PATHS:
        listed += ", ..."
    files = "1 file" if len(paths) == 1 else f"{len(paths)} files"
    logger.warning(
        "The Rust frontend binary %s was built from commit %s, but rust/ in %s "
        "differs from that commit in %s (%s). The binary was not rebuilt after "
        "local edits or a checkout update, or it came from an install with "
        "VLLM_USE_PRECOMPILED=1 or VLLM_USE_PRECOMPILED_RUST=1 on a branch that "
        "changes rust/. It may not match this checkout's Python code, so the "
        "server can fail at startup or on requests, or return wrong results. "
        "Rebuild it with `./tools/build_rust.sh`.",
        binary_path,
        unbuilt.binary_commit[:10],
        unbuilt.checkout,
        files,
        listed,
    )
