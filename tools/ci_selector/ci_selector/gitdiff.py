# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Changed-file extraction from git, rename/delete aware."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiffFile:
    # T (typechange) flows as an ordinary modification downstream: the
    # added-file rules key on "A" exactly.
    status: str  # A M D R C T
    path: str
    old_path: str | None = None


def diff_files(repo: Path, base: str, head: str | None = None) -> list[DiffFile]:
    # -z: NUL-separated raw paths, so non-ASCII names arrive unquoted
    # (C-quoted "caf\303\251.py" strings match no classification rule).
    args = ["git", "-C", str(repo), "diff", "--name-status", "-z", "-M", base]
    if head:
        args.append(head)
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    files: list[DiffFile] = []
    tokens = out.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue
        if status[0] in ("R", "C") and i + 2 < len(tokens):
            files.append(DiffFile(status[0], tokens[i + 2], old_path=tokens[i + 1]))
            i += 3
        elif i + 1 < len(tokens):
            files.append(DiffFile(status[0], tokens[i + 1]))
            i += 2
        else:
            break
    return files


def changed_paths(files: list[DiffFile]) -> list[str]:
    """Both sides of renames: the old path selects via its old closure
    (meaningful when analyzing at the diff's base), and the new path routes
    via the rename-pairing rule (select.DiffContext.renames) when a diff
    context is present, else the fail-open chain."""
    out: list[str] = []
    for f in files:
        out.append(f.path)
        if f.old_path:
            out.append(f.old_path)
    return out


def merge_base(repo: Path, a: str, b: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", a, b],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def resolve_diff_ref(repo: Path, ref: str) -> tuple[str, str | None]:
    """'A...B' -> (merge-base(A,B), B), matching CI's PR-diff semantics;
    'A..B' -> (A, B) snapshot diff; 'A' -> (A, None) = A vs working tree."""
    if "..." in ref:
        left, _, right = ref.partition("...")
        right = right or "HEAD"
        return merge_base(repo, left, right), right
    base, _, head = ref.partition("..")
    return base, head or None
