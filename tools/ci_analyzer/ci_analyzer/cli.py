# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ci-select: diff -> Buildkite job set, locally, in seconds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gitdiff import changed_paths, diff_files, resolve_diff_ref
from .report import render_json
from .select import AnalyzerState, select


def normalize_paths(paths: list[str], repo: Path) -> list[str]:
    """--files hygiene: './'-prefixed and repo-absolute paths would silently
    match no rule and fail open to run-all."""
    out = []
    for p in paths:
        raw = p.replace("\\", "/").removeprefix("./")
        path = Path(raw)
        if path.is_absolute():
            try:
                raw = path.resolve().relative_to(repo).as_posix()
            except ValueError:
                print(
                    f"warning: {p} is outside {repo}; it will fail open",
                    file=sys.stderr,
                )
        out.append(raw)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ci-select",
        description="Deterministic test-job selection for a vLLM diff.",
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--diff",
        help="BASE or BASE..HEAD git range (default: working tree vs HEAD)",
    )
    parser.add_argument("--files", nargs="*", help="explicit changed paths")
    parser.add_argument(
        "--no-base-worktree",
        action="store_true",
        help="with --diff BASE..HEAD, analyze from the current checkout "
        "instead of a cached worktree at BASE (added-file routing and "
        "table scoping may then not fire)",
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    base = head = None
    if args.files:
        paths = normalize_paths(args.files, repo)
    else:
        ref = args.diff or "HEAD"
        base, head = resolve_diff_ref(repo, ref)
        # Pin symbolic refs to oids in THIS repo: in a detached base worktree
        # "HEAD"/branch names resolve to the base commit and empty the diff.
        base = _rev_parse(repo, base)
        if head is not None:
            head = _rev_parse(repo, head)
        paths = changed_paths(diff_files(repo, base, head))
    if not paths:
        print("no changed files", file=sys.stderr)
        return 0

    # A ranged diff is analyzed at its BASE (the golden contract: added-file
    # routing and table scoping assume base-built state); the working tree is
    # only correct when it IS the base of the comparison.
    if base and head and not args.no_base_worktree:
        from .worktree import state_for

        state = state_for(repo, base)
    else:
        if head is not None:
            _warn_if_stale_checkout(repo, head)
        state = AnalyzerState.build(repo)
    pf = state.preflight
    for line in (*pf.run_all_reasons, *pf.warnings):
        print(line, file=sys.stderr)
    sel = select(state, paths, base=base, head=head)
    print(render_json(sel))
    return 0


def _rev_parse(repo: Path, ref: str) -> str:
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ref


def _warn_if_stale_checkout(repo: Path, head: str) -> None:
    import subprocess

    revs = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD", head],
        capture_output=True,
        text=True,
    )
    lines = revs.stdout.split()
    if revs.returncode == 0 and len(lines) == 2 and lines[0] != lines[1]:
        print(
            f"warning: working tree is not at {head}; graph and job data "
            "come from the checkout (mixed-snapshot analysis)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
