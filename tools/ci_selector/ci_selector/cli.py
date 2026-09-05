# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ci-select: diff -> Buildkite job set, locally, in seconds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .codemap.classify import select
from .codemap.state import RepoState
from .gitdiff import changed_paths, diff_files, resolve_diff_ref
from .handwritten import (
    PR_PIPELINE,
)
from .report import render_json

# Every flag defined once, so `--repo X codemap` and `codemap --repo X` agree.
# SUPPRESS is load-bearing: with `parents=`, a subparser re-applies its parents'
# defaults over what was already parsed, so a plain default would quietly reset
# `--repo` to "." and analyze the wrong checkout while still exiting 0.
_DEFAULTS = {
    "repo": Path("."),
    "diff": None,
    "emit_keys": False,
    "pipeline": PR_PIPELINE,
    "no_base_worktree": False,
    "table": None,
}


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    p.add_argument("--repo", type=Path)
    p.add_argument(
        "--diff",
        help="a two-ended git range, BASE...HEAD (PR semantics) or BASE..HEAD",
    )
    p.add_argument(
        "--emit-keys",
        action="store_true",
        help="print the step-key list for CI instead of the full selection",
    )
    p.add_argument("--pipeline", help="which pipeline --emit-keys names steps for")
    p.add_argument(
        "--no-base-worktree",
        action="store_true",
        help="with --diff BASE..HEAD, analyze from the current checkout "
        "instead of a cached worktree at BASE (added-file routing and "
        "table scoping may then not fire)",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface, in one place so tests parse exactly what
    the binary parses."""
    common = _common()
    parser = argparse.ArgumentParser(
        prog="ci-select",
        parents=[common],
        description=(
            "Which vLLM CI jobs a diff needs. With no subcommand, both inputs "
            "decide: the code map and the coverage record."
        ),
    )
    parser.set_defaults(_mode="both")
    sub = parser.add_subparsers(dest="mode")
    codemap_p = sub.add_parser(
        "codemap",
        parents=[common],
        help="what the code alone says (import graph, registries, build DAG)",
    )
    codemap_p.set_defaults(_mode="codemap")
    parser.add_argument("--table", type=Path, help="coverage table to read")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, default in _DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)

    repo = args.repo.resolve()

    # Both commands need a two-ended range. The coverage half works from changed
    # FUNCTION names, which bare paths cannot give it. And a one-ended range is
    # worse than it looks: it quietly compares against the working tree, skips
    # the base worktree so added-file routing and table scoping never fire, and
    # cannot see untracked files at all. CI always has a range.
    if not args.diff:
        print(
            "error: --diff is required, as a two-ended range: "
            "--diff origin/main...HEAD",
            file=sys.stderr,
        )
        return 2
    base, head = resolve_diff_ref(repo, args.diff)
    if not base or head is None:
        print(
            f"error: --diff {args.diff!r} is not a two-ended range. Use "
            "BASE...HEAD (merge-base, what CI diffs) or BASE..HEAD.",
            file=sys.stderr,
        )
        return 2
    # Pin symbolic refs to commits in THIS repo: inside a detached base
    # worktree, "HEAD" and branch names resolve to the base and empty the diff.
    base = _rev_parse(repo, base)
    head = _rev_parse(repo, head)
    paths = changed_paths(diff_files(repo, base, head))
    if not paths:
        print("no changed files", file=sys.stderr)
        return 0

    # A ranged diff is analyzed at its BASE, because added-file routing and
    # table scoping assume state built there. The working tree is only right
    # when it happens to BE the base.
    if base and head and not args.no_base_worktree:
        from .codemap.worktree import state_for

        state = state_for(repo, base)
    else:
        if head is not None:
            _warn_if_stale_checkout(repo, head)
        state = RepoState.build(repo)
    pf = state.preflight
    for line in (*pf.run_all_reasons, *pf.warnings):
        print(line, file=sys.stderr)
    for why, n in pf.forced_by_reason.items():
        print(f"{why} [{n} steps]", file=sys.stderr)
    sel = select(state, paths, base=base, head=head)

    if args._mode != "codemap":
        from .coverage.source import fetch_table
        from .decide import decide

        table = fetch_table(args.table)
        if table.dead_interpreter:
            print(f"coverage: {table.dead_interpreter}", file=sys.stderr)
        d = decide(state, sel, repo, base, head, table=table)
        if d.coverage_note:
            print(f"coverage: {d.coverage_note}", file=sys.stderr)
        else:
            print(
                f"coverage: +{len(d.added_by_coverage)} added, "
                f"-{len(d.dropped_by_coverage)} dropped "
                f"({d.stale_steps} steps held by the freshness gate)",
                file=sys.stderr,
            )
        sel = _restrict(sel, d.steps)

    if args.emit_keys:
        from .codemap.step_keys import emit, render

        print(render(emit(state, sel, args.pipeline)))
        return 0
    print(render_json(sel))
    return 0


def _restrict(sel, steps: set[str]):
    """The selection narrowed to `steps`, with every parallel field kept in
    step. `selected`, `selected_rules` and `selected_paths` are read by position
    against each other, so they can only be filtered together."""
    import dataclasses

    added = {
        s: ["coverage: a row shows this step ran a changed file"]
        for s in steps - set(sel.selected)
    }
    return dataclasses.replace(
        sel,
        selected={**{k: v for k, v in sel.selected.items() if k in steps}, **added},
        selected_rules={
            **{k: v for k, v in sel.selected_rules.items() if k in steps},
            **{s: ["coverage"] for s in added},
        },
        selected_paths={
            **{k: v for k, v in sel.selected_paths.items() if k in steps},
            **{s: [None] for s in added},
        },
    )


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
