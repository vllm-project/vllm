# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is a row still describing the step it was recorded from?

A row is a photograph of a step at the commit it was recorded at, used to judge
a PR based somewhere else. If anything in between moved, the photograph is of
something else and the row must not authorise a drop. Two halves, both a diff
between those commits.

**Test side**: anything under what the step collects, which means its targets
plus every conftest above them and every test file they import. One forward
walk of the import graph picks all of that up.

**Step side**: the files that DEFINE the step. A row recorded under an env pin
that main later dropped describes a step that no longer exists, and the test
side cannot see it, because the tests did not move.

Cheap by construction: the surface is computed once and reused by every PR, so
each one costs a git diff and a set intersection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ci_selector.codemap.pipeline.invoked_tests import invoked_files

from ..gitdiff import changed_paths, diff_files

GLOBAL_TEST_CONFIG = ("pyproject.toml",)


@dataclass
class Surface:
    """What a step reads, and what defines it, at one commit."""

    tests: frozenset[str] = frozenset()
    definition: frozenset[str] = frozenset()

    def touched_by(self, changed: frozenset[str]) -> bool:
        return bool((self.tests | self.definition) & changed)


@dataclass
class Freshness:
    """Per-step surfaces, and the per-PR question they answer. Holds no repo
    handle and runs no git: the caller supplies the changed paths, having
    already resolved both ends."""

    commit: str
    surfaces: dict[str, Surface] = field(default_factory=dict)
    # Steps whose surface we could not compute. Absence of a surface is not
    # evidence of freshness, so these are always treated as stale.
    unknown: set[str] = field(default_factory=set)

    def stale(self, step_id: str, changed: frozenset[str]) -> bool:
        surface = self.surfaces.get(step_id)
        if surface is None:
            return True
        return surface.touched_by(changed)

    def stale_steps(self, step_ids, changed: frozenset[str]) -> set[str]:
        return {s for s in step_ids if self.stale(s, changed)}


def build(state, commit: str) -> Freshness:
    """Every step's surface, from a state already built at that commit.

    `commit` is required rather than set afterwards: it used to default to
    empty, one of the two callers forgot to fill it in, and the symptom shows
    up far from the mistake.
    """
    graph = state.full.graph
    out = Freshness(commit=commit, surfaces={})
    for pipeline in state.pipelines:
        for step in pipeline.steps:
            targets = pipeline.targets.get(step.step_id)
            if targets is None:
                out.unknown.add(step.step_id)
                continue
            collected = invoked_files(state.catalog, [targets])
            # Forward closure, so a change to a widely imported test helper
            # invalidates every row reading it and the conftest chain comes
            # along without being named.
            reachable = graph.forward_closure(collected) if collected else set()
            tests = {f for f in reachable if f.startswith("tests/")}
            tests |= collected
            tests.update(GLOBAL_TEST_CONFIG)

            # NOT the pipeline config, which says which steps exist and when
            # everything runs, never what one step does. Including it was
            # measured as disqualifying every step on every PR, for no gain: if
            # the config moves such that this step stops existing, it stops
            # being selected too.
            definition = {step.source_file}
            definition.update(targets.scripts_seen)

            out.surfaces[step.step_id] = Surface(
                tests=frozenset(tests), definition=frozenset(definition)
            )
    return out


def changed_between(repo: Path, base: str, head: str) -> frozenset[str]:
    """Every path that moved between two commits, renames counted on both
    sides. A step reading the old name is as stale as one reading the new."""
    files = diff_files(repo, base, head)
    paths = set(changed_paths(files))
    paths.update(f.old_path for f in files if f.old_path)
    return frozenset(paths)


def _commit_n_before(repo: Path, ref: str, n: int) -> str | None:
    """The commit `n` steps back on first-parent history from `ref`."""
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{ref}~{n}"],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def main() -> None:
    """What the staleness check costs, against how old the table is.

    Measuring from our one recorded commit answers the wrong question, since
    the benchmark's PRs are months away from it and production reads a table
    hours old. So walk BACKWARDS from each PR's own base: had the table been
    recorded n commits earlier, how much would the check disqualify? The answer
    is a curve, not a number.
    """
    import argparse
    import json
    import statistics

    from ..codemap.worktree import state_for

    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path)
    ap.add_argument("--crosscheck", type=Path, required=True)
    ap.add_argument("--at", default="69d4c3a06bf8d087455544db8cea570721eca415")
    ap.add_argument(
        "--distances", default="6,12,25,60,150,400", help="commits before each PR base"
    )
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    surfaces = build(state_for(args.repo, args.at), args.at)
    records = [r for r in json.load(args.crosscheck.open()) if r.get("selected_ids")]
    records = records[: args.limit]
    print(f"{len(surfaces.surfaces)} surfaces; {len(records)} PRs\n")

    hdr = f"{'commits back':>13} {'PRs':>5} {'median stale':>13} {'mean stale':>11}"
    print(hdr)
    print("-" * len(hdr))
    for n in [int(x) for x in args.distances.split(",")]:
        fractions = []
        for record in records:
            older = _commit_n_before(args.repo, record["base"], n)
            if older is None:
                continue
            changed = changed_between(args.repo, older, record["base"])
            sel = record["selected_ids"]
            known = [s for s in sel if s in surfaces.surfaces]
            if not known:
                continue
            stale = surfaces.stale_steps(known, changed)
            fractions.append(len(stale) / len(known))
        if fractions:
            print(
                f"{n:>13} {len(fractions):>5} "
                f"{100 * statistics.median(fractions):>12.1f}% "
                f"{100 * statistics.fmean(fractions):>10.1f}%"
            )


if __name__ == "__main__":
    main()
