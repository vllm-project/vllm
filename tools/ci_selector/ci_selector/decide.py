# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combine the two inputs into one job list.

Per changed file F and step S:

    a USABLE row shows S ran F     -> select. The map gets no vote.
    a USABLE row shows S ran none  -> drop, if every gate agrees.
    no usable row                  -> the map decides.

Usable, not merely present. A row that is stale, thin, digest-failed, from a job
that did not pass, or recorded by another Python minor is not evidence of
absence, so it takes the third line. Expect that line to be the common case.

Those three describe authority, not order. The map proposes the candidate set
first and the record adjusts it both ways. Evidence cannot originate a set,
being silent about new code and untraced trees, and silence is not "not needed".

Per FILE, not per diff. The recorder watches some trees and is blind to others,
so a diff touching both gets the record's answer for one and the map's for the
other.

Selecting takes one observation and carries no gate. Dropping needs the
recording to still describe the step it came from, so it carries the freshness
gate and every health check in `table.look_up`.

WHEN ANYTHING GOES WRONG, THE MAP'S SELECTION STANDS UNCHANGED -- and note the
shape of that. It is NOT "carry on with an empty stale set": an empty stale set
means nothing is disqualified, so every step stays droppable, and the failure
would land on the one side that can lose a test. The only safe degradation is
to skip dropping entirely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .coverage import freshness
from .coverage.phase import DEFAULT_MODE, PhaseMode, mode_from_env
from .coverage.rules import RowKeys, newest_commit, read_pr, unknown_names
from .coverage.source import fetch_table
from .coverage.table import Table

#: Set to re-enable the freshness gate, which is off by default.
FRESHNESS_ENV = "CI_SELECTOR_FRESHNESS"


@dataclass
class Decision:
    """What each input contributed, kept apart so the answer can be explained."""

    steps: set[str] = field(default_factory=set)
    from_map: set[str] = field(default_factory=set)
    added_by_coverage: set[str] = field(default_factory=set)
    dropped_by_coverage: set[str] = field(default_factory=set)
    # Why the record could not help, when it could not.
    coverage_note: str = ""
    # Steps whose row no longer describes them, so it cannot authorise a drop.
    stale_steps: int = 0
    reasons: dict = field(default_factory=dict)

    @property
    def used_coverage(self) -> bool:
        return not self.coverage_note


def decide(
    state,
    selection,
    repo: Path,
    base: str | None,
    head: str | None,
    *,
    table: Table | None = None,
    mode: PhaseMode | None = None,
) -> Decision:
    """Apply the record to the map's selection.

    When coverage cannot be used, for any reason at all, the map's selection
    comes back unchanged with the reason in `coverage_note`.
    """
    # Resolved above the try on purpose. A bad env value has to kill the run:
    # below, the broad handler would swallow it, every PR would come back
    # map-only, and that reads as "the mode does nothing".
    mode = mode if mode is not None else mode_from_env()
    out = Decision(steps=set(selection.selected), from_map=set(selection.selected))

    table = table if table is not None else fetch_table()
    if not table.available:
        out.coverage_note = table.unavailable
        return out

    try:
        _apply_record(out, table, selection, repo, base, head, mode)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        # Broad on purpose. A narrower handler would have to decide what to do
        # with a half-built reading, and the only safe answer is nothing.
        out.steps = set(out.from_map)
        out.added_by_coverage.clear()
        out.dropped_by_coverage.clear()
        out.coverage_note = f"coverage unusable ({type(exc).__name__}: {exc})"
    return out


def _steps_at(repo: Path, ref: str) -> tuple[frozenset[str], frozenset[tuple]]:
    """Every step the pipeline yaml declares at `ref`, spelled both ways.

    Both, because neither alone answers "is this step here". `step_id` falls
    back to the label, so a reword makes one step look like two; `identity` is
    rename-tolerant but says nothing about a step that was genuinely renamed
    AND moved. `restrict_to` keeps a candidate matching either.

    Deliberately not `state_for`: that builds an import graph and costs ~17s,
    and the state cache holds two entries, both already spoken for here. This
    parses yaml and nothing else, and the identities are free once it has.
    """
    from .codemap.pipeline.buildkite import load_pipeline_configs, load_steps
    from .codemap.worktree import worktree_at

    tree = worktree_at(repo, ref)
    try:
        configs = load_pipeline_configs(tree)
    except FileNotFoundError:
        # No pipeline at that ref is not an answer about which steps exist,
        # only that we could not look. Empty means "do not restrict", so the
        # add side keeps working instead of silently switching itself off.
        return frozenset(), frozenset()
    steps = [step for config in configs for step in load_steps(tree, config)]
    return frozenset(s.step_id for s in steps), frozenset(s.identity for s in steps)


def _apply_record(
    out: Decision,
    table: Table,
    selection,
    repo: Path,
    base: str,
    head: str | None,
    mode: PhaseMode = DEFAULT_MODE,
) -> None:
    from .codemap.worktree import state_for
    from .coverage.changed_funcs import build as build_query
    from .coverage.changed_funcs import mark_unfaithful

    query = mark_unfaithful(build_query(repo, base, head), table.unfaithful_paths)

    recorded_at = newest_commit(table, repo)
    keys = RowKeys.resolve(table, repo, recorded_at)
    keys.restrict_to(*_steps_at(repo, base))

    stale: frozenset[str] = frozenset()
    if os.environ.get(FRESHNESS_ENV):
        # `RowKeys.resolve` already built and cached this state, so this is
        # cheap.
        surfaces = freshness.build(state_for(repo, recorded_at), recorded_at)
        moved = freshness.changed_between(repo, recorded_at, base)
        stale = frozenset(surfaces.stale_steps(selection.selected, moved))

    union: dict[str, set[str]] = {}
    for row in table._rows.values():
        for path, names in row.functions.items():
            union.setdefault(path, set()).update(names)
    union_names = {p: frozenset(n) for p, n in union.items()}

    _append_op_proxies(query, repo, base, union_names, table)

    reading = read_pr(
        table,
        selection,
        query,
        unknown_names(query, union_names, {}),
        frozenset(union_names),
        keys,
        stale,
        mode=mode,
    )
    out.stale_steps = len(stale)
    out.reasons = dict(reading.reasons)
    out.added_by_coverage = set(reading.added)
    out.dropped_by_coverage = set(reading.dropped)
    out.steps |= out.added_by_coverage
    out.steps -= out.dropped_by_coverage


def _append_op_proxies(query, repo: Path, base: str, union_names, table) -> None:
    """Stand-in queries for changed csrc files: the wrapper names the drop
    side weighs instead of the path, which is never recorded.

    A derived name the record has never seen marks the whole stand-in FAILED,
    which keeps the step. Reading it as "never ran" would be a wrong drop, and
    the per-file name gate cannot catch it when the file itself is recorded.
    """
    from .codemap import native_ops as native_ops_mod
    from .codemap.worktree import state_for
    from .coverage.changed_funcs import Attribution, FileQuery

    if native_ops_mod.mode() != "on":
        return
    if not any(f.path.startswith("csrc/") for f in query.files):
        return
    no = getattr(state_for(repo, base), "native_ops", None)
    if no is None or no.error:
        return
    proxies: dict[str, set[str]] = {}
    for f in query.files:
        if f.proxy or not no.owns(f.path):
            continue
        for wf, quals in (no.proxies_for(f.path) or {}).items():
            proxies.setdefault(wf, set()).update(quals)
    for wf in sorted(proxies):
        quals = frozenset(proxies[wf])
        missing = quals - union_names.get(wf, frozenset())
        query.files.append(
            FileQuery(
                path=wf,
                status=Attribution.FAILED if missing else Attribution.ATTRIBUTED,
                head_names=quals,
                unfaithful=wf in table.unfaithful_paths,
                proxy=True,
                note=(
                    f"op-wrapper proxy; unrecorded names: {sorted(missing)[:3]}"
                    if missing
                    else "op-wrapper proxy"
                ),
            )
        )
