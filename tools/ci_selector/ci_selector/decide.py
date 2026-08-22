# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combine the two inputs into one job list.

Per changed file F and step S:

    the row shows S ran F     -> select. The map gets no vote.
    the row shows S ran none  -> drop, if every gate agrees.
    S has no row              -> the map decides.

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

from dataclasses import dataclass, field
from pathlib import Path

from .coverage import freshness
from .coverage.rules import RowKeys, newest_commit, read_pr, unknown_names
from .coverage.source import fetch_table
from .coverage.table import Table


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
) -> Decision:
    """Apply the record to the map's selection.

    When coverage cannot be used, for any reason at all, the map's selection
    comes back unchanged with the reason in `coverage_note`.
    """
    out = Decision(steps=set(selection.selected), from_map=set(selection.selected))

    table = table if table is not None else fetch_table()
    if not table.available:
        out.coverage_note = table.unavailable
        return out

    try:
        _apply_record(out, table, selection, repo, base, head)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        # Broad on purpose. A narrower handler would have to decide what to do
        # with a half-built reading, and the only safe answer is nothing.
        out.steps = set(out.from_map)
        out.added_by_coverage.clear()
        out.dropped_by_coverage.clear()
        out.coverage_note = f"coverage unusable ({type(exc).__name__}: {exc})"
    return out


def _apply_record(
    out: Decision,
    table: Table,
    selection,
    repo: Path,
    base: str,
    head: str | None,
) -> None:
    from .codemap.worktree import state_for
    from .coverage.changed_funcs import build as build_query
    from .coverage.changed_funcs import mark_unfaithful

    query = mark_unfaithful(build_query(repo, base, head), table.unfaithful_paths)

    recorded_at = newest_commit(table, repo)
    keys = RowKeys.resolve(table, repo, recorded_at)

    # `RowKeys.resolve` already built and cached this state, so this is cheap.
    surfaces = freshness.build(state_for(repo, recorded_at), recorded_at)
    moved = freshness.changed_between(repo, recorded_at, base)
    stale = frozenset(surfaces.stale_steps(selection.selected, moved))

    union: dict[str, set[str]] = {}
    for row in table._rows.values():
        for path, names in row.functions.items():
            union.setdefault(path, set()).update(names)
    union_names = {p: frozenset(n) for p, n in union.items()}

    reading = read_pr(
        table,
        selection,
        query,
        unknown_names(query, union_names, {}),
        frozenset(union_names),
        keys,
        stale,
    )
    out.stale_steps = len(stale)
    out.reasons = dict(reading.reasons)
    out.added_by_coverage = set(reading.added)
    out.dropped_by_coverage = set(reading.dropped)
    out.steps |= out.added_by_coverage
    out.steps -= out.dropped_by_coverage
