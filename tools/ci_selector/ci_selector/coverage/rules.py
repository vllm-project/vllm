# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The rule the coverage record applies to one PR. Two directions, gated
differently.

ADD: a row showing a step ran a changed file selects it, whatever the code map
concluded. Ungated, because one observation proves presence.

DROP: a row showing a step ran none of them removes it, but only once every
gate below agrees. Proving ABSENCE needs the recorder to have been watching the
right file, during a run that finished, on code that still exists. Every rung
resolves toward keeping.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .changed_funcs import Query
from .table import Table


def row_key_for(step_id: str) -> str | None:
    """The bare row key inside a step id, or None. Strips the pipeline prefix
    and nothing else, so a mirror keeps its hardware suffix and matches no row.
    A bare key is unique only WITHIN a pipeline, so this alone will hand a step
    another pipeline's row. Use `RowKeys`."""
    pipeline, sep, rest = step_id.partition(":")
    return rest if sep and rest else None


def newest_commit(table: Table, repo: Path) -> str:
    """The checkout to resolve row keys against: the latest commit any row saw.

    Picked by history rather than sort order, which agreed only by luck.
    Production reads one sweep, so one commit, and takes the early return with
    no subprocess; the ordering is for multi-build tables we measure with.
    """
    commits = {c for row in table._rows.values() for c in row.stamp.commits}
    if not commits:
        return "HEAD"
    if len(commits) == 1:
        return next(iter(commits))
    ordered = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--date-order", "--no-walk", *commits],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return ordered[0] if ordered else sorted(commits)[0]


class RowKeys:
    """Which steps a table is entitled to answer for.

    Recordings carry no pipeline, and a bare key is not unique across configs:
    a couple of dozen are defined by two at once. So the owner is derived,
    whichever config's keys explain the most rows, and steps from any other get
    no row. Nothing is named or curated, so a sweep of a different pipeline
    resolves on its own.

    The ambiguity is unreachable today, since only `vllm_ci` is ever emitted.
    TODO: if a second config runs on a PR, the shared keys go live and this
    needs a real tie-break rather than "whoever explains the most".

    Row identity comes from the resolved step and not its printed id, because
    the generator spells a mirror the other way round from us. A step the
    checkout does not explain reads no row at all.
    """

    def __init__(
        self,
        owners: set[str],
        match_rate: dict[str, float],
        spellings: dict[str, str] | None = None,
        steps: dict[str, object] | None = None,
    ):
        self.owners = owners
        self.match_rate = match_rate
        self.spellings = spellings
        # step_id -> Step, for the owning pipelines only. The additive half
        # needs the step objects to ask `match_jobs` whether an added step
        # covers a failure, and a raw key lookup would miss every sharded job.
        self.steps = steps or {}

    @classmethod
    def resolve(cls, table: Table, repo: Path, ref: str) -> RowKeys:
        from ci_selector.codemap.worktree import state_for

        rows = set(table._rows)
        spellings: dict[str, str] = {}
        match_rate: dict[str, float] = {}
        for pipeline in state_for(repo, ref).pipelines:
            idents = set()
            for step in pipeline.steps:
                ident = step.buildkite_key or step.label
                spellings[step.step_id] = ident
                idents.add(ident)
            match_rate[pipeline.config.name] = len(rows & idents) / len(rows)
        resolved = cls.from_match_rates(match_rate, spellings)
        resolved.steps = {
            step.step_id: step
            for pipeline in state_for(repo, ref).pipelines
            if pipeline.config.name in resolved.owners
            for step in pipeline.steps
        }
        return resolved

    @classmethod
    def from_match_rates(
        cls, match_rate: dict[str, float], spellings: dict[str, str] | None = None
    ) -> RowKeys:
        """Pick the owner, separated from the state building so it can be
        tested. No match rates at all means no owner, so no step resolves to a
        row and the table changes nothing."""
        ranked = sorted(match_rate.items(), key=lambda kv: -kv[1])
        owners = {ranked[0][0]} if ranked else set()
        return cls(owners, match_rate, spellings)

    def candidates(self) -> list[str]:
        """Every step the table may ADD, which is not every step it can answer
        for.

        The manual-only filter is doing real work here, not tidying: the sweeps
        unblocked the optional steps so the table would cover them, so it holds
        rows for nightlies and mirrors CI never runs on a PR. Picking one is
        not over-selection, it names a step the generator will not emit.

        The subtractive direction needs no such filter, its population being
        the map's selection, which holds no manual-only step.
        """
        return [
            sid
            for sid, step in self.steps.items()
            # Defaults to manual_only=True: this filter is a safety gate, so an
            # object that cannot answer the question is not addable.
            if self.key_for(sid) is not None and not getattr(step, "manual_only", True)
        ]

    def key_for(self, step_id: str) -> str | None:
        """The row this step may consult, or None when it may consult none."""
        pipeline, sep, rest = step_id.partition(":")
        if not sep or not rest or pipeline not in self.owners:
            return None
        if self.spellings is None:
            return rest
        return self.spellings.get(step_id)


@dataclass
class Reading:
    """One reading of the record over one PR."""

    dropped: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)
    # Steps the record removed whose CI job actually failed. The number that matters
    # more than any ratio in this file.
    dropped_and_failed: list[str] = field(default_factory=list)
    # Steps the RECORD selected that the map did not: a row shows the step
    # executing a changed file. Addition needs one observation where removal
    # needs the recording to be complete, so this half carries no gate.
    added: list[str] = field(default_factory=list)
    # Additions whose CI job actually failed: the recall this half buys, and
    # the mirror of dropped_and_failed.
    added_and_failed: list[str] = field(default_factory=list)


def unknown_names(
    query: Query,
    union_names: dict[str, frozenset[str]],
    fresh: dict[str, frozenset[str]],
) -> dict[str, set[str]]:
    """Changed names the record has no business reasoning about, by file. Two
    sources, which stack rather than compete.

    A name no row has ever recorded is unknown outright: missing from one
    step's row is evidence about that step, missing from every row is no data.
    That is a rule of the design, not an artifact of our table.

    A name that did not exist at the PR's base is unknown too, but only our
    table hides that, having been recorded after those PRs merged. That half is
    measurement bias, so pass an empty dict to leave it out.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for changed in query.files:
        missing = changed.names - union_names.get(changed.path, frozenset())
        if missing:
            out[changed.path] |= missing
    for path, names in fresh.items():
        out[path] |= set(names)
    return dict(out)


def read_pr(
    table: Table,
    selection,
    query: Query,
    unresolved: dict[str, set[str]],
    known_files: frozenset[str],
    keys: RowKeys,
    stale: frozenset[str] = frozenset(),
    *,
    failed_ran: dict[str, str] | None = None,
    matched_slugs: dict[str, list[str]] | None = None,
    failed_missed: dict[str, str] | None = None,
) -> Reading:
    """The record over one PR's map selection.

    Takes a `Selection` so it can run at decide time. It used to take a
    crosscheck record, which confined it to the harness, because half of that
    is CI outcomes that do not exist yet while a PR is being selected for.
    Those survive as keyword arguments the harness passes and decide time does
    not.

    The query and the unknown-name set come from the caller, since both cost a
    git read per file and both want the same answer.
    """
    reading = Reading()
    failed = set(failed_ran or {})
    matched = matched_slugs or {}
    # Absent means the step is not droppable, not droppable against
    # everything. The old fallback claimed to be the more careful reading,
    # which was true of the scope and false of the authority it granted.
    attribution = selection.selected_paths or {}

    for step_id in selection.selected:
        key = keys.key_for(step_id)
        if key is None:
            reading.kept.append(step_id)
            reading.reasons["unmappable-step-id"] += 1
            continue

        # THE INVARIANT: scoping may narrow what you examine, never who may be
        # doubted. So positive evidence is read against the WHOLE diff, before
        # any narrowing. A row that plainly runs a changed function makes its
        # step relevant whatever path the map happened to cite.
        #
        # `names`, not `function_names`: an import counts as use. Excluding the
        # import-time frames is the stricter reading, and it is defensible in
        # the abstract, since every importer runs them so they separate
        # nothing. But a step that imports a file it never otherwise touches
        # can still break when that file changes, and that is what this buys.
        row = table.row(key)
        if row is not None and any(
            row.contains(f.path, name) for f in query.files for name in f.names
        ):
            reading.kept.append(step_id)
            reading.reasons["row-executes-a-changed-function"] += 1
            continue

        # The row describes the step as it was at the table's commit. If the
        # step's tests or its definition moved since, it describes a step that
        # no longer exists. Below the positive rung because both outcomes keep,
        # so the order only decides which counter gets the credit.
        if step_id in stale:
            reading.kept.append(step_id)
            reading.reasons["row-is-stale"] += 1
            continue

        # The other half of the same invariant. A step narrows only when EVERY
        # reason the map had for it is one that function evidence can overturn.
        # A single reason that cannot be holds the step on its own.
        reasons = attribution.get(step_id)
        if reasons is None:
            # No attribution is not "argue with the whole diff". That confuses
            # a scope decision with an authority one, and on an input missing
            # the key it made every step droppable, forced ones included.
            reading.kept.append(step_id)
            reading.reasons["no-attribution"] += 1
            continue
        if any(r is None for r in reasons):
            reading.kept.append(step_id)
            reading.reasons["not droppable-reason"] += 1
            continue
        scope = {p for r in reasons for p in r}
        if not scope:
            reading.kept.append(step_id)
            reading.reasons["no-attributed-file"] += 1
            continue

        scoped = query.restrict(scope)
        # The unknown-name gates, per step and not per PR. Whole-diff was the
        # same mistake as the fail-open: an unknown name in a file unrelated to
        # this step's selection kept every step on the PR.
        local = {p: n for p, n in unresolved.items() if p in scope}
        unseen = [p for p in local if p not in known_files]
        if unseen:
            reading.kept.append(step_id)
            reading.reasons["unknown-code-blocks-narrowing"] += 1
            continue

        evidence = table.look_up(key, scoped).evidence
        if evidence.authorizes_drop and local:
            # The file projection the design specifies under an unknown
            # function: a step whose row touches a file carrying unknown code
            # keeps, because that code could be reached from what it does run.
            row = table.row(key)
            if row is not None and any(p in row.functions for p in local):
                reading.kept.append(step_id)
                reading.reasons["file-projection-keeps"] += 1
                continue

        reading.reasons[evidence.value] += 1
        if evidence.authorizes_drop:
            reading.dropped.append(step_id)
            if set(matched.get(step_id, ())) & failed:
                reading.dropped_and_failed.append(step_id)
        else:
            reading.kept.append(step_id)

    _add_from_rows(table, selection, query, keys, reading, unresolved, failed_missed)
    return reading


def _add_from_rows(
    table: Table,
    selection,
    query: Query,
    keys: RowKeys,
    reading: Reading,
    unresolved: dict[str, set[str]],
    failed_missed: dict[str, str] | None = None,
) -> None:
    """The direction the record is usually forgotten to have: it SELECTS, not
    only removes.

    A row showing a step ran a changed file proves relevance whatever the map
    thought, so this walks every step the table can answer for rather than the
    map's selection. No completeness or freshness gate, on purpose: presence
    needs one observation where absence needs the whole recording.

    `unresolved` brackets this as it does the drop side. Our table postdates
    every measured PR, so it holds functions those PRs added, while in
    production such a name has never been recorded and no row could show it.
    Matching them anyway would credit an addition production could not make, so
    the pessimistic pass declines them and `added` is a bracket, not a figure.

    Measured: that bracket is inert, because most adds fire on an import-time
    frame, which is old code by definition. Which also says what this half
    mostly asserts, that the step imports the changed file. Weak, and the
    likeliest reason it costs something and catches nothing.

    Scored against the failures the map's selection did not cover, through the
    crosscheck's own matcher, since job slugs carry shard suffixes a raw key
    comparison misses.
    """
    already = set(selection.selected)
    for step_id in keys.candidates():
        if step_id in already:
            continue
        key = keys.key_for(step_id)
        row = table.row(key) if key else None
        if row is None:
            continue  # no row: the map decides, and the map did not pick it
        if any(
            row.contains(f.path, name)
            for f in query.files
            for name in f.names - set(unresolved.get(f.path, ()))
        ):
            reading.added.append(step_id)

    # Counted here, above the failure scoring, so this is how often the half
    # fired at all. Below the early return it only counted adds on PRs that
    # also had an uncovered failure, which is almost none, so the histogram
    # read as though the half never ran.
    reading.reasons["row-adds-a-step-the-map-missed"] += len(reading.added)

    missed = failed_missed or {}
    if not (missed and reading.added):
        return
    from ..codemap.pipeline.match import match_jobs

    steps = [keys.steps[s] for s in reading.added if s in keys.steps]
    _, _, by_step = match_jobs(dict(missed), steps)
    reading.added_and_failed = sorted(by_step)


def _mark(query: Query, table: Table) -> Query:
    from ci_selector.coverage.changed_funcs import mark_unfaithful

    return mark_unfaithful(query, table.unfaithful_paths)
