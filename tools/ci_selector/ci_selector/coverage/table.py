# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load an evidence table, or fail in the only safe direction.

Every failure here has the same answer: contribute nothing, and let the map's
selection stand. The failure this must never make is handing back an empty table
that reads as an answer, because "no step runs any changed code" means drop
everything.

So the states are kept apart rather than collapsed into a bool:

  no table at all          nothing was loaded; the record does not run
  nothing to match on      the diff named no function at all; keep it
  no row for this step     the step was never recorded; keep it
  row with no functions    the step recorded and entered no vLLM code; keep it
  row too thin to read     the run behind it was too weak to trust a silence
  row that fails its stamp the bytes disagree with what built them; keep it
  row without the function the only state that is real evidence of absence

Two of those are signals, pointing opposite ways: the last one authorizes a
drop, and `EXECUTES_CHANGE` is what the additive direction selects on. The
rules consuming them live in `rules.py`. Here is the reading, the verification,
and the reasons.
"""

from __future__ import annotations

import gzip
import json
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .changed_funcs import FileQuery, Query, mark_unfaithful
from .model import TABLE_VERSION, Row, Stamp, digest_of


class Evidence(str, Enum):
    """Why a step did or did not get an answer from the table."""

    NO_TABLE = "no-table"
    NO_ROW = "no-row"
    ROW_EMPTY = "row-has-no-functions"
    ROW_THIN = "row-too-thin-to-read-a-silence"
    ROW_FAILED_JOB = "row-built-from-a-failed-job"
    ROW_REJECTED = "row-failed-its-stamp"
    INTERPRETER_MISMATCH = "row-recorded-by-another-python"
    QUERY_FAILS_OPEN = "query-cannot-answer"
    NOTHING_TO_MATCH = "diff-named-no-function"
    NAMELESS_IN_SCOPE = "changed-file-has-no-names-to-match"
    EXECUTES_CHANGE = "row-contains-a-changed-function"
    ABSENT_FROM_ROW = "row-contains-none-of-them"

    @property
    def authorizes_drop(self) -> bool:
        return self is Evidence.ABSENT_FROM_ROW


@dataclass
class Verdict:
    step: str
    evidence: Evidence
    detail: str = ""

    @property
    def keep(self) -> bool:
        return not self.evidence.authorizes_drop


class Table:
    """Rows plus the reasons some of them are not usable."""

    def __init__(
        self,
        rows: dict[str, Row] | None,
        unavailable: str = "",
        rejected: dict[str, str] | None = None,
    ):
        self._rows = rows or {}
        self.unavailable = unavailable
        self.rejected = rejected or {}

    @property
    def available(self) -> bool:
        return not self.unavailable

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def unfaithful_paths(self) -> set[str]:
        """Files whose recorded names their source cannot produce, over all
        rows. A union on purpose: if one step's recording of a file is wrong,
        the recorder cannot spell that file, and matching names on it is unsafe
        everywhere."""
        return {p for row in self._rows.values() for p in row.stamp.unfaithful_files}

    def row(self, step: str) -> Row | None:
        return self._rows.get(step)

    def look_up(self, step: str, query: Query) -> Verdict:
        """What the table can say about one step, given a diff. Not the drop
        rule itself: no completeness check, no caller walk, no file-level
        fallback. This is the reading those are built on."""
        if not self.available:
            return Verdict(step, Evidence.NO_TABLE, self.unavailable)
        if step in self.rejected:
            return Verdict(step, Evidence.ROW_REJECTED, self.rejected[step])
        row = self._rows.get(step)
        if row is None:
            return Verdict(step, Evidence.NO_ROW)
        if not row.stamp.has_evidence:
            return Verdict(step, Evidence.ROW_EMPTY)
        # A silence is only worth as much as the run behind it. A row can hold
        # a great many functions while its tests all skipped, which is nothing
        # but imports, and that reads as "this step does not run your function"
        # for every function you could ask about.
        if row.stamp.thin:
            return Verdict(step, Evidence.ROW_THIN, _why_thin(row.stamp))
        # A step's command list stops at the first failure, so a job that did
        # not pass recorded only a prefix of the step's work. Every other field
        # still reads healthy, since they describe the recorder and not the
        # step. Coarse on purpose: recovering the partial cases means parsing
        # which command died, and a mistake there fails toward dropping.
        if row.stamp.failed_jobs:
            return Verdict(
                step,
                Evidence.ROW_FAILED_JOB,
                f"{len(row.stamp.failed_jobs)} contributing job(s) did not pass",
            )
        mismatch = _interpreter_mismatch(row.stamp)
        if mismatch:
            return Verdict(step, Evidence.INTERPRETER_MISMATCH, mismatch)

        answerable: list[FileQuery] = []
        for changed in query.files:
            if changed.fail_open:
                return Verdict(
                    step,
                    Evidence.QUERY_FAILS_OPEN,
                    f"{changed.path}: {changed.status.value}"
                    + (" unfaithful" if changed.unfaithful else "")
                    + (" residue" if changed.residue else "")
                    + ("" if changed.in_recorder_scope else " outside recorder root"),
                )
            answerable.append(changed)

        # A diff naming no function satisfies "the row contains none of them"
        # for free and would drop every step on an empty set. Emptiness is not
        # absence. The shape that causes it is a file inside the recorder root
        # with no Python in it, like a tuned-kernel config: nothing fails open,
        # nothing can ever match, and mixed with a real file it would otherwise
        # be ignored while being the reason the step was selected at all.
        blind = next(
            (
                c.path
                for c in answerable
                if c.in_recorder_scope and not c.names and not c.function_names
            ),
            None,
        )
        if blind is not None:
            return Verdict(
                step,
                Evidence.NAMELESS_IN_SCOPE,
                f"{blind} is inside the recorder root and names no function",
            )

        if not any(changed.names for changed in answerable):
            return Verdict(
                step,
                Evidence.NOTHING_TO_MATCH,
                f"{len(answerable)} changed files, none of them naming a function",
            )

        for changed in answerable:
            for name in changed.names:
                if row.contains(changed.path, name):
                    return Verdict(
                        step, Evidence.EXECUTES_CHANGE, f"{changed.path}::{name}"
                    )
        return Verdict(step, Evidence.ABSENT_FROM_ROW)


def _why_thin(stamp: Stamp) -> str:
    """Which weakness makes the row unreadable. Diagnosis, never load-bearing.
    The empty-row case never arrives, since `look_up` reports that as
    `ROW_EMPTY` first, being the more specific answer."""
    if stamp.clean_exits == 0:
        return "no contributing process exited cleanly"
    if not stamp.shards_complete:
        return "no build recorded every declared shard"
    if stamp.lost_lines:
        return "a contributing process lost lines"
    return f"{stamp.jobs_ran_no_tests} contributing job(s) executed no test"


def _interpreter_mismatch(stamp: Stamp) -> str:
    """Names are only comparable across a matching Python minor version. Patch
    level does not matter, which was measured rather than assumed."""
    ours = f"{sys.version_info.major}.{sys.version_info.minor}"
    theirs = {v.rsplit(".", 1)[0] for v in stamp.interpreters if v}
    if theirs and ours not in theirs:
        return f"row recorded by {sorted(theirs)}, reading with {ours}"
    return ""


def _verify(key: str, blob: dict) -> tuple[Row | None, str]:
    """A row is usable only if its content still matches the stamp that built it."""
    try:
        stamp = Stamp(**blob["stamp"])
        functions = {p: frozenset(n) for p, n in blob["functions"].items()}
    except (KeyError, TypeError) as exc:
        return None, f"unreadable row: {exc}"

    if len(functions) != stamp.n_files:
        return None, f"file count {len(functions)} != stamp {stamp.n_files}"
    total = sum(len(v) for v in functions.values())
    if total != stamp.n_functions:
        return None, f"function count {total} != stamp {stamp.n_functions}"
    if stamp.digest and digest_of(functions, stamp) != stamp.digest:
        return None, "digest mismatch"
    return Row(
        key=key, keyed=bool(blob.get("keyed")), functions=functions, stamp=stamp
    ), ""


def load(path: Path) -> Table:
    """Read a table. Any problem yields a table that authorizes nothing."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return Table(None, unavailable=f"cannot read {path}: {exc}")
    try:
        text = gzip.decompress(raw) if path.suffix == ".gz" else raw
        payload = json.loads(text)
    except Exception as exc:
        return Table(None, unavailable=f"cannot parse {path}: {exc}")

    # A table written by another shape of stamp cannot be judged by this one.
    # A missing field takes its default, and every default is the healthy
    # value, so an older table would read healthier than it was recorded.
    # Refusing is the only reading that fails safe.
    version = payload.get("version")
    if version != TABLE_VERSION:
        return Table(
            None,
            unavailable=f"{path} is table version {version!r}, expected "
            f"{TABLE_VERSION}; re-merge it from the raw recordings",
        )

    blobs = payload.get("rows")
    if not isinstance(blobs, dict):
        return Table(None, unavailable=f"{path} has no rows object")
    # An empty rows object is a broken table, never the answer "nothing matched".
    if not blobs:
        return Table(None, unavailable=f"{path} contains no rows")

    rows: dict[str, Row] = {}
    rejected: dict[str, str] = {}
    for key, blob in blobs.items():
        row, why = _verify(key, blob)
        if row is None:
            rejected[key] = why
        else:
            rows[key] = row
    if not rows:
        return Table(
            None,
            unavailable=f"{path}: every row failed verification",
            rejected=rejected,
        )
    return Table(rows, rejected=rejected)


def apply(table: Table, query: Query, steps: list[str]) -> list[Verdict]:
    """Verdicts for a map selection. Wiring for the smoke test, not the rule."""
    mark_unfaithful(query, table.unfaithful_paths)
    return [table.look_up(step, query) for step in steps]


def main() -> None:
    import argparse

    from ..gitdiff import resolve_diff_ref
    from .changed_funcs import build

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", type=Path)
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--ref", help="'A...B', 'A..B', or 'A' vs the worktree")
    ap.add_argument("--steps", nargs="*", help="step keys to ask about; default all")
    args = ap.parse_args()

    table = load(args.table)
    print(f"table: {len(table)} rows, available={table.available} {table.unavailable}")
    if table.rejected:
        print(f"  rejected rows: {len(table.rejected)}")
        for key, why in list(table.rejected.items())[:10]:
            print(f"      {key}: {why}")
    if not (args.repo and args.ref):
        return

    base, head = resolve_diff_ref(args.repo, args.ref)
    query = build(args.repo, base, head)
    steps = args.steps or sorted(table._rows)
    verdicts = apply(table, query, steps)

    from collections import Counter

    counts = Counter(v.evidence.value for v in verdicts)
    print(f"\n{len(query.files)} changed files, query fail_open={query.fail_open}")
    for reason, n in counts.most_common():
        print(f"  {n:5d}  {reason}")
    droppable = [v for v in verdicts if v.evidence.authorizes_drop]
    print(f"\n{len(droppable)} of {len(steps)} steps have evidence of absence")


if __name__ == "__main__":
    main()
