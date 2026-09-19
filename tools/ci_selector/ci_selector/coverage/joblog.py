# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test outcomes for a job, read off its Buildkite log.

A row built from a job whose tests all skipped holds little more than that
job's imports, and nothing else in the row says so: the function counts look
healthy and the processes exited cleanly, while the row describes almost none
of what the step is meant to cover. So the counts travel with the row.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

# Buildkite interleaves timestamp markers into the stream, sometimes mid-word,
# so a count can arrive with one glued into it. Strip those and the colour
# codes first, or the counts quietly come out low.
_NOISE = re.compile(r"\x1b\[[0-9;]*m|_bk;t=\d+")

_OUTCOMES = (
    "passed",
    "failed",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
    "errors",
    "error",
)
_OUTCOME = re.compile(r"(\d+) (" + "|".join(_OUTCOMES) + r")\b")

# Counted inside summary lines only, since scanning the whole log double-counts
# what pytest also prints on the collection line. The elapsed time identifies a
# summary, because the outcome words vary and an all-skipped run names none.
#
# If pytest changes this format it matches nothing, and on its own that reads
# HEALTHY, not weak: every count stays zero, so `ran_nothing` cannot fire. That
# is why `summary_unparsed` exists and watches `collected` instead, which a
# separate regex takes off the collection line.
_SUMMARY = re.compile(r"=+ ([^=\n]*? in \d[\d.]*s[^=\n]*?) =+")


@dataclass
class TestCounts:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    deselected: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errors: int = 0
    collected: int = 0
    invocations: int = 0  # summaries seen; none means the step is not pytest

    @property
    def executed(self) -> int:
        return self.passed + self.failed + self.xfailed + self.xpassed + self.errors

    @property
    def ran_nothing(self) -> bool:
        """pytest ran and executed no test at all.

        No threshold on purpose. "Almost all skipped" needs a fraction nobody
        can derive, while the case that matters is clear-cut: a job that
        collected tests and ran none of them recorded its imports and nothing
        else. Silent when the step is not pytest, since those are mostly shell
        scripts and their rows are not weak for it.

        No `skipped > 0` clause: it used to be here and let two more
        ran-nothing shapes through, an all-deselected run and pytest's own
        "no tests ran", both of which report zero skips.
        """
        return self.invocations > 0 and self.executed == 0

    @property
    def summary_unparsed(self) -> bool:
        """pytest collected tests and printed no summary we could read.

        The witness for `_SUMMARY` drift, and it has to be independent or it
        cannot see its own parser fail: `collected` comes off the collection
        line, a different regex. Also catches a job killed mid-run, which
        prints a collection line and never reaches a summary. Silent on a
        non-pytest step, which collects nothing.
        """
        return self.collected > 0 and self.invocations == 0


@dataclass
class LogSummary:
    counts: TestCounts = field(default_factory=TestCounts)
    unreadable: bool = False


def read_counts(path: Path) -> LogSummary:
    """Test outcomes summed over every pytest invocation in one job's log.
    Summed rather than read off the last summary, because a step usually runs
    several pytest commands and only the total describes the whole run."""
    try:
        raw = path.read_bytes()
        text = gzip.decompress(raw) if path.suffix == ".gz" else raw
        body = _NOISE.sub("", text.decode(errors="ignore"))
    except (OSError, EOFError):
        # EOFError is not an OSError. A partially uploaded `.gz` raises it, and
        # uncaught it takes down the whole sweep rather than one job's counts.
        return LogSummary(unreadable=True)

    counts = TestCounts()
    summaries = _SUMMARY.findall(body)
    for summary in summaries:
        for number, outcome in _OUTCOME.findall(summary):
            name = "errors" if outcome.startswith("error") else outcome
            setattr(counts, name, getattr(counts, name) + int(number))
    counts.collected = sum(int(n) for n in re.findall(r"collected (\d+) items?", body))
    counts.invocations = len(summaries)
    return LogSummary(counts=counts)


# A verbose per-test line: the node id, then the verdict. Stripping the
# timestamp marker leaves its control bytes behind, so the node id can arrive
# glued to them, hence the explicit strip rather than anchoring at the start.
_SKIPPED = re.compile(r"([\w./\\-]+\.py(?:::[^\s]+)?)?\s*\bSKIPPED\b")

# A node id printed with no verdict, because vLLM's own logging interleaved and
# pushed the verdict onto a later line. A good fraction of them arrive this way,
# so tracking the pending id is not an edge case.
_NODE = re.compile(r"([\w./\\-]+\.py::[^\s]+)")


def read_skipped_nodes(path: Path) -> tuple[list[str], int]:
    """Pytest node ids that skipped, plus how many skips had no id at all.

    Kept out of `read_counts` and never stamped: a sweep holds far too many
    skips to put in every table, and nothing in the drop path reads them. This
    is analysis input, loaded from the raw log.

    The ids are relative to pytest's invocation cwd, which varies per step and
    which the declared working_dir does not reliably predict, since a step that
    runs inside a container uses the container's directory instead. Resolving
    them to repo paths is done elsewhere, by probing the catalog.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return [], 0
    if path.suffix == ".gz":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError):
            return [], 0
    body = _NOISE.sub("", raw.decode("utf-8", "replace"))

    nodes: list[str] = []
    unattributed = 0
    pending: str | None = None
    for line in body.splitlines():
        hit = _SKIPPED.search(line)
        if hit is None:
            found = _NODE.search(line)
            if found:
                pending = found.group(1)
            continue
        node = hit.group(1)
        if node and "::" not in node:
            # A bare filename before SKIPPED is not a node id.
            node = None
        if node is None:
            node = pending
        if node is None:
            unattributed += 1
        else:
            nodes.append(node)
        pending = None
    return nodes, unattributed
