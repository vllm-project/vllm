# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The coverage table's data model: what a row is, what a stamp records, and
how a table is verified.

Split from the merger so the read path does not import the offline builder.
The version and the digest live here, on the verifying side, and `build.py`
imports them to stamp what it writes. The other direction would let a builder
and a reader disagree, every row would fail verification, and that reads as
"keep everything": safe, and therefore an invisible bug.

One rule before reading `Stamp`. A file whose recorded names its own source
cannot produce is one the recorder cannot spell, which happens when a decorator
rewrites the tree at runtime. Matching names there means nothing and is wrong
in the direction that drops steps, so those files are flagged and the record
says nothing about them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import types
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from inspect import CO_OPTIMIZED
from pathlib import Path

from ..handwritten import RECORDER_SCOPE

JOB_META_KEY = "step_key"
JOB_META_LABEL = "label"
JOB_META_PARALLEL_TOTAL = "parallel_total"
JOB_META_PARALLEL_INDEX = "parallel_index"
BUILDKITE_JOB_ID_ENV = "BUILDKITE_JOB_ID"
BUILDKITE_RETRY_COUNT_ENV = "BUILDKITE_RETRY_COUNT"

ENV_ABSENT = "<unset>"

# The producer writes recordings into `<checkout>/.fnrec/<job-id>/` and the
# Buildkite agent uploads them from there. Named once, so the half that writes
# the job directory and the half that reads it cannot drift apart silently.
FNREC_DIR = ".fnrec"
RECORD_GLOB = "fn.*.txt"

# A build whose started jobs recorded less often than this is a delivery
# failure, so the merge CLI refuses to write a table from it without
# --allow-partial. Healthy builds measured no worse than 0.66.
MIN_RECORD_RATE = 0.50

# A name held by more than this share of rows says nothing about any one of
# them, so the add side will not select on it. `<module>` is in every row.
MAX_ADD_ROW_SHARE = 0.90

# Too few rows to measure a share at all: with 3 rows a name is in 33%, 67% or
# 100% of them and none of those means anything. Under this count the gate
# above is skipped and every name is accepted.
MIN_ROWS_FOR_BREADTH = 20

# Bumped whenever the stamp's shape changes. `load` refuses a version it does
# not know, which is the only thing stopping an older table from reading
# healthier than it was recorded: a missing field takes its default, and every
# default here is the healthy value.
TABLE_VERSION = 4

# Fingerprint of `Stamp`'s fields, so remembering to bump the version above is a
# mechanism and not a discipline. A test recomputes it and fails when the two
# disagree. Change both together, in the same commit that changes the stamp.
STAMP_SHAPE = "060984df5663c507"

MIRROR_NOTE = (
    "A mirror owns its own row and never inherits its parent's. Keyless mirrors "
    "carry their own label, which already produces that; do not 'fix' it into "
    "row sharing. NVIDIA evidence must never clear an AMD job, and a mirror may "
    "override commands and env entirely."
)


@dataclass
class ProcessRecord:
    file: str
    root: str
    job: str | None
    py: str | None
    retry: str | None
    functions: dict[str, set[str]]
    data_lines: int
    counter: int | None  # last root=N written, a floor on what was recorded
    clean_exit: bool
    errors: int
    outside_root: int
    malformed: int

    @property
    def lost_lines(self) -> bool:
        """Fewer records than the counter promised. A missing end marker is not
        this, since plenty of processes are killed by design."""
        return self.counter is not None and self.data_lines < self.counter


def _kv(parts: list[str]) -> dict[str, str]:
    return dict(p.split("=", 1) for p in parts if "=" in p)


def read_process(path: Path) -> ProcessRecord | None:
    """One fnrec process file, read all the way to EOF."""
    raw: list[tuple[str, str]] = []
    header_root = effective_root = None
    job = py = retry = None
    counter = None
    clean_exit = False
    errors = 0
    malformed = 0

    with open(path, errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            tag = parts[0]
            if tag == "#start":
                meta = _kv(parts[1:])
                header_root = meta.get("root") or None
                job = meta.get(BUILDKITE_JOB_ID_ENV)
                py = meta.get("py")
                retry = meta.get(BUILDKITE_RETRY_COUNT_ENV)
            elif tag == "#root":
                effective_root = parts[1] if len(parts) > 1 and parts[1] else None
            elif tag in ("#stat", "#end"):
                meta = _kv(parts[1:])
                if "root" in meta:
                    counter = int(meta["root"] or 0)
                # Accumulate. Assigning let a trailing `errors=0` erase every
                # `#error` line counted before it.
                errors += int(meta.get("errors", 0) or 0)
                if tag == "#end":
                    clean_exit = True
            elif tag == "#error":
                errors += 1
            elif len(parts) == 3:
                raw.append((parts[0], parts[1]))
            else:
                malformed += 1

    root = effective_root or header_root
    if root is None:
        return None
    if not root.endswith("/"):
        root += "/"
    # The root ends at the package directory, so the repo-relative path keeps it.
    prefix = RECORDER_SCOPE

    functions: dict[str, set[str]] = defaultdict(set)
    outside = 0
    for filename, qualname in raw:
        if not filename.startswith(root):
            outside += 1
            continue
        functions[prefix + filename[len(root) :]].add(qualname)

    return ProcessRecord(
        file=path.name,
        root=root,
        job=job,
        py=py,
        retry=retry,
        functions=dict(functions),
        data_lines=len(raw),
        counter=counter,
        clean_exit=clean_exit,
        errors=errors,
        outside_root=outside,
        malformed=malformed,
    )


def row_key(meta: dict) -> tuple[str, bool]:
    """(row identity, whether it came from a step key).

    Raw Buildkite identity, resolved to a step id only at read time, so the
    merger needs no checkout and the keyless jobs keep their rows.
    """
    key = meta.get(JOB_META_KEY)
    if key:
        return key, True
    label = meta.get(JOB_META_LABEL) or ""
    total, index = meta.get(JOB_META_PARALLEL_TOTAL), meta.get(JOB_META_PARALLEL_INDEX)
    if total and index is not None:
        suffix = f" {index + 1}"
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label, False


class UnresolvableCommit(Exception):
    """The merge repo cannot see the commit a recording was taken at.

    Fatal rather than tolerated, because every symptom of it is a lie: `names`
    and `exists` both fail for every path, so the merger reads the build as one
    where no file existed and drops all of them. The row comes out empty and
    thin, which is safe alone, but unioned with a good build it adds nothing
    while lending the merged row its completeness and its vintage.
    """


class SourceIndex:
    """Names a file's source compiles to, at one commit. Cached per path."""

    def __init__(self, repo: Path, commit: str):
        self.repo = repo
        self.commit = commit
        # path -> (every qualname, the subset that runs on import). One entry,
        # because both fall out of the same compile.
        self._cache: dict[str, tuple[frozenset[str], frozenset[str]] | None] = {}

    def require_commit(self) -> None:
        """Raise unless the commit resolves here. Call before reading a build."""
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "rev-parse",
                "--verify",
                f"{self.commit}^{{commit}}",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise UnresolvableCommit(
                f"commit {self.commit} is not in {self.repo}. A shallow clone, the "
                "wrong remote, or an unfetched branch will do this. Fetch it and "
                "re-merge; merging anyway silently drops every recorded file."
            )

    def _compiled(self, path: str) -> tuple[frozenset[str], frozenset[str]] | None:
        if path in self._cache:
            return self._cache[path]
        proc = subprocess.run(
            ["git", "-C", str(self.repo), "show", f"{self.commit}:{path}"],
            capture_output=True,
        )
        result: tuple[frozenset[str], frozenset[str]] | None = None
        if proc.returncode == 0:
            try:
                source = proc.stdout.decode()
                code = compile(source, path, "exec")
                every = frozenset(_qualnames(code))
                at_import = frozenset(
                    c.co_qualname
                    for c in _code_objects(code)
                    if not c.co_flags & CO_OPTIMIZED
                )
                result = (every, at_import)
            except Exception:
                result = None
        self._cache[path] = result
        return result

    def names(self, path: str) -> frozenset[str] | None:
        """None means the file is absent, unreadable, or will not compile."""
        compiled = self._compiled(path)
        return None if compiled is None else compiled[0]

    def import_time_names(self, path: str) -> frozenset[str]:
        """Names whose code runs on import: the module body and every class body.

        Read off the compiler, not guessed from the name. Empty when the file
        cannot be read, which leaves its names unclassified and so treated as
        calls, the fail-open direction.

        `changed_funcs.import_time_names` makes the same reading on the diff
        side. Separate because that one takes source it already holds and this
        one takes a path at a commit.
        """
        compiled = self._compiled(path)
        return frozenset() if compiled is None else compiled[1]

    def exists(self, path: str) -> bool:
        return (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo),
                    "cat-file",
                    "-e",
                    f"{self.commit}:{path}",
                ],
                capture_output=True,
            ).returncode
            == 0
        )


def _code_objects(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _qualnames(code: types.CodeType):
    yield code.co_qualname
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _qualnames(const)


@dataclass
class Stamp:
    """Everything a reader needs to judge a row without the raw bytes."""

    jobs: list[str] = field(default_factory=list)
    builds: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    interpreters: list[str] = field(default_factory=list)
    processes: int = 0
    clean_exits: int = 0
    # Contributing jobs Buildkite did not mark passed. A step's commands stop
    # at the first failure, so the rest never ran and never recorded, while
    # every other field here still reads healthy.
    failed_jobs: list[str] = field(default_factory=list)
    # build -> which shard indexes produced a recording, and how many the step
    # declares. Indexes rather than a count: two jobs for the same shard, which
    # a manual retry produces, counted as two shards and read as complete.
    shards_seen: dict[str, list[int]] = field(default_factory=dict)
    shards_expected: dict[str, int] = field(default_factory=dict)
    # Recorded names that are module or class bodies. Counted so `_verify` can
    # cross-check them the way it cross-checks n_files.
    n_import_time: int = 0
    lost_lines: bool = False
    outside_root_lines: int = 0
    malformed_lines: int = 0
    process_errors: int = 0
    n_files: int = 0
    n_functions: int = 0
    # Files whose recorded names their source cannot produce. No answer here.
    unfaithful_files: list[str] = field(default_factory=list)
    dropped_absent_files: int = 0
    # Test outcomes across the contributing jobs. Function counts alone cannot
    # tell a job that exercised the step from one whose tests all skipped and
    # recorded little more than its imports.
    tests_executed: int = 0
    tests_skipped: int = 0
    tests_collected: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_deselected: int = 0
    tests_xfailed: int = 0
    tests_xpassed: int = 0
    tests_errors: int = 0
    pytest_invocations: int = 0
    jobs_ran_no_tests: int = 0
    # Jobs that collected tests and printed no readable summary. Separate from
    # `jobs_ran_no_tests` because it is our parser failing, not the step.
    jobs_summary_unparsed: int = 0
    logs_unreadable: int = 0
    # The build's own identity. Without it a torch-nightly table is
    # byte-indistinguishable from a plain one, and "only the daily sweeps feed
    # the acting table" is prose with nothing enforcing it.
    pipeline_slug: str = ""
    # Contributing builds whose world file could not be read, so their env is
    # unknown rather than empty. Diagnostic: an unread world does not weaken
    # the recording, it only means `build_env` is short.
    worlds_unread: int = 0
    sources: list[str] = field(default_factory=list)  # "ui" / "schedule"
    build_env: dict[str, str] = field(default_factory=dict)
    digest: str = ""

    @property
    def thin(self) -> bool:
        """The row exists but is too weak to read a silence off. Cruder than
        the completeness check on purpose, and it must never be promoted into
        authorizing a drop: nothing found in a weak row is not proof that
        nothing is there."""
        return (
            not self.has_evidence
            or self.clean_exits == 0
            or not self.shards_complete
            or self.lost_lines
            or self.jobs_ran_no_tests > 0
            or self.jobs_summary_unparsed > 0
            # An unreadable log zeroes the two counters above, so without this
            # the strongest health signals just go quiet.
            or self.logs_unreadable > 0
        )

    @property
    def has_evidence(self) -> bool:
        """A row with no functions says nothing about what the step covers, so
        it can never authorize a drop. Different from a row that simply lacks
        the changed function, which is the real signal."""
        return self.n_functions > 0

    @property
    def shards_complete(self) -> bool:
        """At least one build where every declared shard produced a recording.
        Not 'each shard passed at some point': shard boundaries move."""
        return (
            any(
                len(set(self.shards_seen.get(b, ()))) >= self.shards_expected.get(b, 0)
                for b in self.shards_expected
            )
            or not self.shards_expected
        )


@dataclass
class Row:
    key: str
    keyed: bool
    functions: dict[str, frozenset[str]]
    stamp: Stamp
    # The subset of `functions` that runs on import: module bodies and class
    # bodies, read off `CO_OPTIMIZED` at merge time against the source at the
    # recording commit. Stored rather than derived at load, because deriving it
    # needs the repo at that commit and `load` has no git.
    import_time: dict[str, frozenset[str]] = field(default_factory=dict)

    def contains(self, path: str, name: str) -> bool:
        """Entered this frame at all, import included. What the drop side reads
        under `PhaseMode.OFF`: a step that only imports a changed file can still
        break on it."""
        return name in self.functions.get(path, ())

    def contains_call(self, path: str, name: str) -> bool:
        """Entered this frame somewhere other than a module or class body.

        `<module>` is in every (row, file) pair there is, so `contains` alone
        says "this job imported the file". The add side always asks this; the
        drop side asks it under `PhaseMode.CARVED` and `STRICT`.

        Not the whole question: a function CALLED during import records exactly
        like one a test called, and source cannot tell them apart. Separating
        those needs a re-record.
        """
        return self.contains(path, name) and name not in self.import_time.get(path, ())


def digest_of(
    functions: dict[str, frozenset[str]],
    stamp: Stamp | None = None,
    import_time: dict[str, frozenset[str]] | None = None,
) -> str:
    """Sign the functions AND the stamp that judges them. Signing only the
    functions left every field droppability turns on unsigned, so deleting a
    health counter from a stored row still verified clean. The stamp's own
    digest is excluded, being written into the thing it signs."""
    h = hashlib.sha256()
    for path in sorted(functions):
        h.update(path.encode())
        h.update(b"\0")
        for name in sorted(functions[path]):
            h.update(name.encode())
            h.update(b"\0")
        h.update(b"\1")
    # Signed on its own, not folded into the walk above: an unsigned phase could
    # be flipped on disk and still verify, which is the hole signing exists to
    # close. `\2` keeps it from colliding with the functions encoding.
    if import_time:
        h.update(b"\2")
        for path in sorted(import_time):
            h.update(path.encode())
            h.update(b"\0")
            for name in sorted(import_time[path]):
                h.update(name.encode())
                h.update(b"\0")
            h.update(b"\1")
    if stamp is not None:
        fields = {k: v for k, v in asdict(stamp).items() if k != "digest"}
        h.update(json.dumps(fields, sort_keys=True, default=str).encode())
    return h.hexdigest()
