# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fold raw recordings from an instrumented build into one row per step.

The offline half: runs after a CI sweep, never on a PR. Its data model lives in
`model.py`.

Three set unions, over the processes in a job, the shards of a step, and builds
over time. Every record goes in, even from failed shards and killed processes,
because adding functions to a row can only make us keep more steps. Treating
partial data as complete is the unsafe move, so the real output is the stamp:
what the row was built from, and every way it might be less than it looks.

Three shapes were read off the data rather than assumed. The end marker is not
a terminator, since teardown keeps entering vLLM functions after the exit hook
runs, so read to EOF. The root comes from the root line and not the header,
which is written before the root is knowable. And shard structure comes from
the parallel index, never from the label.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..coverage.joblog import read_counts
from ..coverage.model import (
    ENV_ABSENT,
    MIN_RECORD_RATE,
    MIRROR_NOTE,
    RECORD_GLOB,
    TABLE_VERSION,
    Row,
    SourceIndex,
    Stamp,
    digest_of,
    read_process,
    row_key,
)

# Build env worth keeping on a recorded row, by name. Wholesale capture is not
# an option: a recording is a public artifact and the agent token lives in that
# mapping.
WORLD_ENV = (
    "NIGHTLY",
    "TORCH_NIGHTLY",
    "RUN_ALL",
    "CONTINUE_ON_FAILURE",
    "FNREC",
    "VLLM_CI_BRANCH",
)
JOB_STATE_PASSED = "passed"


def read_world(build_dir: Path, index: dict) -> tuple[str, str, dict[str, str], bool]:
    """The build's identity: which pipeline, triggered how, under what env.

    The env is an allowlist and not the whole mapping, because a recording is a
    public artifact. A listed key that was not set is recorded as absent rather
    than omitted, since "no NIGHTLY" is what separates a hand-triggered sweep
    from a production one and has to survive as data.

    The fourth value says whether the world file was readable. Otherwise an
    empty env means two things, "every variable was unset" and "we never found
    out", and only the first is data.
    """
    slug = index.get("pipeline") or ""
    try:
        build_json = json.loads((build_dir / "build.json").read_text())
    except (OSError, ValueError):
        return slug, "", {}, False
    env = build_json.get("env") or {}
    return (
        slug,
        build_json.get("source") or "",
        {k: env.get(k, ENV_ABSENT) for k in WORLD_ENV},
        True,
    )


@dataclass
class BuildCensus:
    """What one build's index promised, against what its jobs delivered.

    Not a `Stamp` field and never written to the table: a stamp is per row, and
    "282 jobs recorded nothing" belongs to the rows that do not exist, so any
    per-row encoding has to pick a row to blame. This lives for one merge, gets
    printed, and sets an exit code.

    The three causes mean different things: no directory is a delivery failure,
    files that are not recordings is usually a failed install, and an
    unresolvable root is our own parser.
    """

    build: str = ""
    n_jobs: int = 0
    attempted: int = 0
    recorded: int = 0
    no_dir: list[str] = field(default_factory=list)
    no_record_files: list[str] = field(default_factory=list)
    no_resolvable_root: list[str] = field(default_factory=list)
    delivery: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.recorded / self.attempted if self.attempted else 0.0

    @property
    def collapsed(self) -> bool:
        return not self.attempted or self.rate < MIN_RECORD_RATE

    def summary(self) -> str:
        causes = (
            f"no dir {len(self.no_dir)} \u00b7 "
            f"no {RECORD_GLOB} {len(self.no_record_files)} \u00b7 "
            f"no root {len(self.no_resolvable_root)}"
        )
        return (
            f"{self.build}: {self.recorded}/{self.attempted} started jobs recorded "
            f"({self.rate:.2f})   {causes}"
        )


def _attempted(job: dict) -> bool:
    """Buildkite gives a job that never ran a null `started_at`.

    A blocked or expired job cannot have recorded, so it does not belong in the
    denominator. An index without the field counts the job: a larger denominator
    fails toward noticing.
    """
    return "started_at" not in job or job["started_at"] is not None


def merge_build(
    build_dir: Path,
    repo: Path,
    verbose: bool = False,
    census: BuildCensus | None = None,
) -> dict[str, Row]:
    """Every recorded job in one build, folded into rows.

    `census` is an optional out-parameter so existing callers stay untouched;
    when omitted a local one is built and printed.
    """
    index = json.loads((build_dir / "index.json").read_text())
    commit = index["commit"]
    build = str(index["build"])
    source = SourceIndex(repo, commit)
    # Before anything reads a path. Without this an unreachable commit makes
    # every file look absent at that commit, and the build merges to nothing.
    source.require_commit()
    slug, trigger, world_env, world_read = read_world(build_dir, index)

    # Expected shard counts come from the pipeline, not from what recorded, so a
    # step whose shard never uploaded is visibly short rather than silently thin.
    expected: dict[str, int] = {}
    for job in index["jobs"]:
        key, _ = row_key(job)
        if job.get("parallel_total"):
            expected[key] = max(expected.get(key, 0), job["parallel_total"])

    accumulated: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    stamps: dict[str, Stamp] = {}
    keyed_flag: dict[str, bool] = {}

    census = census if census is not None else BuildCensus()
    census.build = build_dir.name

    for job in index["jobs"]:
        census.n_jobs += 1
        census.attempted += int(_attempted(job))
        delivery = job.get("artifact") or "unknown"
        census.delivery[delivery] = census.delivery.get(delivery, 0) + 1

        fnrec_dir = build_dir / "jobs" / job["job"] / "fnrec"
        if not fnrec_dir.is_dir():
            census.no_dir.append(job["job"])
            continue
        files = sorted(fnrec_dir.glob(RECORD_GLOB))
        if not files:
            # Delivered something, but no recording. `install.err` with no
            # fn.*.txt beside it is the live shape: the installer failed.
            census.no_record_files.append(job["job"])
            continue
        processes = [read_process(p) for p in files]
        processes = [p for p in processes if p is not None]
        if not processes:
            # Files, but none names a root, so nothing in them can be placed in
            # the tree. Our parser, not the build.
            census.no_resolvable_root.append(job["job"])
            continue
        census.recorded += 1

        key, keyed = row_key(job)
        keyed_flag[key] = keyed
        stamp = stamps.setdefault(key, Stamp())
        if job["job"] not in stamp.jobs:
            stamp.jobs.append(job["job"])
        if build not in stamp.builds:
            stamp.builds.append(build)
        if commit not in stamp.commits:
            stamp.commits.append(commit)
        # A job Buildkite did not pass stopped its command list early, so the
        # recording is a prefix of the step's work and cannot authorize a drop.
        # Both fields, because either one being absent or newly spelled would
        # otherwise promote a partial recording to a healthy one on its own.
        exit_status = job.get("exit_status")
        if (
            job.get("state") != JOB_STATE_PASSED
            or (exit_status is not None and exit_status != 0)
        ) and job["job"] not in stamp.failed_jobs:
            stamp.failed_jobs.append(job["job"])
        stamp.pipeline_slug = _union_slug(stamp.pipeline_slug, slug)
        if trigger and trigger not in stamp.sources:
            stamp.sources.append(trigger)
        if not world_read:
            stamp.worlds_unread += 1
        for name, value in world_env.items():
            stamp.build_env.setdefault(name, value)
        if job.get("parallel_total"):
            # The index, not a tally. Buildkite returns every attempt of a
            # retried job, so counting jobs lets one shard stand in for another.
            seen = stamp.shards_seen.setdefault(build, [])
            shard = job.get("parallel_index")
            if shard is not None and shard not in seen:
                seen.append(shard)
            stamp.shards_expected[build] = expected.get(key, job["parallel_total"])

        log = read_counts(build_dir / "jobs" / job["job"] / "job.log.gz")
        if log.unreadable:
            stamp.logs_unreadable += 1
        else:
            counts = log.counts
            stamp.tests_executed += counts.executed
            stamp.tests_skipped += counts.skipped
            stamp.tests_collected += counts.collected
            stamp.tests_passed += counts.passed
            stamp.tests_failed += counts.failed
            stamp.tests_deselected += counts.deselected
            stamp.tests_xfailed += counts.xfailed
            stamp.tests_xpassed += counts.xpassed
            stamp.tests_errors += counts.errors
            stamp.pytest_invocations += counts.invocations
            stamp.jobs_ran_no_tests += int(counts.ran_nothing)
            stamp.jobs_summary_unparsed += int(counts.summary_unparsed)

        for process in processes:
            stamp.processes += 1
            stamp.clean_exits += int(process.clean_exit)
            stamp.lost_lines |= process.lost_lines
            stamp.outside_root_lines += process.outside_root
            stamp.malformed_lines += process.malformed
            stamp.process_errors += process.errors
            if process.py and process.py not in stamp.interpreters:
                stamp.interpreters.append(process.py)
            for path, names in process.functions.items():
                accumulated[key][path] |= names

        if verbose:
            print(f"  {key}: {len(accumulated[key])} files")

    rows: dict[str, Row] = {}
    # Keyed on stamps and not on what accumulated, because a step can record
    # cleanly while entering no vLLM function at all. That row is complete and
    # empty, the most dangerous shape there is: "contains none of the changed
    # functions" would drop the step for every diff forever. Emitting it and
    # letting the reader treat an empty row as no evidence is safe on purpose.
    for key, stamp in stamps.items():
        by_file = accumulated.get(key, {})
        functions: dict[str, frozenset[str]] = {}
        import_time: dict[str, frozenset[str]] = {}
        for path, names in by_file.items():
            compiled = source.names(path)
            if compiled is None:
                # Either absent at this commit, so no diff can name it, or it
                # will not compile, so we cannot judge it.
                if not source.exists(path):
                    stamp.dropped_absent_files += 1
                    continue
                stamp.unfaithful_files.append(path)
            elif not names <= compiled:
                stamp.unfaithful_files.append(path)
            functions[path] = frozenset(names)
            # Reuses the source the faithfulness check just compiled, so this
            # is free. An unreadable file yields nothing, leaving its names
            # unclassified and so treated as calls, which only over-selects.
            at_import = source.import_time_names(path)
            if at_import:
                shared = frozenset(names) & at_import
                if shared:
                    import_time[path] = shared

        stamp.unfaithful_files.sort()
        stamp.n_files = len(functions)
        stamp.n_functions = sum(len(v) for v in functions.values())
        stamp.n_import_time = sum(len(v) for v in import_time.values())
        stamp.digest = digest_of(functions, stamp, import_time)
        rows[key] = Row(
            key=key,
            keyed=keyed_flag[key],
            functions=functions,
            stamp=stamp,
            import_time=import_time,
        )
    print(census.summary())
    if census.delivery:
        shapes = " \u00b7 ".join(
            f"{name} {count}" for name, count in sorted(census.delivery.items())
        )
        print(f"    delivery: {shapes}")

    return rows


def _union_env(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    """Keep both values when two builds disagree about a variable.

    Silently preferring one would hide exactly the thing the field exists to
    show: that the table blends worlds.
    """
    out = dict(a)
    for key, value in b.items():
        if key in out and out[key] != value:
            out[key] = "|".join(sorted({out[key], value}))
        else:
            out.setdefault(key, value)
    return out


def _union_slug(a: str, b: str) -> str:
    """Keep both slugs when they disagree, same reason as `_union_env`: this
    field exists to make a blended table visible, and first-wins hides it."""
    if not a:
        return b
    if not b or a == b:
        return a
    return "|".join(sorted(set(a.split("|")) | {b}))


def _union_shards(
    a: dict[str, list[int]], b: dict[str, list[int]]
) -> dict[str, list[int]]:
    """Shard indexes per build, unioned. A dict merge would let the second row
    replace the first's indexes for a build both rows saw, dropping shards from
    a row that is then read as complete."""
    out = {k: sorted(set(v)) for k, v in a.items()}
    for key, seen in b.items():
        out[key] = sorted(set(out.get(key, [])) | set(seen))
    return out


def union_rows(left: Row, right: Row) -> Row:
    """Combine the same step's rows from two builds. Union, never replace."""
    if left.key != right.key:
        raise ValueError(
            f"refusing to union different steps: {left.key} vs {right.key}"
        )
    functions: dict[str, frozenset[str]] = dict(left.functions)
    for path, names in right.functions.items():
        functions[path] = functions.get(path, frozenset()) | names
    # Union, not intersect: whether a name runs at import is a property of the
    # source, so builds at one commit agree; across commits, a name that was
    # ever a module body is one the add side should not add on.
    import_time: dict[str, frozenset[str]] = dict(left.import_time)
    for path, names in right.import_time.items():
        import_time[path] = import_time.get(path, frozenset()) | names

    a, b = left.stamp, right.stamp
    stamp = Stamp(
        jobs=sorted(set(a.jobs) | set(b.jobs)),
        builds=sorted(set(a.builds) | set(b.builds)),
        commits=sorted(set(a.commits) | set(b.commits)),
        interpreters=sorted(set(a.interpreters) | set(b.interpreters)),
        processes=a.processes + b.processes,
        clean_exits=a.clean_exits + b.clean_exits,
        failed_jobs=sorted(set(a.failed_jobs) | set(b.failed_jobs)),
        shards_seen=_union_shards(a.shards_seen, b.shards_seen),
        shards_expected={**a.shards_expected, **b.shards_expected},
        lost_lines=a.lost_lines or b.lost_lines,
        outside_root_lines=a.outside_root_lines + b.outside_root_lines,
        malformed_lines=a.malformed_lines + b.malformed_lines,
        process_errors=a.process_errors + b.process_errors,
        unfaithful_files=sorted(set(a.unfaithful_files) | set(b.unfaithful_files)),
        dropped_absent_files=a.dropped_absent_files + b.dropped_absent_files,
        tests_executed=a.tests_executed + b.tests_executed,
        tests_skipped=a.tests_skipped + b.tests_skipped,
        tests_collected=a.tests_collected + b.tests_collected,
        tests_passed=a.tests_passed + b.tests_passed,
        tests_failed=a.tests_failed + b.tests_failed,
        tests_deselected=a.tests_deselected + b.tests_deselected,
        tests_xfailed=a.tests_xfailed + b.tests_xfailed,
        tests_xpassed=a.tests_xpassed + b.tests_xpassed,
        tests_errors=a.tests_errors + b.tests_errors,
        pytest_invocations=a.pytest_invocations + b.pytest_invocations,
        jobs_ran_no_tests=a.jobs_ran_no_tests + b.jobs_ran_no_tests,
        jobs_summary_unparsed=a.jobs_summary_unparsed + b.jobs_summary_unparsed,
        logs_unreadable=a.logs_unreadable + b.logs_unreadable,
        pipeline_slug=_union_slug(a.pipeline_slug, b.pipeline_slug),
        worlds_unread=a.worlds_unread + b.worlds_unread,
        sources=sorted(set(a.sources) | set(b.sources)),
        # Disagreeing builds keep both values, so a blended table is visible
        # rather than silently resolved in favour of whichever merged last.
        build_env=_union_env(a.build_env, b.build_env),
    )
    stamp.n_files = len(functions)
    stamp.n_functions = sum(len(v) for v in functions.values())
    stamp.n_import_time = sum(len(v) for v in import_time.values())
    stamp.digest = digest_of(functions, stamp, import_time)
    return Row(
        key=left.key,
        keyed=left.keyed or right.keyed,
        functions=functions,
        stamp=stamp,
        import_time=import_time,
    )


def merge_builds(
    build_dirs: list[Path],
    repo: Path,
    verbose: bool = False,
    censuses: list[BuildCensus] | None = None,
) -> dict[str, Row]:
    table: dict[str, Row] = {}
    for build_dir in build_dirs:
        if verbose:
            print(f"reading {build_dir.name}")
        census = BuildCensus()
        rows = merge_build(build_dir, repo, verbose=verbose, census=census)
        if censuses is not None:
            censuses.append(census)
        for key, row in rows.items():
            table[key] = union_rows(table[key], row) if key in table else row
    return table


def write_table(rows: dict[str, Row], out: Path) -> None:
    payload = {
        "version": TABLE_VERSION,
        "note": MIRROR_NOTE,
        "rows": {
            key: {
                "keyed": row.keyed,
                "stamp": asdict(row.stamp),
                "functions": {p: sorted(n) for p, n in sorted(row.functions.items())},
                "import_time": {
                    p: sorted(n) for p, n in sorted(row.import_time.items())
                },
            }
            for key, row in sorted(rows.items())
        },
    }
    text = json.dumps(payload, indent=1)
    if out.suffix == ".gz":
        out.write_bytes(gzip.compress(text.encode()))
    else:
        out.write_text(text)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo", type=Path)
    ap.add_argument("builds", type=Path, nargs="+", help="sweep build directories")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="merge even when a build delivered almost nothing",
    )
    args = ap.parse_args()

    censuses: list[BuildCensus] = []
    rows = merge_builds(args.builds, args.repo, verbose=args.verbose, censuses=censuses)

    collapsed = [c for c in censuses if c.collapsed]
    if collapsed and not args.allow_partial:
        # Refused BEFORE writing. A table merged from a build that lost its
        # recordings is exactly the artifact that gets committed and then quoted.
        for census in collapsed:
            print(f"REFUSING: {census.summary()}")
        print(
            "\nA build this thin is a delivery failure, not a build that ran no "
            "vLLM code.\nRe-run with --allow-partial to merge it anyway."
        )
        raise SystemExit(2)

    write_table(rows, args.out)

    keyless = sum(1 for r in rows.values() if not r.keyed)
    lossy = sum(1 for r in rows.values() if r.stamp.lost_lines)
    short = [k for k, r in rows.items() if not r.stamp.shards_complete]
    unfaithful = sorted({p for r in rows.values() for p in r.stamp.unfaithful_files})
    recorded = sum(c.recorded for c in censuses)
    attempted = sum(c.attempted for c in censuses)
    print(f"\n{len(rows)} rows ({keyless} keyless)")
    print(f"  functions: {sum(r.stamp.n_functions for r in rows.values())}")
    print(f"  files:     {len({p for r in rows.values() for p in r.functions})}")
    print(f"  builds:    {len(censuses)}   jobs recorded {recorded}/{attempted}")
    print(f"  rows with lost lines:   {lossy}")
    print(f"  rows short on shards:   {len(short)} {short[:5]}")
    stamps = [r.stamp for r in rows.values()]
    print(f"  rows with a failed job: {sum(1 for st in stamps if st.failed_jobs)}")
    print(f"  jobs that ran no tests: {sum(st.jobs_ran_no_tests for st in stamps)}")
    # The two "we did not find out" counters. Not-found-out must never read as
    # found-nothing.
    print(f"  logs unreadable:        {sum(st.logs_unreadable for st in stamps)}")
    print(f"  worlds unread:          {sum(st.worlds_unread for st in stamps)}")
    # Our parser failing, on a separate line from the step running no tests.
    print(f"  unparsed test summary:  {sum(st.jobs_summary_unparsed for st in stamps)}")
    print(f"  dropped absent files:   {sum(st.dropped_absent_files for st in stamps)}")
    print(f"  unfaithful files:       {len(unfaithful)}")
    for path in unfaithful[:20]:
        print(f"      {path}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
