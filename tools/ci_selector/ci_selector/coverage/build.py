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
from dataclasses import asdict
from pathlib import Path

from .joblog import read_counts
from .model import (
    ENV_ABSENT,
    MIRROR_NOTE,
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


def read_world(build_dir: Path, index: dict) -> tuple[str, str, dict[str, str]]:
    """The build's identity: which pipeline, triggered how, under what env.

    The env is an allowlist and not the whole mapping, because a recording is a
    public artifact. A listed key that was not set is recorded as absent rather
    than omitted, since "no NIGHTLY" is what separates a hand-triggered sweep
    from a production one and has to survive as data.
    """
    slug = index.get("pipeline") or ""
    try:
        build_json = json.loads((build_dir / "build.json").read_text())
    except (OSError, ValueError):
        return slug, "", {}
    env = build_json.get("env") or {}
    return (
        slug,
        build_json.get("source") or "",
        {k: env.get(k, ENV_ABSENT) for k in WORLD_ENV},
    )


def merge_build(build_dir: Path, repo: Path, verbose: bool = False) -> dict[str, Row]:
    """Every recorded job in one build, folded into rows."""
    index = json.loads((build_dir / "index.json").read_text())
    commit = index["commit"]
    build = str(index["build"])
    source = SourceIndex(repo, commit)
    slug, trigger, world_env = read_world(build_dir, index)

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

    for job in index["jobs"]:
        fnrec_dir = build_dir / "jobs" / job["job"] / "fnrec"
        if not fnrec_dir.is_dir():
            continue
        processes = [read_process(p) for p in sorted(fnrec_dir.glob("fn.*.txt"))]
        processes = [p for p in processes if p is not None]
        if not processes:
            continue

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
        if job.get("state") != JOB_STATE_PASSED and job["job"] not in stamp.failed_jobs:
            stamp.failed_jobs.append(job["job"])
        stamp.pipeline_slug = stamp.pipeline_slug or slug
        if trigger and trigger not in stamp.sources:
            stamp.sources.append(trigger)
        for name, value in world_env.items():
            stamp.build_env.setdefault(name, value)
        if job.get("parallel_total"):
            stamp.shards_seen[build] = stamp.shards_seen.get(build, 0) + 1
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

        stamp.unfaithful_files.sort()
        stamp.n_files = len(functions)
        stamp.n_functions = sum(len(v) for v in functions.values())
        stamp.digest = digest_of(functions, stamp)
        rows[key] = Row(
            key=key, keyed=keyed_flag[key], functions=functions, stamp=stamp
        )
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


def union_rows(left: Row, right: Row) -> Row:
    """Combine the same step's rows from two builds. Union, never replace."""
    if left.key != right.key:
        raise ValueError(
            f"refusing to union different steps: {left.key} vs {right.key}"
        )
    functions: dict[str, frozenset[str]] = dict(left.functions)
    for path, names in right.functions.items():
        functions[path] = functions.get(path, frozenset()) | names

    a, b = left.stamp, right.stamp
    stamp = Stamp(
        jobs=sorted(set(a.jobs) | set(b.jobs)),
        builds=sorted(set(a.builds) | set(b.builds)),
        commits=sorted(set(a.commits) | set(b.commits)),
        interpreters=sorted(set(a.interpreters) | set(b.interpreters)),
        processes=a.processes + b.processes,
        clean_exits=a.clean_exits + b.clean_exits,
        failed_jobs=sorted(set(a.failed_jobs) | set(b.failed_jobs)),
        shards_seen={**a.shards_seen, **b.shards_seen},
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
        logs_unreadable=a.logs_unreadable + b.logs_unreadable,
        pipeline_slug=a.pipeline_slug or b.pipeline_slug,
        sources=sorted(set(a.sources) | set(b.sources)),
        # Disagreeing builds keep both values, so a blended table is visible
        # rather than silently resolved in favour of whichever merged last.
        build_env=_union_env(a.build_env, b.build_env),
    )
    stamp.n_files = len(functions)
    stamp.n_functions = sum(len(v) for v in functions.values())
    stamp.digest = digest_of(functions, stamp)
    return Row(
        key=left.key, keyed=left.keyed or right.keyed, functions=functions, stamp=stamp
    )


def merge_builds(
    build_dirs: list[Path], repo: Path, verbose: bool = False
) -> dict[str, Row]:
    table: dict[str, Row] = {}
    for build_dir in build_dirs:
        if verbose:
            print(f"reading {build_dir.name}")
        for key, row in merge_build(build_dir, repo, verbose=verbose).items():
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
    args = ap.parse_args()

    rows = merge_builds(args.builds, args.repo, verbose=args.verbose)
    write_table(rows, args.out)

    keyless = sum(1 for r in rows.values() if not r.keyed)
    lossy = sum(1 for r in rows.values() if r.stamp.lost_lines)
    short = [k for k, r in rows.items() if not r.stamp.shards_complete]
    unfaithful = sorted({p for r in rows.values() for p in r.stamp.unfaithful_files})
    print(f"{len(rows)} rows ({keyless} keyless)")
    print(f"  functions: {sum(r.stamp.n_functions for r in rows.values())}")
    print(f"  files:     {len({p for r in rows.values() for p in r.functions})}")
    print(f"  rows with lost lines:   {lossy}")
    print(f"  rows short on shards:   {len(short)} {short[:5]}")
    print(f"  unfaithful files:       {len(unfaithful)}")
    for path in unfaithful[:20]:
        print(f"      {path}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
