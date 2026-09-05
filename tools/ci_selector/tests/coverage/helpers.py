# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builders for synthetic coverage fixtures: a throwaway git repo, a recorded
process file, a merged table, all shaped like the real ones.

A plain module and not a conftest, which is for fixtures. Importing a conftest
as a module works until the tree moves."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

ROOT = "/usr/local/lib/python3.12/dist-packages/vllm/"

MODULE_SOURCE = """\
CONSTANT = 1


def plain():
    return CONSTANT


class Holder:
    def method(self):
        return 2
"""


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    def write(self, path: str, text: str) -> None:
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text)

    def commit(self, message: str = "c") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.head()

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()


def process_file(
    path: Path,
    entries: list[tuple[str, str]],
    *,
    root: str = ROOT,
    header_root: str | None = None,
    counter: int | None = None,
    clean_exit: bool = True,
    after_end: list[tuple[str, str]] | None = None,
    py: str = "3.12.13",
    job: str = "job-1",
) -> None:
    """Write one fnrec process file in the recorder's real on-disk shape."""
    shown = ROOT if header_root is None else header_root
    lines = [
        "\t".join(
            [
                "#start",
                "pid=1",
                f"root={shown}",
                f"py={py}",
                f"BUILDKITE_JOB_ID={job}",
            ]
        ),
        f"#root\t{root}\tt=1",
    ]
    lines += [f"{root}{rel}\t{name}\t1" for rel, name in entries]
    if clean_exit:
        total = len(entries) if counter is None else counter
        lines.append(f"#end\troot={total}\tother=0\terrors=0\tlast_error=\tt=2")
    elif counter is not None:
        lines.append(f"#stat\troot={counter}\tother=0\terrors=0\tlast_error=\tt=2")
    lines += [f"{root}{rel}\t{name}\t1" for rel, name in (after_end or [])]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# A job log that reads healthy: tests collected and executed, one summary the
# parser understands. The default, because a job with no log at all is now
# thin, and without this every fixture in the suite would be.
HEALTHY_LOG = "collected 3 items\n===== 3 passed in 1.20s =====\n"

# The sentinel is what lets a caller ask for no log file at all, which is a
# different thing from asking for the default one.
_DEFAULT_LOG = object()


class Build:
    """A sweep build directory: index.json plus jobs/<id>/fnrec/."""

    def __init__(self, root: Path, number: str, commit: str, pipeline: str = "ci"):
        self.root = root
        self.number = number
        self.commit = commit
        self.pipeline = pipeline
        self.jobs: list[dict] = []
        root.mkdir(parents=True, exist_ok=True)

    def job(
        self,
        job_id: str,
        *,
        step_key: str | None = None,
        label: str = "",
        parallel_index: int | None = None,
        parallel_total: int | None = None,
        recorded: bool = True,
        state: str = "passed",
        exit_status: int | None = 0,
        log: object = _DEFAULT_LOG,
        started: bool = True,
        artifact: str | None = None,
    ) -> Path:
        self.jobs.append(
            {
                "job": job_id,
                "step_key": step_key,
                "label": label,
                "state": state,
                "exit_status": exit_status,
                "parallel_index": parallel_index,
                "parallel_total": parallel_total,
                "build": self.number,
                "commit": self.commit,
                # `started_at` is how a blocked job is told from one that ran and
                # recorded nothing, which is what keeps it out of the denominator.
                "started_at": "2026-08-26T00:00:00Z" if started else None,
                "finished_at": "2026-08-26T00:10:00Z" if started else None,
                "artifact": artifact,
                "n_files": 1 if recorded else 0,
            }
        )
        d = self.root / "jobs" / job_id / "fnrec"
        # Only when something was delivered. Creating it unconditionally hid the
        # "no directory at all" branch, which is the one 253 real jobs took.
        if recorded:
            d.mkdir(parents=True, exist_ok=True)
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
        body = HEALTHY_LOG if log is _DEFAULT_LOG else log
        if body is not None:
            (d.parent / "job.log.gz").write_bytes(gzip.compress(body.encode()))
        return d

    def finish(self) -> Path:
        (self.root / "index.json").write_text(
            json.dumps(
                {
                    "build": self.number,
                    "commit": self.commit,
                    # read_world reads this; without it every fixture row stamps
                    # an empty slug while every real row carries "ci".
                    "pipeline": self.pipeline,
                    "n_jobs": len(self.jobs),
                    "n_with_record": sum(1 for j in self.jobs if j["n_files"]),
                    "jobs": self.jobs,
                }
            )
        )
        return self.root


def make_table(
    tmp_path: Path,
    repo,
    jobs: dict,
    *,
    build_no: str = "1",
    thin_keys: set | None = None,
):
    """A loaded table from {row key: [(relative file, qualname), ...]}.

    Saves every test from restating the merge; the merge itself is covered in
    test_merge.py.
    """
    from ci_selector.coverage.table import load
    from ci_selector.scripts.build import merge_build, write_table

    build = Build(tmp_path / f"sweep-{build_no}" / "b", build_no, repo.head())
    for index, (key, entries) in enumerate(jobs.items()):
        process_file(
            build.job(f"j{index}", step_key=key) / "fn.a.txt",
            entries,
            # No clean exit anywhere is one of the things that makes a row thin.
            clean_exit=key not in (thin_keys or set()),
        )
    out = tmp_path / f"table-{build_no}.json"
    write_table(merge_build(build.finish(), repo.root), out)
    return load(out)
