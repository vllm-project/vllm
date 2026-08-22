# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builders for synthetic coverage fixtures: a throwaway git repo, a recorded
process file, a merged table, all shaped like the real ones.

A plain module and not a conftest, which is for fixtures. Importing a conftest
as a module works until the tree moves."""

from __future__ import annotations

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


class Build:
    """A sweep build directory: index.json plus jobs/<id>/fnrec/."""

    def __init__(self, root: Path, number: str, commit: str):
        self.root = root
        self.number = number
        self.commit = commit
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
    ) -> Path:
        self.jobs.append(
            {
                "job": job_id,
                "step_key": step_key,
                "label": label,
                "state": "passed",
                "parallel_index": parallel_index,
                "parallel_total": parallel_total,
                "build": self.number,
                "commit": self.commit,
                "n_files": 1 if recorded else 0,
            }
        )
        d = self.root / "jobs" / job_id / "fnrec"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def finish(self) -> Path:
        (self.root / "index.json").write_text(
            json.dumps(
                {
                    "build": self.number,
                    "commit": self.commit,
                    "n_jobs": len(self.jobs),
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
    from ci_selector.coverage.build import merge_build, write_table
    from ci_selector.coverage.table import load

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
