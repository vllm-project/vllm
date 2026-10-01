# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that every `source_file_dependencies` path in Buildkite jobs exists.

A job runs when a change touches one of its `source_file_dependencies`. A path
that no longer matches a tracked file can never be touched, so after a rename or
delete the job silently stops triggering on the code it was written to guard.
This walks the directories the pipeline generator reads and fails on any entry
that is not a prefix of a tracked file. A leading `!` (an exclusion) is ignored.

Paths already dead when the check was added are listed in
`source_file_dependencies_allowlist.txt`, which is only meant to shrink.

Usage:
    python tools/pre_commit/check_source_file_dependencies.py
"""

import sys
from pathlib import Path

import yaml
from check_buildkite_step_keys import JOB_GLOB, job_dirs
from check_label_rules import tracked_files

ALLOWLIST_PATH = (
    Path(__file__).resolve().parent / "source_file_dependencies_allowlist.txt"
)


def dead_paths(node: object, tracked: list[str]) -> list[str]:
    """Every `source_file_dependencies` entry under `node` matching no tracked file."""
    if isinstance(node, list):
        return [p for item in node for p in dead_paths(item, tracked)]
    if not isinstance(node, dict):
        return []
    dead: list[str] = []
    for key, value in node.items():
        if key == "source_file_dependencies":
            for path in value:
                path = path.removeprefix("!")
                if not any(f.startswith(path) for f in tracked):
                    dead.append(path)
        else:
            dead += dead_paths(value, tracked)
    return dead


def main() -> int:
    tracked = tracked_files()
    allowed = {
        line.split("#", 1)[0].strip()
        for line in ALLOWLIST_PATH.read_text().splitlines()
    }
    files = sorted({p for d in job_dirs() for p in d.rglob(JOB_GLOB)})
    if not files:
        raise SystemExit("no pipeline files under the declared job_dirs")
    found = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            found += [(path, p) for p in dead_paths(yaml.safe_load(f), tracked)]
    found = [(path, p) for path, p in found if p not in allowed]
    if not found:
        return 0
    print(
        f"{len(found)} source_file_dependencies path(s) match no file:\n",
        file=sys.stderr,
    )
    for path, p in found:
        print(f"  {path}: {p}", file=sys.stderr)
    print(
        "\nA job never triggers on a path that does not exist. Fix the path in"
        "\nthe job's `source_file_dependencies`, or delete it if the code is gone.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
