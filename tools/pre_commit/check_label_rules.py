# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that auto-labeling rules still match real files.

Mergify silently ignores a condition that can never match, so a file-path
condition keeps passing review long after the file it names has moved or been
deleted. This walks every `files=` / `files~=` condition in `.github/mergify.yml`
and fails if one matches nothing in the tree.

Usage:
    python tools/pre_commit/check_label_rules.py
"""

import re
import subprocess
import sys

import yaml

MERGIFY = ".github/mergify.yml"


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.split("\n") if line]


def file_conditions(node, found: list[str]) -> list[str]:
    """Collect every string condition nested under a rule's `conditions`."""
    if isinstance(node, dict):
        for value in node.values():
            file_conditions(value, found)
    elif isinstance(node, list):
        for value in node:
            file_conditions(value, found)
    elif isinstance(node, str):
        found.append(node)
    return found


def check(files: list[str]) -> list[tuple[str, str, str]]:
    with open(MERGIFY) as f:
        rules = yaml.safe_load(f)["pull_request_rules"]

    dead = []
    for rule in rules:
        for cond in file_conditions(rule.get("conditions", []), []):
            cond = cond.strip()
            # Negated conditions (label-tpu-remove) are expected to match nothing.
            if cond.startswith("-files"):
                continue
            if cond.startswith("files~="):
                pattern = cond[len("files~=") :]
                try:
                    regex = re.compile(pattern)
                except re.error as exc:
                    dead.append((rule["name"], cond, f"invalid regex: {exc}"))
                    continue
                matched = any(regex.search(f) for f in files)
            elif cond.startswith("files="):
                matched = cond[len("files=") :] in files
            else:
                continue
            if not matched:
                dead.append((rule["name"], cond, "matches no tracked file"))
    return dead


def main() -> int:
    dead = check(tracked_files())
    if not dead:
        return 0
    print(f"{len(dead)} label condition(s) match nothing:\n", file=sys.stderr)
    for name, cond, why in dead:
        print(f"  [{name}] {cond}\n      {why}", file=sys.stderr)
    print(
        "\nRepoint the condition at the file's new location, or remove it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
