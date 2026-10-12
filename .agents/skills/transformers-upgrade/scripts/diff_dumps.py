# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff two dump_configs.py or dump_model.py outputs, per repo.

python diff_dumps.py before.jsonl after.jsonl
"""

import json
import sys
from collections import Counter

import regex as re

IGNORED = {"transformers_version", "_name_or_path", "architectures", "auto_map"}


def load(path: str) -> dict[str, dict]:
    with open(path) as f:
        return {record["repo"]: record for record in map(json.loads, f)}


def diff(before: dict, after: dict, prefix: str = ""):
    for key in sorted(set(before) | set(after)):
        if key in IGNORED:
            continue
        old, new = before.get(key, "<MISSING>"), after.get(key, "<MISSING>")
        if isinstance(old, dict) and isinstance(new, dict) and key != "rope_parameters":
            yield from diff(old, new, f"{prefix}{key}.")
        elif old != new:
            yield f"{prefix}{key}", json.dumps(old), json.dumps(new)


def collapse(diffs) -> Counter:
    """Merge diffs that differ only by layer index, e.g. `layers.*.scaling`."""
    return Counter(
        (re.sub(r"\.\d+\.", ".*.", path), old, new) for path, old, new in diffs
    )


before, after = load(sys.argv[1]), load(sys.argv[2])
for repo, old in before.items():
    new = after.get(repo, {"error": "missing from after"})
    if "error" in old or "error" in new:
        print(f"== {repo}\n   before: {old.get('error', 'ok')}")
        print(f"   after:  {new.get('error', 'ok')}")
        continue
    diffs = collapse(diff(old["cfg"], new["cfg"]))
    print(f"== {repo}: {old['cls']} -> {new['cls']} ({sum(diffs.values())} diffs)")
    for (path, old_value, new_value), count in diffs.items():
        times = f" (x{count})" if count > 1 else ""
        print(f"   {path}{times}: before={old_value} after={new_value}")
