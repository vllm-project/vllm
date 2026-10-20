# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture bounded copies of experiment evidence without touching its producers."""

import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DESTINATION = Path(__file__).resolve().parent
RESULTS = REPO / "benchmarks/programmable_kv_cache/results"
PHOENIX = dt.timezone(dt.timedelta(hours=-7), "America/Phoenix")


def timestamp():
    return dt.datetime.now(PHOENIX).isoformat(timespec="seconds")


def command(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def main():
    manifest_path = DESTINATION / "evidence-manifest.json"
    if manifest_path.exists():
        raise SystemExit("This snapshot already exists; create a new directory.")
    began = time.time()
    manifest = {
        "snapshot_started": timestamp(),
        "timezone": "America/Phoenix (UTC-07:00)",
        "repository": str(REPO),
        "git_revision": command("git", "rev-parse", "HEAD"),
        "git_branch": command("git", "branch", "--show-current"),
        "git_status_at_start": command("git", "status", "--short"),
        "copy_policy": (
            "Capture each regular file's observed byte length; SHA256 the saved copy. "
            "No source files or experiment processes are modified. Active files can "
            "represent different instants within the capture interval."
        ),
        "selection": [
            "acceptance-2026-10-19: all regular files present at enumeration",
            "rollout-2026-10-19: all regular files present at enumeration",
            "rollout-2026-10-18: baseline, priority-quota, "
            "quota-wait120-aging30 and top-level metadata",
            "local design, OpenSpec, runtime, benchmark and related test source",
        ],
        "excluded": [
            "Model weights, datasets, compiled artifacts and virtual environments "
            "are identified by saved metadata, not duplicated.",
            "October 18 large profiler/diagnostic subdirectories remain in the "
            "original results tree; they are outside this report's accepted timing "
            "evidence.",
            "Files created after enumeration are outside this frozen snapshot.",
        ],
        "files": [],
    }
    selections = {}

    def include(path, target):
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    selections[child] = target / child.relative_to(path)
        elif path.is_file() and not path.is_symlink():
            selections[path] = target

    for name in ["acceptance-2026-10-19", "rollout-2026-10-19"]:
        include(RESULTS / name, Path("raw/results") / name)
    older = RESULTS / "rollout-2026-10-18"
    for name in ["baseline", "priority-quota", "quota-wait120-aging30"]:
        include(older / name, Path("raw/results/rollout-2026-10-18") / name)
    for child in older.iterdir():
        if child.is_file():
            include(child, Path("raw/results/rollout-2026-10-18") / child.name)
    source_paths = [
        "docs/programmable-kvc",
        "openspec/changes/programmable-kv-cache",
        "vllm/v1/kv_hints",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/config/kv_control.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/kv_connector",
        "vllm/v1/kv_offload",
        "tests/v1/test_kv_hint_control.py",
        "tests/v1/core/test_prefix_caching.py",
        "tests/v1/core/test_scheduler.py",
    ]
    source_paths += [
        str(p.relative_to(REPO)) for p in (REPO / "vllm/v1/core").glob("kv_cache*.py")
    ]
    source_paths += [
        str(p.relative_to(REPO))
        for p in (REPO / "benchmarks/programmable_kv_cache").rglob("*.py")
    ]
    source_paths += ["benchmarks/programmable_kv_cache/README.md"]
    for name in source_paths:
        path = REPO / name
        if name == "docs/programmable-kvc":
            for child in path.glob("*.md"):
                include(child, Path("raw/source") / child.relative_to(REPO))
        else:
            include(path, Path("raw/source") / name)
    selections = {
        path: target
        for path, target in selections.items()
        if "__pycache__" not in path.parts
    }
    metadata_first = sorted(
        selections, key=lambda p: (p.suffix not in {".json", ".md", ".py"}, str(p))
    )
    for source in metadata_first:
        target = DESTINATION / selections[source]
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as reader, target.open("xb") as writer:
            before = os.fstat(reader.fileno())
            remaining = before.st_size
            while remaining:
                block = reader.read(min(1024 * 1024, remaining))
                if not block:
                    break
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
                if before.st_size > 4 * 1024 * 1024:
                    time.sleep(0.03)
            after = os.fstat(reader.fileno())
        row = {
            "source": str(source.relative_to(REPO)),
            "saved_as": str(selections[source]),
            "bytes": before.st_size - remaining,
            "sha256": digest.hexdigest(),
            "source_observed_bytes": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "changed_during_copy": before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns,
            "short_read": bool(remaining),
        }
        if target.suffix == ".json":
            try:
                json.loads(target.read_bytes())
                row["json_parseable"] = True
            except (ValueError, UnicodeError):
                row["json_parseable"] = False
        manifest["files"].append(row)
    manifest["snapshot_finished"] = timestamp()
    manifest["elapsed_seconds"] = round(time.time() - began, 2)
    manifest["file_count"] = len(manifest["files"])
    manifest["bytes"] = sum(row["bytes"] for row in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in [
                    "snapshot_started",
                    "snapshot_finished",
                    "file_count",
                    "bytes",
                    "elapsed_seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
