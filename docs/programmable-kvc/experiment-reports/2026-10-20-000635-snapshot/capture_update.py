# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Save current evidence separately, reusing unchanged files in the base snapshot."""

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RESULTS = REPO / "benchmarks/programmable_kv_cache/results/acceptance-2026-10-19"
ZONE = dt.timezone(dt.timedelta(hours=-7))


def now():
    return dt.datetime.now(ZONE)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    base = json.loads((ROOT / "evidence-manifest.json").read_text())
    indexed = {row["source"]: row for row in base["files"]}
    stamp = now()
    destination = ROOT / "updates" / stamp.strftime("%Y-%m-%d-%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    selected = set(indexed)
    selected.update(
        str(path.relative_to(REPO))
        for path in RESULTS.rglob("*")
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    )
    selected.add("docs/programmable-kvc/acceptance-report-2026-10-20.md")
    manifest = {
        "snapshot_started": stamp.isoformat(timespec="seconds"),
        "base_manifest": "evidence-manifest.json",
        "path_base": "all saved_as paths are relative to the report directory",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ).strip(),
        "copy_policy": base["copy_policy"],
        "files": [],
        "missing_since_base": [],
    }
    temporary = destination / "capture.tmp"
    for name in sorted(selected):
        source = REPO / name
        if not source.is_file():
            manifest["missing_since_base"].append(name)
            continue
        digest = hashlib.sha256()
        with source.open("rb") as reader, temporary.open("wb") as writer:
            before = os.fstat(reader.fileno())
            remaining = before.st_size
            while remaining:
                block = reader.read(min(1024 * 1024, remaining))
                if not block:
                    break
                writer.write(block)
                digest.update(block)
                remaining -= len(block)
            after = os.fstat(reader.fileno())
        reused = name in indexed and digest.hexdigest() == indexed[name]["sha256"]
        if reused:
            saved_as = indexed[name]["saved_as"]
            temporary.unlink()
        else:
            target = destination / "raw" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.rename(target)
            saved_as = str(target.relative_to(ROOT))
        manifest["files"].append(
            {
                "source": name,
                "saved_as": saved_as,
                "bytes": before.st_size - remaining,
                "sha256": digest.hexdigest(),
                "reused_from_base": reused,
                "changed_during_copy": before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns,
                "short_read": bool(remaining),
            }
        )
    paths = {row["source"]: ROOT / row["saved_as"] for row in manifest["files"]}
    tasks = paths["openspec/changes/programmable-kv-cache/tasks.md"].read_text()
    status = {
        "observed_at": now().isoformat(timespec="seconds"),
        "openspec": {
            "completed": len(re.findall(r"^- \[x\] ", tasks, re.M)),
            "total": len(re.findall(r"^- \[[x ]\] ", tasks, re.M)),
            "remaining": re.findall(r"^- \[ \] (.*)", tasks, re.M),
        },
        "queue": json.loads(
            subprocess.check_output(
                [str(REPO / ".venv/bin/python"), str(RESULTS / "progress.py")],
                cwd=REPO,
                text=True,
            )
        ),
    }
    manifest["snapshot_finished"] = now().isoformat(timespec="seconds")
    manifest["current_file_count"] = len(manifest["files"])
    added = [row for row in manifest["files"] if not row["reused_from_base"]]
    manifest["additional_file_count"] = len(added)
    manifest["additional_bytes"] = sum(row["bytes"] for row in added)
    write_json(destination / "evidence-manifest.json", manifest)
    write_json(destination / "status.json", status)
    latest = {
        "manifest": str((destination / "evidence-manifest.json").relative_to(ROOT)),
        "status": str((destination / "status.json").relative_to(ROOT)),
        "snapshot_finished": manifest["snapshot_finished"],
        "additional_file_count": len(added),
        "additional_bytes": manifest["additional_bytes"],
        "total_saved_evidence_files": base["file_count"] + len(added),
        "total_saved_evidence_bytes": base["bytes"] + manifest["additional_bytes"],
    }
    write_json(ROOT / "latest-update.json", latest)
    print(json.dumps({**latest, "observed_status": status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
