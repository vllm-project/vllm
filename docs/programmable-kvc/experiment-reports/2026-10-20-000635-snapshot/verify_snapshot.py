# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify archived bytes, local report links and independently computed gains."""

import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    latest = read("latest-update.json")
    manifests = [read("evidence-manifest.json"), read(latest["manifest"])]
    checked = {}
    failures = []
    for manifest in manifests:
        for row in manifest["files"]:
            name = row["saved_as"]
            path = ROOT / name
            if row["short_read"]:
                failures.append(f"Short read: {name}")
            if name not in checked:
                with path.open("rb") as file:
                    checked[name] = (
                        path.stat().st_size,
                        hashlib.file_digest(file, "sha256").hexdigest(),
                    )
                if path.suffix == ".json":
                    json.loads(path.read_text())
            if checked[name] != (row["bytes"], row["sha256"]):
                failures.append(f"Checksum/length mismatch: {name}")
    links = 0
    for path in ROOT.glob("*.md"):
        for href in re.findall(r"\]\(([^\n]+?)\)", path.read_text()):
            if href.startswith(("http:", "https:", "#")):
                continue
            links += 1
            if href != "verification.json" and not (ROOT / href.split("#")[0]).exists():
                failures.append(f"Missing link in {path.name}: {href}")
    summary = read("summary.json")
    pressure = summary["primary_pressure"]
    native = statistics.median(pressure["native"]["seconds"])
    gains = {}
    for group, data in pressure.items():
        gain = (native / statistics.median(data["seconds"]) - 1) * 100
        gains[group] = gain
        if abs(gain - data["reported_gain_percent"]) > 1e-8:
            failures.append(f"Incorrect throughput gain: {group}")
    for group, data in summary["acceptance_forecasts"].items():
        gain = (native / statistics.median(data["seconds"]) - 1) * 100
        gains[group] = gain
        if abs(gain - data["gain_vs_native_percent"]) > 1e-8:
            failures.append(f"Incorrect forecast gain: {group}")
    result = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evidence_files_checked": len(checked),
        "evidence_bytes_checked": sum(size for size, _ in checked.values()),
        "local_links_checked": links,
        "throughput_gain_percent": gains,
        "changed_during_capture": [
            row["saved_as"]
            for manifest in manifests
            for row in manifest["files"]
            if row["changed_during_copy"]
        ],
        "failures": failures,
        "passed": not failures,
    }
    (ROOT / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
