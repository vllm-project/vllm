# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build report tables exclusively from the frozen experiment snapshot."""

import gzip
import json
import statistics
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw/results"


def read(path):
    return json.loads(path.read_text())


def compact_run(row):
    keep = [
        "arm",
        "repeat",
        "seconds",
        "completed_rollouts",
        "requests",
        "generated_tokens",
        "prompt_tokens",
        "cached_tokens",
        "ttft_p99",
        "rollout_latency_p95",
        "mean_server_queue_seconds",
        "driver_cpu_seconds",
        "server_preemptions",
        "transfers",
        "workload_sha256",
    ]
    return {key: row[key] for key in keep if key in row}


def comparison(path):
    data = read(path)
    result = {}
    for group, rows in data["runs"].items():
        summaries = [row["summary"] for row in rows]
        result[group] = {
            "seconds": [row["seconds"] for row in rows],
            "median_seconds": statistics.median(row["seconds"] for row in rows),
            "reported_gain_percent": data["comparison"][group][
                "throughput_gain_percent"
            ],
            "source_revisions": sorted({row["source_revision"] for row in rows}),
            "runs": [
                {"seconds": row["seconds"], **compact_run(row["summary"])}
                for row in rows
            ],
            "median_metrics": {
                key: statistics.median(row[key] for row in summaries)
                for key in [
                    "cached_tokens",
                    "ttft_p99",
                    "rollout_latency_p95",
                    "driver_cpu_seconds",
                    "mean_server_queue_seconds",
                ]
            },
        }
    return result


def main():
    archive = read(ROOT / "evidence-manifest.json")
    main_pressure = comparison(
        RAW / "rollout-2026-10-19/final-pressure-comparison.json"
    )
    tasks = (
        ROOT / "raw/source/openspec/changes/programmable-kv-cache/tasks.md"
    ).read_text()
    output = {
        "snapshot_started": archive["snapshot_started"],
        "snapshot_finished": archive["snapshot_finished"],
        "snapshot_git_revision": archive["git_revision"],
        "primary_pressure": main_pressure,
        "capacity_fit": comparison(
            RAW / "rollout-2026-10-19/final-fit-comparison.json"
        ),
        "openspec": {
            "completed": len(re.findall(r"^- \[x\] ", tasks, re.M)),
            "total": len(re.findall(r"^- \[[x ]\] ", tasks, re.M)),
            "remaining": re.findall(r"^- \[ \] (.*)", tasks, re.M),
        },
        "matrix_index": [],
        "acceptance_forecasts": {},
    }
    for path in sorted(RAW.rglob("matrix.json")):
        data = read(path)
        output["matrix_index"].append(
            {
                "path": str(path.relative_to(ROOT)),
                "passed": data.get("passed"),
                "error": data.get("error"),
                "planned_repeats": data.get("config", {}).get("repeats"),
                "arms": data.get("config", {}).get("arms"),
                "runs": [compact_run(row) for row in data.get("runs", [])],
                "live_runs": [compact_run(row) for row in data.get("live_runs", [])],
            }
        )
        if path.parent.name in {
            "pressure-oracle",
            "pressure-forecast-early",
            "pressure-forecast-late",
        }:
            seconds = [row["seconds"] for row in data["runs"]]
            median = statistics.median(seconds)
            output["acceptance_forecasts"][path.parent.name] = {
                "passed": data["passed"],
                "seconds": seconds,
                "median_seconds": median,
                "gain_vs_native_percent": (
                    main_pressure["native"]["median_seconds"] / median - 1
                )
                * 100,
                "gain_vs_selected_cooperative_percent": (
                    main_pressure["cooperative"]["median_seconds"] / median - 1
                )
                * 100,
                "prediction_mode": "recorded_tool_pause"
                if data["config"].get("oracle")
                else data["config"].get("prediction_mode"),
                "fixed_prediction_seconds": data["config"].get("prediction_seconds"),
                "runs": [compact_run(row) for row in data["runs"]],
            }
    output["functional"] = read(RAW / "acceptance-2026-10-19/functional-summary.json")
    output["fit_overhead"] = read(
        RAW / "acceptance-2026-10-19/fit-overhead-summary.json"
    )
    output["queue_at_capture"] = read(RAW / "acceptance-2026-10-19/queue.json")
    priority = RAW / "acceptance-2026-10-19/pressure-direct/priority-0.json.gz"
    with gzip.open(priority, "rt") as file:
        data = json.load(file)
    output["excluded_priority_diagnostic"] = {
        "client_passed": data["passed"],
        "seconds": data["seconds"],
        "summary": compact_run(data["summary"]),
        "exclusion_reason": read(priority.parent / "matrix.json").get("error"),
        "accepted_for_speed_claims": False,
    }
    indexed = {row["source"]: row for row in archive["files"]}
    runtime = read(RAW / "acceptance-2026-10-19/runtime-identity.json")
    checks = {
        "all_json_parseable": all(
            row.get("json_parseable", True) for row in archive["files"]
        ),
        "no_short_copies": not any(row["short_read"] for row in archive["files"]),
        "source_changed_during_copy": [
            row["source"] for row in archive["files"] if row["changed_during_copy"]
        ],
        "runtime_hashes_match_saved_identity": all(
            indexed[name]["sha256"] == digest
            for name, digest in runtime["source_hashes"].items()
        ),
        "accepted_pressure_artifact_hashes_match": True,
        "archived_bytes": archive["bytes"],
        "archived_files": archive["file_count"],
    }
    for filename in ["final-pressure-comparison.json", "final-fit-comparison.json"]:
        data = read(RAW / "rollout-2026-10-19" / filename)
        for rows in data["runs"].values():
            for row in rows:
                name = str(Path(row["path"]).relative_to(archive["repository"]))
                checks["accepted_pressure_artifact_hashes_match"] &= (
                    indexed[name]["sha256"] == row["sha256"]
                )
    output["snapshot_checks"] = checks
    assert checks["all_json_parseable"] and checks["no_short_copies"]
    assert checks["runtime_hashes_match_saved_identity"]
    assert checks["accepted_pressure_artifact_hashes_match"]
    (ROOT / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "checks": checks,
                "forecast_results": {
                    key: {k: v for k, v in row.items() if k != "runs"}
                    for key, row in output["acceptance_forecasts"].items()
                },
                "matrix_count": len(output["matrix_index"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
