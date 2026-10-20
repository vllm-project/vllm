# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete equal-work runs and audit application control timing."""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


def audit(path):
    with gzip.open(path, "rt") as file:
        report = json.load(file)
    trajectories = defaultdict(list)
    for row in report["requests"]:
        trajectories[row["request_id"]].append(row)
    pauses = []
    for rows in trajectories.values():
        rows.sort(key=lambda row: row["turn"])
        pauses.extend(b["started"] - a["finished"] for a, b in zip(rows, rows[1:]))
    signals = defaultdict(list)
    for row in report["controls"]:
        if "driver_signal" in row:
            signals[row["driver_signal"]].append(row.get("driver_seconds", 0))
    summary = report.get("summary", {})
    forecasts = report.get("tool_waits", [])
    errors = [r["predicted_seconds"] - r["actual_seconds"] for r in forecasts]
    manifest_path = path.parent / "matrix.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config = dict(manifest.get("config", {}))
    for command in manifest.get("commands", []):
        if "vllm.entrypoints.openai.api_server" not in command:
            continue
        for key, flag in (
            ("max_num_seqs", "--max-num-seqs"),
            ("batch_tokens", "--max-num-batched-tokens"),
        ):
            if flag in command:
                config[key] = int(command[command.index(flag) + 1])
        break
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_revision": report.get("source_revision"),
        "source_hashes": report.get("source_hashes"),
        "passed": report["passed"],
        "seconds": report.get("seconds"),
        "workload_sha256": summary.get("workload_sha256"),
        "pause_seconds": distribution(pauses),
        "pauses_below_100ms": sum(value < 0.1 for value in pauses),
        "signal_seconds": {
            name: distribution(values) for name, values in signals.items()
        },
        "forecast_error_seconds": distribution(errors),
        "forecast_absolute_error_seconds": distribution([abs(v) for v in errors]),
        "forecast_underestimates": sum(v < 0 for v in errors),
        "forecast_overestimates": sum(v > 0 for v in errors),
        "client_queue_seconds": distribution(
            [r["client_queue_seconds"] for r in report.get("episodes", [])]
        ),
        "summary": summary,
        "instrumented": config.get("profile", False),
        "conditions": {
            key: config.get(key)
            for key in (
                "model",
                "gpus",
                "gpu_memory_utilization",
                "cache_gib",
                "concurrency",
                "max_num_seqs",
                "batch_tokens",
                "max_context",
                "enforce_eager",
                "disable_custom_all_reduce",
                "replay_pause_floor",
            )
        }
        if config
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, help="LABEL=report.json.gz"
    )
    parser.add_argument("--baseline", default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = defaultdict(list)
    for entry in args.run:
        label, path = entry.split("=", 1)
        runs[label].append(audit(Path(path)))
    fingerprints = {row["workload_sha256"] for group in runs.values() for row in group}
    comparable = len(fingerprints) == 1 and None not in fingerprints
    complete = all(row["passed"] for group in runs.values() for row in group)
    conditions = {
        json.dumps(r["conditions"], sort_keys=True)
        for group in runs.values()
        for r in group
        if not r["instrumented"]
    }
    equal_conditions = len(conditions) == 1 and "null" not in conditions
    comparison = {}
    if comparable and complete and equal_conditions and args.baseline in runs:
        if any(r["instrumented"] for r in runs[args.baseline]):
            raise ValueError("Timing baseline must not be instrumented")
        baseline = statistics.median(r["seconds"] for r in runs[args.baseline])
        baseline_families = defaultdict(list)
        for row in runs[args.baseline]:
            for family, values in row["summary"].get("family_latency", {}).items():
                baseline_families[family].append(values["p95_seconds"])
        for label, group in runs.items():
            if any(r["instrumented"] for r in group):
                continue
            seconds = [r["seconds"] for r in group]
            families = defaultdict(list)
            for row in group:
                for family, values in row["summary"].get("family_latency", {}).items():
                    families[family].append(values["p95_seconds"])
            ratios = {
                family: statistics.median(values)
                / statistics.median(baseline_families[family])
                for family, values in families.items()
                if family in baseline_families
                and statistics.median(baseline_families[family]) > 0
            }
            comparison[label] = {
                "seconds": distribution(seconds),
                "throughput_gain_percent": 100
                * (baseline / statistics.median(seconds) - 1),
                "ttft_p99_seconds": distribution(
                    r["summary"]["ttft_p99"] for r in group
                ),
                "worst_family_p95_seconds": distribution(
                    max(
                        f["p95_seconds"]
                        for f in r["summary"]["family_latency"].values()
                    )
                    for r in group
                    if r["summary"].get("family_latency")
                ),
                "family_p95_ratio_to_baseline": ratios,
                "families_with_higher_p95": sum(v > 1 for v in ratios.values()),
            }
    result = {
        "equal_work": comparable,
        "all_complete": complete,
        "equal_timing_conditions": equal_conditions,
        "comparison": comparison,
        "runs": dict(runs),
        "limitations": (
            "Equal token inputs/output lengths do not establish equal sampling, "
            "hardware occupancy, profiling or configuration. Inspect run manifests. "
            "Pause intervals include client work; signal intervals include RPC, "
            "polling and ACK time, and are not device transfer durations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("equal_work", "all_complete", "comparison")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
