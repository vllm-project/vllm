#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare n>1 and n=1 vLLM bench serve JSON results (issue #69408).

This is a performance *diagnostic*, not a fix. Run the same workload across
releases on the same GPU, alternating the order to reduce warmup drift.
"""

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def _positive_number(run: dict, key: str) -> float:
    value = run.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return float(value)


def _get_metric(run: dict, n: int) -> tuple[float, float]:
    """Return (completed samples/s, per-run median TPOT milliseconds)."""
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    failed = run.get("failed", 0)
    if type(failed) is not int or failed != 0:
        raise ValueError("failed must be zero for comparable runs")
    if "completed" in run or "duration" in run:
        completed = run.get("completed")
        if type(completed) is not int or completed <= 0:
            raise ValueError("completed must be a positive integer")
        requests_per_second = completed / _positive_number(run, "duration")
    else:
        requests_per_second = _positive_number(run, "request_throughput")
    samples_per_second = n * requests_per_second
    if not math.isfinite(samples_per_second):
        raise ValueError("samples/s must be finite")
    return samples_per_second, _positive_number(run, "median_tpot_ms")


def _load_files(paths: list[str], n: int) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    files: set[str] = set()
    for pattern in paths:
        matches = [name for name in glob.glob(pattern) if Path(name).is_file()]
        if not matches:
            raise ValueError(f"No JSON files matched {pattern!r}")
        files.update(matches)
    for name in sorted(files):
        data = json.loads(Path(name).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object: {name}")
        try:
            result.append(_get_metric(data, n))
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
    return result


def summarize(
    baseline: list[tuple[float, float]], candidate: list[tuple[float, float]]
):
    if not baseline or not candidate:
        raise ValueError("Need baseline and candidate results")
    base_samples = statistics.median(row[0] for row in baseline)
    new_samples = statistics.median(row[0] for row in candidate)
    base_tpot = statistics.median(row[1] for row in baseline)
    new_tpot = statistics.median(row[1] for row in candidate)
    if base_samples <= 0 or base_tpot <= 0:
        raise ValueError("Baseline throughput and TPOT must be positive")
    return {
        "runs_baseline": len(baseline),
        "runs_candidate": len(candidate),
        "baseline_samples_per_s": base_samples,
        "candidate_samples_per_s": new_samples,
        "baseline_samples_per_s_range": [
            min(row[0] for row in baseline),
            max(row[0] for row in baseline),
        ],
        "candidate_samples_per_s_range": [
            min(row[0] for row in candidate),
            max(row[0] for row in candidate),
        ],
        "throughput_change_pct": 100 * (new_samples / base_samples - 1),
        "baseline_median_tpot_ms": base_tpot,
        "candidate_median_tpot_ms": new_tpot,
        "tpot_change_pct": 100 * (new_tpot / base_tpot - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--control-baseline", nargs="+", required=True)
    parser.add_argument("--control-candidate", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        high_n = summarize(
            _load_files(args.baseline, args.n),
            _load_files(args.candidate, args.n),
        )
        control = summarize(
            _load_files(args.control_baseline, 1),
            _load_files(args.control_candidate, 1),
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "n": args.n,
        "parallel": high_n,
        "control_n1": control,
        "likely_n_specific_regression": (
            min(
                high_n["runs_baseline"],
                high_n["runs_candidate"],
                control["runs_baseline"],
                control["runs_candidate"],
            )
            >= 3
            and high_n["throughput_change_pct"] <= -5
            and abs(control["throughput_change_pct"]) <= 3
        ),
        "limitations": (
            "Performance attribution needs identical workloads, hardware, "
            "image dependencies, sufficient repeat runs, and profiler evidence. "
            "The threshold flag is a descriptive heuristic, and ranges describe "
            "only the observed repeats."
        ),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    print(encoded)
    if args.output_json:
        args.output_json.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
