#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Select one max-concurrency recommendation from a workload sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_TTFT_SLA_MS: float | None = None
DEFAULT_TPOT_SLA_MS: float | None = None
DEFAULT_SEED_CONCURRENCY: int | None = None
DEFAULT_MINIMUM_COMPLIANCE: float = 0.99
DEFAULT_RESULTS_DIR = "results/concurrency-tuning"
DEFAULT_OUTPUT_JSON = "concurrency-recommendation.json"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    numbers = [value for value in values if value is not None]
    return mean(numbers) if numbers else None


def _percentile_summary(
    rows: list[dict[str, Any]], key: str
) -> tuple[float | None, float | None, float | None]:
    values = [_number(row.get(key)) for row in rows]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None, None, None
    return mean(numbers), median(numbers), max(numbers)


def _load_runs(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict):
                copied = dict(row)
                copied["_summary_path"] = str(path)
                rows.append(copied)
    return rows


def _aggregate_candidates(
    rows: list[dict[str, Any]],
    *,
    use_goodput: bool,
    ttft_sla_ms: float | None,
    tpot_sla_ms: float | None,
    minimum_compliance: float,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        concurrency = row.get("max_concurrency")
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency <= 0
        ):
            continue
        grouped[concurrency].append(row)

    candidates: list[dict[str, Any]] = []
    for concurrency, runs in sorted(grouped.items()):
        failed_requests = sum(int(_number(run.get("failed")) or 0) for run in runs)
        output_throughput = _mean(runs, "output_throughput")
        request_throughput = _mean(runs, "request_throughput")
        request_goodput = _mean(runs, "request_goodput")
        mean_p99_ttft_ms, median_p99_ttft_ms, worst_p99_ttft_ms = _percentile_summary(
            runs, "p99_ttft_ms"
        )
        mean_p99_tpot_ms, median_p99_tpot_ms, worst_p99_tpot_ms = _percentile_summary(
            runs, "p99_tpot_ms"
        )

        completed_requests = sum(
            int(_number(run.get("completed")) or 0) for run in runs
        )
        estimated_compliant_requests = sum(
            (_number(run.get("request_goodput")) or 0.0)
            * (_number(run.get("duration")) or 0.0)
            for run in runs
        )
        combined_compliance_ratio = (
            estimated_compliant_requests / completed_requests
            if completed_requests > 0
            else None
        )

        valid = failed_requests == 0 and output_throughput is not None
        reason = None
        if failed_requests:
            valid = False
            reason = f"{failed_requests} failed request(s) across successful runs"
        elif use_goodput and request_goodput is None:
            valid = False
            reason = "request_goodput is missing; rerun with SLO goodput enabled"

        p99_sla_eligible = valid
        if ttft_sla_ms is not None:
            p99_sla_eligible = bool(
                p99_sla_eligible
                and median_p99_ttft_ms is not None
                and median_p99_ttft_ms <= ttft_sla_ms
            )
        if tpot_sla_ms is not None:
            p99_sla_eligible = bool(
                p99_sla_eligible
                and median_p99_tpot_ms is not None
                and median_p99_tpot_ms <= tpot_sla_ms
            )
        compliance_eligible = bool(
            valid
            and combined_compliance_ratio is not None
            and combined_compliance_ratio >= minimum_compliance
        )

        candidates.append(
            {
                "max_concurrency": concurrency,
                "run_count": len(runs),
                "failed_requests": failed_requests,
                "mean_request_goodput": request_goodput,
                "mean_request_throughput": request_throughput,
                "mean_output_throughput": output_throughput,
                "mean_p99_ttft_ms": mean_p99_ttft_ms,
                "median_p99_ttft_ms": median_p99_ttft_ms,
                "worst_p99_ttft_ms": worst_p99_ttft_ms,
                "mean_p99_tpot_ms": mean_p99_tpot_ms,
                "median_p99_tpot_ms": median_p99_tpot_ms,
                "worst_p99_tpot_ms": worst_p99_tpot_ms,
                "estimated_compliant_requests": estimated_compliant_requests,
                "completed_requests": completed_requests,
                "combined_compliance_ratio": combined_compliance_ratio,
                "combined_compliance_percent": (
                    combined_compliance_ratio * 100
                    if combined_compliance_ratio is not None
                    else None
                ),
                "benchmark_valid": valid,
                "p99_sla_eligible": p99_sla_eligible,
                "compliance_eligible": compliance_eligible,
                "sla_eligible": p99_sla_eligible and compliance_eligible,
                "valid": valid,
                "invalid_reason": reason,
                "summary_files": sorted({str(run["_summary_path"]) for run in runs}),
            }
        )

    return candidates


def _select_candidate(
    candidates: list[dict[str, Any]], *, use_goodput: bool
) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError(
            "No valid workload-sweep point was found. Inspect failed runs and metrics."
        )

    if use_goodput:
        objective = "highest_sla_feasible_max_concurrency"
        eligible = [candidate for candidate in valid if candidate["sla_eligible"]]
        winner = (
            max(
                eligible,
                key=lambda candidate: (
                    candidate["max_concurrency"],
                    candidate["mean_output_throughput"],
                    candidate["combined_compliance_ratio"] or 0.0,
                    candidate["mean_request_goodput"] or 0.0,
                ),
            )
            if eligible
            else None
        )
        best_effort = max(
            valid,
            key=lambda candidate: (
                candidate["mean_request_goodput"] or 0.0,
                candidate["combined_compliance_ratio"] or 0.0,
                candidate["mean_output_throughput"],
                candidate["max_concurrency"],
            ),
        )
    else:
        objective = "highest_mean_output_throughput"
        winner = max(
            valid,
            key=lambda candidate: (
                candidate["mean_output_throughput"],
                candidate["max_concurrency"],
            ),
        )
        best_effort = winner

    return winner, best_effort, objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select max_concurrency from vLLM Workload Explorer results."
    )
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ttft-sla-ms", type=float, default=DEFAULT_TTFT_SLA_MS)
    parser.add_argument("--tpot-sla-ms", type=float, default=DEFAULT_TPOT_SLA_MS)
    parser.add_argument(
        "--seed-concurrency", type=int, default=DEFAULT_SEED_CONCURRENCY
    )
    parser.add_argument(
        "--minimum-compliance",
        type=float,
        default=DEFAULT_MINIMUM_COMPLIANCE,
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_compliance <= 1.0:
        raise ValueError("--minimum-compliance must be greater than 0 and at most 1.")

    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = script_dir / results_dir
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = script_dir / output_json

    if not results_dir.exists():
        raise ValueError(f"Workload-sweep results were not found at {results_dir}.")

    rows = _load_runs(results_dir)
    if not rows:
        raise ValueError(f"No summary.json results found under {results_dir}.")

    use_goodput = args.ttft_sla_ms is not None or args.tpot_sla_ms is not None
    candidates = _aggregate_candidates(
        rows,
        use_goodput=use_goodput,
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
        minimum_compliance=args.minimum_compliance,
    )
    winner, best_effort, objective = _select_candidate(
        candidates, use_goodput=use_goodput
    )

    recommendation = {
        "status": (
            "sla_feasible" if winner is not None else "no_sla_feasible_concurrency"
        ),
        "selection_objective": objective,
        "seed_concurrency": args.seed_concurrency,
        "minimum_compliance_ratio": args.minimum_compliance,
        "slo": {
            "ttft_ms": args.ttft_sla_ms,
            "tpot_ms": args.tpot_sla_ms,
        },
        "recommended": (
            {"max_concurrency": winner["max_concurrency"]}
            if winner is not None
            else None
        ),
        "measured": (
            {
                key: winner[key]
                for key in (
                    "run_count",
                    "mean_request_throughput",
                    "mean_request_goodput",
                    "mean_output_throughput",
                    "combined_compliance_ratio",
                    "combined_compliance_percent",
                    "mean_p99_ttft_ms",
                    "median_p99_ttft_ms",
                    "worst_p99_ttft_ms",
                    "mean_p99_tpot_ms",
                    "median_p99_tpot_ms",
                    "worst_p99_tpot_ms",
                )
            }
            if winner is not None
            else None
        ),
        "best_effort": {
            "max_concurrency": best_effort["max_concurrency"],
            "mean_request_goodput": best_effort["mean_request_goodput"],
            "mean_output_throughput": best_effort["mean_output_throughput"],
            "combined_compliance_ratio": best_effort["combined_compliance_ratio"],
            "p99_sla_eligible": best_effort["p99_sla_eligible"],
        },
        "candidates": candidates,
    }
    output_json.write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )

    if winner is None:
        print("No SLA-feasible max_concurrency was found.")
        print(f"Wrote diagnostic evidence to {output_json}")
        return 2

    print(f"Recommended max_concurrency: {winner['max_concurrency']}")
    print(f"Selection objective: {objective}")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
