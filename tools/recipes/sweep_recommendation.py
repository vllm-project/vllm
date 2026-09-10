#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Select one measured parallel-layout or scheduler sweep recommendation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


DEFAULT_CONFIG_PATH: str | None = None
DEFAULT_ENV_PATH: str | None = None
DEFAULT_TTFT_SLA_MS: float | None = None
DEFAULT_TPOT_SLA_MS: float | None = None
DEFAULT_MINIMUM_COMPLIANCE: float = 0.99
DEFAULT_RESULTS_DIR = "results/runtime-tuning"
DEFAULT_OUTPUT_CONFIG = "recommended-config.yml"
DEFAULT_OUTPUT_JSON = "recommendation.json"


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


def _scheduler_value(
    row: dict[str, Any], key: str
) -> tuple[bool, int | None]:
    if key not in row:
        # Default-reference sweep candidates omit scheduler keys entirely.
        return True, None
    value = row[key]
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return False, None
    return True, value


def _parallel_value(
    row: dict[str, Any], key: str, default: int
) -> tuple[bool, int]:
    if key not in row:
        return True, default
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return False, default
    return True, value


def _scheduler_sort_value(value: int | None) -> int:
    return -1 if value is None else value


def _scheduler_preference_key(value: object) -> tuple[int, int]:
    """Prefer vLLM default, then the smaller explicit value on exact ties."""
    if value is None:
        return (1, 0)
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, -value)
    return (0, 0)


def _resolve(script_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else script_dir / path


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
    default_tensor_parallel_size: int = 1,
    default_data_parallel_size: int = 1,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
    minimum_compliance: float = DEFAULT_MINIMUM_COMPLIANCE,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, int, int | None, int | None], list[dict[str, Any]]
    ] = defaultdict(list)

    for row in rows:
        seqs_valid, seqs = _scheduler_value(row, "max_num_seqs")
        batch_valid, batch = _scheduler_value(row, "max_num_batched_tokens")
        tp_valid, tp = _parallel_value(
            row, "tensor_parallel_size", default_tensor_parallel_size
        )
        dp_valid, dp = _parallel_value(
            row, "data_parallel_size", default_data_parallel_size
        )
        if not seqs_valid or not batch_valid or not tp_valid or not dp_valid:
            continue
        grouped[(tp, dp, seqs, batch)].append(row)

    candidates: list[dict[str, Any]] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            _scheduler_sort_value(item[0][2]),
            _scheduler_sort_value(item[0][3]),
        ),
    )
    for (tp, dp, seqs, batch), runs in sorted_groups:
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
            reason = f"{failed_requests} failed request(s) across repeated runs"
        elif use_goodput and request_goodput is None:
            valid = False
            reason = "request_goodput is missing; rerun sweep with SLO goodput enabled"

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
                "tensor_parallel_size": tp,
                "data_parallel_size": dp,
                "max_num_seqs": seqs,
                "max_num_batched_tokens": batch,
                "run_count": len(runs),
                "vllm_default_parameters": [
                    name
                    for name, value in (
                        ("max_num_seqs", seqs),
                        ("max_num_batched_tokens", batch),
                    )
                    if value is None
                ],
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
    candidates: list[dict[str, Any]],
    *,
    use_goodput: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError(
            "No valid sweep configuration was found. Inspect the sweep results "
            "for failed requests or missing metrics."
        )

    if use_goodput:
        objective = "p99_constrained_throughput"
        best_effort = max(
            valid,
            key=lambda candidate: (
                candidate["mean_request_goodput"],
                candidate["combined_compliance_ratio"] or 0.0,
                candidate["mean_output_throughput"],
                -candidate["data_parallel_size"],
                -candidate["tensor_parallel_size"],
                _scheduler_preference_key(candidate["max_num_batched_tokens"]),
                _scheduler_preference_key(candidate["max_num_seqs"]),
            ),
        )
        if best_effort["mean_request_goodput"] <= 0:
            raise ValueError(
                "No sweep configuration produced non-zero request goodput for "
                "the supplied TTFT/TPOT objectives."
            )
        eligible = [candidate for candidate in valid if candidate["sla_eligible"]]
        winner = (
            max(
                eligible,
                key=lambda candidate: (
                    candidate["mean_output_throughput"],
                    candidate["combined_compliance_ratio"],
                    candidate["mean_request_goodput"],
                    -candidate["data_parallel_size"],
                    -candidate["tensor_parallel_size"],
                    _scheduler_preference_key(candidate["max_num_batched_tokens"]),
                    _scheduler_preference_key(candidate["max_num_seqs"]),
                ),
            )
            if eligible
            else None
        )
    else:
        objective = "highest_mean_output_throughput"
        winner = max(
            valid,
            key=lambda candidate: (
                candidate["mean_output_throughput"],
                -candidate["data_parallel_size"],
                -candidate["tensor_parallel_size"],
                _scheduler_preference_key(candidate["max_num_batched_tokens"]),
                _scheduler_preference_key(candidate["max_num_seqs"]),
            ),
        )
        best_effort = winner

    return winner, best_effort, objective


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML configuration object.")
    return data


def _build_recommended_config(
    initial_config: dict[str, Any],
    winner: dict[str, Any],
) -> dict[str, Any]:
    recommended_config = dict(initial_config)
    recommended_config["tensor-parallel-size"] = winner["tensor_parallel_size"]
    recommended_config["data-parallel-size"] = winner["data_parallel_size"]
    for config_key, result_key in (
        ("max-num-seqs", "max_num_seqs"),
        ("max-num-batched-tokens", "max_num_batched_tokens"),
    ):
        value = winner[result_key]
        if value is None:
            recommended_config.pop(config_key, None)
        else:
            recommended_config[config_key] = value
    return recommended_config


def _format_scheduler_value(value: object) -> str:
    if value is None:
        return "vLLM default"
    return str(value)


def _write_config(
    path: Path,
    *,
    source_path: Path,
    config: dict[str, Any],
    objective: str,
) -> None:
    body = yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    header = (
        "# Generated from the initial recipe config after a vLLM parameter sweep.\n"
        f"# Initial config: {source_path}\n"
        f"# Selection objective: {objective}\n"
    )
    path.write_text(header + body, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one recommended TP/DP layout and scheduler setting from "
            "generated vLLM sweep results."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Sweep experiment directory (default: results/runtime-tuning).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Initial config.yml used by the sweep.",
    )
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_PATH,
        help="Environment file to show in the deployment instructions.",
    )
    parser.add_argument(
        "--ttft-sla-ms",
        type=float,
        default=DEFAULT_TTFT_SLA_MS,
        help="TTFT objective used to generate request goodput.",
    )
    parser.add_argument(
        "--tpot-sla-ms",
        type=float,
        default=DEFAULT_TPOT_SLA_MS,
        help="TPOT objective used to generate request goodput.",
    )
    parser.add_argument(
        "--minimum-compliance",
        type=float,
        default=DEFAULT_MINIMUM_COMPLIANCE,
        help=(
            "Minimum duration-weighted fraction of completed requests that must "
            "meet all supplied objectives (default: 0.99)."
        ),
    )
    parser.add_argument(
        "--output-config",
        default=DEFAULT_OUTPUT_CONFIG,
        help="Recommended config output path.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Recommendation evidence output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_compliance <= 1.0:
        raise ValueError("--minimum-compliance must be greater than 0 and at most 1.")
    script_dir = Path(__file__).resolve().parent

    config_path = _resolve(script_dir, args.config)
    env_path = _resolve(script_dir, args.env)
    results_dir = _resolve(script_dir, args.results_dir)
    output_config = _resolve(script_dir, args.output_config)
    output_json = _resolve(script_dir, args.output_json)

    if config_path is None:
        raise ValueError("--config is required.")
    assert results_dir is not None
    assert output_config is not None
    assert output_json is not None

    if not results_dir.exists():
        raise ValueError(
            f"Sweep results were not found at {results_dir}. Run the sweep first."
        )

    initial_config = _load_config(config_path)
    rows = _load_runs(results_dir)
    if not rows:
        raise ValueError(f"No summary.json sweep results found under {results_dir}.")

    use_goodput = args.ttft_sla_ms is not None or args.tpot_sla_ms is not None
    candidates = _aggregate_candidates(
        rows,
        use_goodput=use_goodput,
        default_tensor_parallel_size=int(initial_config.get("tensor-parallel-size", 1)),
        default_data_parallel_size=int(initial_config.get("data-parallel-size", 1)),
        ttft_sla_ms=args.ttft_sla_ms,
        tpot_sla_ms=args.tpot_sla_ms,
        minimum_compliance=args.minimum_compliance,
    )
    winner, best_effort, objective = _select_candidate(
        candidates, use_goodput=use_goodput
    )

    if winner is not None:
        recommended_config = _build_recommended_config(initial_config, winner)
        _write_config(
            output_config,
            source_path=config_path,
            config=recommended_config,
            objective=objective,
        )
    elif output_config.exists():
        output_config.unlink()

    measured = None
    recommended = None
    if winner is not None:
        recommended = {
            "tensor_parallel_size": winner["tensor_parallel_size"],
            "data_parallel_size": winner["data_parallel_size"],
            "max_num_seqs": winner["max_num_seqs"],
            "max_num_batched_tokens": winner["max_num_batched_tokens"],
            "vllm_default_parameters": winner["vllm_default_parameters"],
        }
        measured = {
            key: winner[key]
            for key in (
                "run_count",
                "mean_request_throughput",
                "mean_request_goodput",
                "mean_output_throughput",
                "combined_compliance_ratio",
                "combined_compliance_percent",
                "estimated_compliant_requests",
                "completed_requests",
                "mean_p99_ttft_ms",
                "median_p99_ttft_ms",
                "worst_p99_ttft_ms",
                "mean_p99_tpot_ms",
                "median_p99_tpot_ms",
                "worst_p99_tpot_ms",
            )
        }

    recommendation = {
        "status": (
            "sla_feasible" if winner is not None else "no_sla_feasible_configuration"
        ),
        "selection_objective": objective,
        "minimum_compliance_ratio": args.minimum_compliance,
        "slo": {
            "ttft_ms": args.ttft_sla_ms,
            "tpot_ms": args.tpot_sla_ms,
        },
        "initial": {
            "tensor_parallel_size": initial_config.get("tensor-parallel-size"),
            "data_parallel_size": initial_config.get("data-parallel-size", 1),
            "max_num_seqs": initial_config.get("max-num-seqs"),
            "max_num_batched_tokens": initial_config.get("max-num-batched-tokens"),
        },
        "recommended": recommended,
        "measured": measured,
        "best_effort": {
            "tensor_parallel_size": best_effort["tensor_parallel_size"],
            "data_parallel_size": best_effort["data_parallel_size"],
            "max_num_seqs": best_effort["max_num_seqs"],
            "max_num_batched_tokens": best_effort["max_num_batched_tokens"],
            "vllm_default_parameters": best_effort["vllm_default_parameters"],
            "mean_request_goodput": best_effort["mean_request_goodput"],
            "combined_compliance_ratio": best_effort["combined_compliance_ratio"],
            "p99_sla_eligible": best_effort["p99_sla_eligible"],
        },
        "candidates": candidates,
    }
    output_json.write_text(
        json.dumps(recommendation, indent=2) + "\n",
        encoding="utf-8",
    )

    if winner is None:
        print("No SLA-feasible runtime configuration was found.")
        print(f"Wrote diagnostic evidence to {output_json}")
        print("No recommended config was written.")
        return 2

    print("Recommended runtime configuration")
    print()
    print(f"  tensor-parallel-size:   {winner['tensor_parallel_size']}")
    print(f"  data-parallel-size:     {winner['data_parallel_size']}")
    print(
        "  max-num-seqs:           "
        + _format_scheduler_value(winner["max_num_seqs"])
    )
    print(
        "  max-num-batched-tokens: "
        + _format_scheduler_value(winner["max_num_batched_tokens"])
    )
    print()
    print(f"Selection objective: {objective}")
    if use_goodput:
        print(f"  mean request goodput:   {winner['mean_request_goodput']:.2f} req/s")
        print(f"  combined compliance:   {winner['combined_compliance_percent']:.2f}%")
    print(f"  mean output throughput: {winner['mean_output_throughput']:.2f} tok/s")
    if winner["median_p99_ttft_ms"] is not None:
        print(f"  median P99 TTFT:        {winner['median_p99_ttft_ms']:.2f} ms")
    if winner["median_p99_tpot_ms"] is not None:
        print(f"  median P99 TPOT:        {winner['median_p99_tpot_ms']:.2f} ms")
    print()
    print(f"Wrote {output_config}")
    print(f"Wrote {output_json}")
    print()
    print("Deploy:")
    if env_path is not None:
        print(f"  source {env_path}")
    print(f"  vllm serve --config {output_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
