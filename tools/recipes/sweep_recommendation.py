#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Select one measured scheduler recommendation from vLLM sweep results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


DEFAULT_CONFIG_PATH: str | None = None
DEFAULT_ENV_PATH: str | None = None
DEFAULT_TTFT_SLA_MS: float | None = None
DEFAULT_TPOT_SLA_MS: float | None = None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_number(row.get(key)) for row in rows]
    numbers = [value for value in values if value is not None]
    return mean(numbers) if numbers else None


def _positive_int(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


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
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        seqs = _positive_int(row, "max_num_seqs")
        batch = _positive_int(row, "max_num_batched_tokens")
        if seqs is None or batch is None:
            continue
        grouped[(seqs, batch)].append(row)

    candidates: list[dict[str, Any]] = []
    for (seqs, batch), runs in sorted(grouped.items()):
        failed_requests = sum(int(_number(run.get("failed")) or 0) for run in runs)
        output_throughput = _mean(runs, "output_throughput")
        request_goodput = _mean(runs, "request_goodput")
        p99_ttft_ms = _mean(runs, "p99_ttft_ms")
        p99_tpot_ms = _mean(runs, "p99_tpot_ms")

        valid = failed_requests == 0 and output_throughput is not None
        reason = None
        if failed_requests:
            valid = False
            reason = f"{failed_requests} failed request(s) across repeated runs"
        elif use_goodput and request_goodput is None:
            valid = False
            reason = "request_goodput is missing; rerun sweep with SLO goodput enabled"

        candidates.append(
            {
                "max_num_seqs": seqs,
                "max_num_batched_tokens": batch,
                "run_count": len(runs),
                "failed_requests": failed_requests,
                "mean_request_goodput": request_goodput,
                "mean_output_throughput": output_throughput,
                "mean_p99_ttft_ms": p99_ttft_ms,
                "mean_p99_tpot_ms": p99_tpot_ms,
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
) -> tuple[dict[str, Any], str]:
    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        raise ValueError(
            "No valid sweep configuration was found. Inspect the sweep results "
            "for failed requests or missing metrics."
        )

    if use_goodput:
        objective = "highest_mean_request_goodput"
        winner = max(
            valid,
            key=lambda candidate: (
                candidate["mean_request_goodput"],
                candidate["mean_output_throughput"],
                -candidate["max_num_batched_tokens"],
                -candidate["max_num_seqs"],
            ),
        )
        if winner["mean_request_goodput"] <= 0:
            raise ValueError(
                "No sweep configuration produced non-zero request goodput for "
                "the supplied TTFT/TPOT objectives."
            )
    else:
        objective = "highest_mean_output_throughput"
        winner = max(
            valid,
            key=lambda candidate: (
                candidate["mean_output_throughput"],
                -candidate["max_num_batched_tokens"],
                -candidate["max_num_seqs"],
            ),
        )

    return winner, objective


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a YAML configuration object.")
    return data


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
            "Select one recommended max-num-seqs/max-num-batched-tokens pair "
            "from generated vLLM sweep results."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results/runtime-tuning",
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
        "--output-config",
        default="recommended-config.yml",
        help="Recommended config output path.",
    )
    parser.add_argument(
        "--output-json",
        default="recommendation.json",
        help="Recommendation evidence output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    rows = _load_runs(results_dir)
    if not rows:
        raise ValueError(f"No summary.json sweep results found under {results_dir}.")

    use_goodput = args.ttft_sla_ms is not None or args.tpot_sla_ms is not None
    candidates = _aggregate_candidates(rows, use_goodput=use_goodput)
    winner, objective = _select_candidate(candidates, use_goodput=use_goodput)

    initial_config = _load_config(config_path)
    recommended_config = dict(initial_config)
    recommended_config["max-num-seqs"] = winner["max_num_seqs"]
    recommended_config["max-num-batched-tokens"] = winner["max_num_batched_tokens"]

    _write_config(
        output_config,
        source_path=config_path,
        config=recommended_config,
        objective=objective,
    )

    recommendation = {
        "selection_objective": objective,
        "slo": {
            "ttft_ms": args.ttft_sla_ms,
            "tpot_ms": args.tpot_sla_ms,
        },
        "initial": {
            "max_num_seqs": initial_config.get("max-num-seqs"),
            "max_num_batched_tokens": initial_config.get("max-num-batched-tokens"),
        },
        "recommended": {
            "max_num_seqs": winner["max_num_seqs"],
            "max_num_batched_tokens": winner["max_num_batched_tokens"],
        },
        "measured": {
            "run_count": winner["run_count"],
            "mean_request_goodput": winner["mean_request_goodput"],
            "mean_output_throughput": winner["mean_output_throughput"],
            "mean_p99_ttft_ms": winner["mean_p99_ttft_ms"],
            "mean_p99_tpot_ms": winner["mean_p99_tpot_ms"],
        },
        "candidates": candidates,
    }
    output_json.write_text(
        json.dumps(recommendation, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Recommended runtime configuration")
    print()
    print(f"  max-num-seqs:           {winner['max_num_seqs']}")
    print(f"  max-num-batched-tokens: {winner['max_num_batched_tokens']}")
    print()
    print(f"Selection objective: {objective}")
    if use_goodput:
        print(f"  mean request goodput:   {winner['mean_request_goodput']:.2f} req/s")
    print(f"  mean output throughput: {winner['mean_output_throughput']:.2f} tok/s")
    if winner["mean_p99_ttft_ms"] is not None:
        print(f"  mean P99 TTFT:          {winner['mean_p99_ttft_ms']:.2f} ms")
    if winner["mean_p99_tpot_ms"] is not None:
        print(f"  mean P99 TPOT:          {winner['mean_p99_tpot_ms']:.2f} ms")
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
