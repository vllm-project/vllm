#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Find the maximum SLA-feasible serving concurrency adaptively."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from statistics import median
from typing import Any

from vllm.benchmarks.sweep.param_sweep import ParameterSweepItem
from vllm.benchmarks.sweep.serve import run_server
from vllm.benchmarks.sweep.serve_workload import run_comb_workload

DEFAULT_MINIMUM_COMPLIANCE = 0.99
DEFAULT_MAX_CONCURRENCY_CAP = 1000
DEFAULT_GROWTH_FACTOR = 2.0


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _num_prompts(concurrency: int) -> int:
    return min(1000, max(100, concurrency * 10))


def _sla_eligible(
    rows: list[dict[str, Any]],
    *,
    expected_runs: int,
    ttft_sla_ms: float | None,
    tpot_sla_ms: float | None,
    minimum_compliance: float,
) -> bool:
    if len(rows) < expected_runs:
        return False
    if any(int(_number(row.get("failed")) or 0) for row in rows):
        return False

    completed = sum(int(_number(row.get("completed")) or 0) for row in rows)
    if completed <= 0:
        return False

    compliant = sum(
        (_number(row.get("request_goodput")) or 0.0)
        * (_number(row.get("duration")) or 0.0)
        for row in rows
    )
    if compliant / completed < minimum_compliance:
        return False

    if ttft_sla_ms is not None:
        values = [
            value
            for row in rows
            if (value := _number(row.get("p99_ttft_ms"))) is not None
        ]
        if not values or median(values) > ttft_sla_ms:
            return False

    if tpot_sla_ms is not None:
        values = [
            value
            for row in rows
            if (value := _number(row.get("p99_tpot_ms"))) is not None
        ]
        if not values or median(values) > tpot_sla_ms:
            return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the highest SLA-feasible max_concurrency using exponential "
            "bracketing followed by integer binary search."
        )
    )
    parser.add_argument("--serve-cmd", required=True)
    parser.add_argument("--bench-cmd", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--seed-concurrency", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--ttft-sla-ms", type=float)
    parser.add_argument("--tpot-sla-ms", type=float)
    parser.add_argument(
        "--minimum-compliance",
        type=float,
        default=DEFAULT_MINIMUM_COMPLIANCE,
    )
    parser.add_argument(
        "--max-concurrency-cap",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY_CAP,
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=DEFAULT_GROWTH_FACTOR,
    )
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument("--show-stdout", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.seed_concurrency <= 0:
        raise ValueError("--seed-concurrency must be positive.")
    if not 0.0 < args.minimum_compliance <= 1.0:
        raise ValueError("--minimum-compliance must be greater than 0 and at most 1.")
    if args.growth_factor <= 1.0:
        raise ValueError("--growth-factor must be greater than 1.")
    if args.max_concurrency_cap <= 0:
        raise ValueError("--max-concurrency-cap must be positive.")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if args.ttft_sla_ms is None and args.tpot_sla_ms is None:
        raise ValueError("Adaptive search requires TTFT and/or TPOT SLA.")

    seed = args.seed_concurrency
    cap = max(seed, args.max_concurrency_cap)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        first_upper = min(
            cap,
            max(seed + 1, int(round(seed * args.growth_factor))),
        )
        print("Adaptive concurrency search plan:")
        print(f"  seed: {seed}")
        print(f"  first upper probe: {first_upper}")
        print(f"  hard cap: {cap}")
        print("  strategy: bracket SLA boundary, then integer binary search")
        return 0

    serve_cmd = shlex.split(args.serve_cmd)
    bench_cmd = shlex.split(args.bench_cmd)
    serve_comb = ParameterSweepItem()

    def run_point(server: object, concurrency: int) -> bool:
        bench_comb = ParameterSweepItem(
            {
                "_benchmark_name": "user_workload",
                "random_input_len": args.input_tokens,
                "random_output_len": args.output_tokens,
                "num_prompts": _num_prompts(concurrency),
            }
        )
        print(f"[ADAPTIVE CONCURRENCY] testing max_concurrency={concurrency}")
        rows = (
            run_comb_workload(
                server,
                bench_cmd,
                serve_comb=serve_comb,
                bench_comb=bench_comb,
                link_vars=[],
                experiment_dir=results_dir,
                num_runs=args.num_runs,
                dry_run=False,
                warmup_num_prompts=min(concurrency, 1000),
                continue_on_error=args.continue_on_error,
                workload_var="max_concurrency",
                workload_value=concurrency,
            )
            or []
        )

        eligible = _sla_eligible(
            rows,
            expected_runs=args.num_runs,
            ttft_sla_ms=args.ttft_sla_ms,
            tpot_sla_ms=args.tpot_sla_ms,
            minimum_compliance=args.minimum_compliance,
        )
        print(
            "[ADAPTIVE CONCURRENCY] "
            f"max_concurrency={concurrency}: {'PASS' if eligible else 'FAIL'}"
        )
        return eligible

    with run_server(
        serve_cmd,
        [],
        show_stdout=args.show_stdout,
        serve_overrides=serve_comb,
        dry_run=False,
        server_ready_timeout=args.server_ready_timeout,
    ) as server:
        seed_passes = run_point(server, seed)

        if seed_passes:
            low = seed
            high: int | None = None
            while low < cap:
                probe = min(
                    cap,
                    max(low + 1, int(round(low * args.growth_factor))),
                )
                if run_point(server, probe):
                    low = probe
                    if low == cap:
                        print(
                            "[ADAPTIVE CONCURRENCY] reached cap without an SLA failure"
                        )
                        return 0
                else:
                    high = probe
                    break
            if high is None:
                return 0
        else:
            high = seed
            low: int | None = None
            probe = seed
            while probe > 1:
                probe = max(1, probe // 2)
                if run_point(server, probe):
                    low = probe
                    break
                high = probe
            if low is None:
                print(
                    "[ADAPTIVE CONCURRENCY] no SLA-feasible concurrency found, "
                    "including max_concurrency=1"
                )
                return 0

        assert low is not None
        assert high is not None
        print(f"[ADAPTIVE CONCURRENCY] refining bracket PASS={low}, FAIL={high}")

        while high - low > 1:
            midpoint = (low + high) // 2
            if run_point(server, midpoint):
                low = midpoint
            else:
                high = midpoint

        print(f"[ADAPTIVE CONCURRENCY] boundary resolved: PASS={low}, FAIL={high}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
