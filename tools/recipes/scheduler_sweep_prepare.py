#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare scheduler sweep inputs after TP/DP and concurrency selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

PROMPTS_PER_CONCURRENCY = 10
MIN_NUM_PROMPTS = 100
MAX_NUM_PROMPTS = 1000


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Expected positive integer {key!r}, got {value!r}.")
    return value


def _num_prompts_for_concurrency(concurrency: int) -> int:
    return min(
        MAX_NUM_PROMPTS,
        max(MIN_NUM_PROMPTS, concurrency * PROMPTS_PER_CONCURRENCY),
    )


def _strict_lower_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length() - 1)


def _strict_upper_power_of_two(value: int) -> int:
    return 1 << value.bit_length()


def _scheduler_baseline(
    config: dict[str, Any],
    *,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    tpot_sla_ms: float | None,
    target_qps: float | None,
) -> tuple[int, int]:
    data_parallel_size = int(config.get("data-parallel-size", 1))
    if data_parallel_size <= 0:
        raise ValueError("data-parallel-size must be positive.")

    per_replica_concurrency = math.ceil(concurrency / data_parallel_size)
    prefills_per_step = max(1.0, per_replica_concurrency / max(output_tokens, 1))
    if target_qps is not None and tpot_sla_ms is not None:
        per_replica_qps = target_qps / data_parallel_size
        prefills_per_step = max(
            prefills_per_step,
            per_replica_qps * tpot_sla_ms / 1000.0,
        )
    prefills_per_step = min(float(per_replica_concurrency), prefills_per_step)

    batch = max(
        2048,
        per_replica_concurrency,
        per_replica_concurrency + math.ceil(input_tokens * prefills_per_step),
    )
    if config.get("enable-chunked-prefill") is False:
        max_model_len = config.get("max-model-len")
        if isinstance(max_model_len, int) and not isinstance(max_model_len, bool):
            batch = max(batch, max_model_len)

    return per_replica_concurrency, batch


def _build_serve_params(config: dict[str, Any]) -> list[dict[str, Any]]:
    initial_seqs = _positive_int(config, "max-num-seqs")
    initial_batch = _positive_int(config, "max-num-batched-tokens")

    minimum_batch = initial_seqs
    if config.get("enable-chunked-prefill") is False:
        max_model_len = config.get("max-model-len")
        if (
            isinstance(max_model_len, int)
            and not isinstance(max_model_len, bool)
            and max_model_len > 0
        ):
            minimum_batch = max(minimum_batch, max_model_len)

    lower_seqs = max(1, (initial_seqs + 1) // 2)
    middle_seqs = max(lower_seqs, (3 * initial_seqs + 3) // 4)
    lower_batch = max(minimum_batch, _strict_lower_power_of_two(initial_batch))
    smaller_batch = max(minimum_batch, _strict_lower_power_of_two(lower_batch))
    higher_batch = max(minimum_batch, _strict_upper_power_of_two(initial_batch))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int | None, int | None]] = set()

    def add(
        name: str,
        max_num_seqs: int | None,
        max_num_batched_tokens: int | None,
    ) -> None:
        if max_num_seqs is not None and max_num_batched_tokens is not None:
            max_num_batched_tokens = max(max_num_batched_tokens, max_num_seqs)
        signature = (max_num_seqs, max_num_batched_tokens)
        if signature in seen:
            return
        seen.add(signature)

        candidate: dict[str, Any] = {"_benchmark_name": name}
        if max_num_seqs is not None:
            candidate["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            candidate["max_num_batched_tokens"] = max_num_batched_tokens
        candidates.append(candidate)

    add("initial", initial_seqs, initial_batch)
    add("smaller_batch_budget", initial_seqs, smaller_batch)
    add("lower_batch_budget", initial_seqs, lower_batch)
    add("higher_batch_budget", initial_seqs, higher_batch)
    add("middle_seqs_lower_batch", middle_seqs, lower_batch)
    add("middle_seqs_higher_batch", middle_seqs, higher_batch)
    add("lower_seqs_lower_batch", lower_seqs, lower_batch)
    add("lower_seqs_higher_batch", lower_seqs, higher_batch)
    add("vllm_default_max_num_seqs", None, initial_batch)
    add("vllm_default_max_num_batched_tokens", initial_seqs, None)
    add("vllm_defaults", None, None)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--concurrency-recommendation", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--tpot-sla-ms", type=float)
    parser.add_argument("--target-qps", type=float)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--serve-params", required=True)
    parser.add_argument("--bench-params", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    recommendation_path = Path(args.concurrency_recommendation)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} does not contain a YAML object.")

    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    selected = recommendation.get("recommended")
    if not isinstance(selected, dict):
        raise ValueError(
            f"{recommendation_path} has no SLA-feasible concurrency recommendation."
        )
    concurrency = selected.get("max_concurrency")
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
    ):
        raise ValueError("Recommended max_concurrency is missing or invalid.")

    seqs, batch = _scheduler_baseline(
        config,
        concurrency=concurrency,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        tpot_sla_ms=args.tpot_sla_ms,
        target_qps=args.target_qps,
    )

    scheduler_seed = dict(config)
    scheduler_seed["max-num-seqs"] = seqs
    scheduler_seed["max-num-batched-tokens"] = batch

    scheduler_config = dict(config)
    scheduler_config.pop("max-num-seqs", None)
    scheduler_config.pop("max-num-batched-tokens", None)
    Path(args.output_config).write_text(
        yaml.safe_dump(
            scheduler_config,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    Path(args.serve_params).write_text(
        json.dumps(_build_serve_params(scheduler_seed), indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.bench_params).write_text(
        json.dumps(
            [
                {
                    "_benchmark_name": "selected_concurrency",
                    "random_input_len": args.input_tokens,
                    "random_output_len": args.output_tokens,
                    "max_concurrency": concurrency,
                    "num_prompts": _num_prompts_for_concurrency(concurrency),
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared scheduler sweep for max_concurrency={concurrency}")
    print(f"Scheduler seed max-num-seqs={seqs}")
    print(f"Scheduler seed max-num-batched-tokens={batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
