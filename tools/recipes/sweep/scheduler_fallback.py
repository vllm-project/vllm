#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Preserve an SLA-feasible concurrency baseline when scheduler tuning fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def fallback(sweep_dir: Path) -> None:
    source = sweep_dir / "parallel-layout-config.yml"
    concurrency_json = sweep_dir / "concurrency-recommendation.json"
    target = sweep_dir / "recommended-config.yml"
    report_path = sweep_dir / "recommendation.json"

    result = json.loads(concurrency_json.read_text(encoding="utf-8"))
    selected = result.get("recommended")
    if not isinstance(selected, dict) or not isinstance(
        selected.get("max_concurrency"), int
    ):
        raise RuntimeError("Cannot fall back: no validated concurrency recommendation")
    if result.get("status") not in ("sla_feasible", "pass"):
        raise RuntimeError("Cannot fall back: concurrency is not SLA-feasible")

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Invalid parallel-layout-config.yml")
    config.pop("max-num-seqs", None)
    config.pop("max-num-batched-tokens", None)
    target.write_text(
        "# Scheduler fallback: previously validated concurrency-stage baseline.\n"
        + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    prior = {}
    if report_path.is_file():
        from contextlib import suppress

        with suppress(ValueError, OSError):
            prior = json.loads(report_path.read_text(encoding="utf-8"))
    report = {
        "status": "fallback_concurrency_baseline",
        "selection_objective": "preserve_prior_sla_feasible_config",
        "recommended": {
            "tensor_parallel_size": config.get("tensor-parallel-size", 1),
            "data_parallel_size": config.get("data-parallel-size", 1),
            "max_num_seqs": None,
            "max_num_batched_tokens": None,
            "vllm_default_parameters": ["max_num_seqs", "max_num_batched_tokens"],
        },
        "measured": None,
        "baseline_source": "concurrency-recommendation.json",
        "baseline_measured": result.get("measured"),
        "scheduler_attempt_status": prior.get("status", "sweep_failed"),
        "candidates": prior.get("candidates", []),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Scheduler tuning failed: using SLA-feasible concurrency baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()
    fallback(args.sweep_dir)


if __name__ == "__main__":
    main()
