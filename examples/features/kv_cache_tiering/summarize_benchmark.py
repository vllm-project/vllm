#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize baseline and Tiering benchmark_serving JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def change(current: float, baseline: float) -> float:
    return (current / baseline - 1.0) * 100.0


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(data: Any) -> dict[str, float | int]:
    if isinstance(data, dict):
        return data
    if not isinstance(data, list) or not data:
        raise ValueError("benchmark JSON must be a non-empty object or request list")

    rows = [
        row
        for row in data
        if row.get("start_time_ms") is not None
        and row.get("latency_ms") is not None
        and row.get("ttft_ms") is not None
        and row.get("tpot_ms") is not None
    ]
    if not rows:
        raise ValueError("benchmark JSON contains no completed request rows")

    start_ms = min(float(row["start_time_ms"]) for row in rows)
    end_ms = max(float(row["start_time_ms"]) + float(row["latency_ms"]) for row in rows)
    runtime_sec = (end_ms - start_ms) / 1000.0
    input_tokens = sum(int(row["input_num_tokens"]) for row in rows)
    output_tokens = sum(int(row["output_num_tokens"]) for row in rows)
    ttft_values = [float(row["ttft_ms"]) for row in rows]
    tpot_values = [float(row["tpot_ms"]) for row in rows]

    return {
        "completed": len(rows),
        "failed": len(data) - len(rows),
        "request_throughput": len(rows) / runtime_sec,
        "total_token_throughput": (input_tokens + output_tokens) / runtime_sec,
        "mean_ttft_ms": statistics.fmean(ttft_values),
        "p50_ttft_ms": percentile(ttft_values, 0.50),
        "p90_ttft_ms": percentile(ttft_values, 0.90),
        "p99_ttft_ms": percentile(ttft_values, 0.99),
        "mean_tpot_ms": statistics.fmean(tpot_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("tiering", type=Path)
    args = parser.parse_args()

    baseline = summarize(load(args.baseline))
    tiering = summarize(load(args.tiering))
    fields = (
        ("total_token_throughput", "tokens/s"),
        ("request_throughput", "requests/s"),
        ("mean_ttft_ms", "ms"),
        ("mean_tpot_ms", "ms"),
    )
    for field, unit in fields:
        base_value = float(baseline[field])
        tier_value = float(tiering[field])
        print(
            f"{field}: baseline={base_value:.3f} {unit}, "
            f"tiering={tier_value:.3f} {unit}, "
            f"change={change(tier_value, base_value):+.1f}%"
        )

    print(
        f"completed: baseline={baseline.get('completed')} "
        f"tiering={tiering.get('completed')}"
    )
    print(f"failed: baseline={baseline.get('failed')} tiering={tiering.get('failed')}")


if __name__ == "__main__":
    main()
