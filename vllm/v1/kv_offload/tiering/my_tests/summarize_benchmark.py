#!/usr/bin/env python3
"""Summarize baseline and Tiering benchmark_serving JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def change(current: float, baseline: float) -> float:
    return (current / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("tiering", type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    tiering = load(args.tiering)
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
