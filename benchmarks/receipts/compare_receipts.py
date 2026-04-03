# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare two receipts (before/after) and print ratios/deltas.

This does not assume a specific acceptance currency. It prints the raw deltas so
teams can decide what is meaningful for their boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _g(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or abs(float(b)) < 1e-9:
        return None
    return float(a) / float(b)


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "null"
    return f"{float(x):.{nd}f}"


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python -m benchmarks.receipts.compare_receipts "
            "<before.json> <after.json>",
            file=sys.stderr,
        )
        return 2

    try:
        a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input file: {e}", file=sys.stderr)
        return 1

    dur_a = _g(a, "duration_s")
    dur_b = _g(b, "duration_s")

    e_a = _g(a, "telemetry", "energy_joules")
    e_b = _g(b, "telemetry", "energy_joules")

    util_a = _g(a, "telemetry", "summary", "avg_gpu_util_pct")
    util_b = _g(b, "telemetry", "summary", "avg_gpu_util_pct")

    pwr_a = _g(a, "telemetry", "summary", "avg_power_w")
    pwr_b = _g(b, "telemetry", "summary", "avg_power_w")

    print("before:")
    print(
        "  "
        f"duration_s={_fmt(dur_a, 3)}  "
        f"energy_j={_fmt(e_a, 3)}  "
        f"util%={_fmt(util_a, 2)}  "
        f"power_w={_fmt(pwr_a, 2)}"
    )
    print("after:")
    print(
        "  "
        f"duration_s={_fmt(dur_b, 3)}  "
        f"energy_j={_fmt(e_b, 3)}  "
        f"util%={_fmt(util_b, 2)}  "
        f"power_w={_fmt(pwr_b, 2)}"
    )
    print("")
    print("ratios (after / before):")
    print(f"  duration={_fmt(_ratio(dur_b, dur_a), 4)}")
    print(f"  energy={_fmt(_ratio(e_b, e_a), 4)}")
    print(f"  util%={_fmt(_ratio(util_b, util_a), 4)}")
    print(f"  power_w={_fmt(_ratio(pwr_b, pwr_a), 4)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
