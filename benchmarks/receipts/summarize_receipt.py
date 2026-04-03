# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize a receipt JSON produced by benchmarks.receipts.run_receipt."""

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


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "null"
    return f"{float(x):.{nd}f}"


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python -m benchmarks.receipts.summarize_receipt <receipt.json>",
            file=sys.stderr,
        )
        return 2

    p = Path(sys.argv[1])
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        print(f"Error: Receipt file not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in receipt file: {e}", file=sys.stderr)
        return 1

    duration_s = _g(data, "duration_s")
    rc = _g(data, "result", "returncode")
    cmd = _g(data, "command", "argv_shell_escaped")
    git_head = _g(data, "env", "git_head")

    tel = _g(data, "telemetry") or {}
    backend = tel.get("backend")
    energy_j = tel.get("energy_joules")
    summary = tel.get("summary") or {}

    print(f"receipt: {p}")
    print(f"returncode: {rc}")
    print(f"duration_s: {_fmt(duration_s, 3)}")
    print(f"git_head: {git_head}")
    print(f"cmd: {cmd}")
    print("")
    print(f"telemetry_backend: {backend}")
    print(f"energy_joules: {_fmt(energy_j, 3)}")
    print(f"avg_gpu_util_pct: {_fmt(summary.get('avg_gpu_util_pct'), 2)}")
    print(f"avg_power_w: {_fmt(summary.get('avg_power_w'), 2)}")
    print(f"avg_temp_c: {_fmt(summary.get('avg_temp_c'), 2)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
