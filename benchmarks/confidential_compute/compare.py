#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a CC-off and a CC-on ``cc_bridge_bench.py`` run as Markdown.

Usage::

    python benchmarks/confidential_compute/compare.py OFF.json ON.json [-o out.md]

Rows are joined on their ``key``; the ratio column is ON / OFF.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# test -> (metric, header, higher_is_better)
METRICS: dict[str, list[tuple[str, str, bool]]] = {
    "semantics": [("call_return_us", "copy call returns (us)", False)],
    "bandwidth": [
        ("latency_median_us", "latency (us)", False),
        ("bandwidth_gbps", "bandwidth (GB/s)", True),
    ],
    "concurrency": [("aggregate_gbps", "aggregate (GB/s)", True)],
    "decode": [
        ("step_us", "step (us)", False),
        ("speedup_vs_sync", "speedup vs sync", True),
        ("gpu_busy_fraction", "GPU busy", True),
    ],
}


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def compare(off: dict[str, Any], on: dict[str, Any]) -> str:
    lines = [
        "# CC bridge microbench: CC off vs CC on",
        "",
        (
            f"- OFF: {off['env']['gpu']} on `{off['env']['hostname']}` "
            f"({off['env']['timestamp']}, cc={off['env']['cc']})"
        ),
        (
            f"- ON:  {on['env']['gpu']} on `{on['env']['hostname']}` "
            f"({on['env']['timestamp']}, cc={on['env']['cc']})"
        ),
        "",
    ]
    if off["env"]["gpu"] != on["env"]["gpu"]:
        lines += ["> warning: the two runs used different GPUs.", ""]
    if on.get("findings"):
        lines += ["## Findings (CC on)", ""]
        lines += [f"- {f}" for f in on["findings"]] + [""]

    for test, metrics in METRICS.items():
        off_rows = {r["key"]: r for r in off.get(test, [])}
        on_rows = {r["key"]: r for r in on.get(test, [])}
        keys = [k for k in off_rows if k in on_rows]
        keys += [k for k in on_rows if k not in off_rows]
        if not keys:
            continue
        header = ["case"]
        for _, name, _ in metrics:
            header += [f"{name} OFF", f"{name} ON", "ON/OFF"]
        if test == "semantics":
            header += ["verdict OFF", "verdict ON"]
        lines += [
            f"## {test}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        for k in keys:
            a, b = off_rows.get(k, {}), on_rows.get(k, {})
            cells = [k]
            for metric, _, _ in metrics:
                va, vb = a.get(metric), b.get(metric)
                ratio = vb / va if va and vb is not None else None
                cells += [
                    _fmt(va),
                    _fmt(vb),
                    "-" if ratio is None else f"{ratio:.2f}x",
                ]
            if test == "semantics":
                cells += [a.get("verdict", "-"), b.get("verdict", "-")]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("off", help="JSON from a CC-off run")
    p.add_argument("on", help="JSON from a CC-on run")
    p.add_argument("-o", "--output", help="also write the Markdown here")
    args = p.parse_args(argv)
    with open(args.off) as f:
        off = json.load(f)
    with open(args.on) as f:
        on = json.load(f)
    if off.get("schema_version") != on.get("schema_version"):
        print(
            "warning: runs use different schema versions "
            f"({off.get('schema_version')} vs {on.get('schema_version')})",
            file=sys.stderr,
        )
    for name, run, want in (("OFF", off, "off"), ("ON", on, "on")):
        if run["env"]["cc"] not in (want, "unknown"):
            print(
                f"warning: {name} run is labelled cc={run['env']['cc']}",
                file=sys.stderr,
            )
    md = compare(off, on)
    print(md)
    if args.output:
        with open(args.output, "w") as f:
            f.write(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
