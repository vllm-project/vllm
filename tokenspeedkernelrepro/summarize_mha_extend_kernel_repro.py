#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize TokenSpeed native-extend vs AITER repro logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchRow:
    workload: str
    backend: str
    ms: float
    throughput: float
    relative_to_aiter: float
    throughput_name: str


@dataclass(frozen=True)
class CompareRow:
    workload: str
    name: str
    allclose: str
    max_abs: str
    mean_abs: str


def _csv_rows_after(lines: list[str], start_idx: int) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in lines[start_idx + 1 :]:
        raw = raw.strip()
        if not raw or raw.startswith("=="):
            break
        if raw.count(",") != 3:
            break
        rows.append(next(csv.reader([raw])))
    return rows


def _parse_compare_line(workload: str, line: str) -> CompareRow | None:
    if " allclose=" not in line:
        return None
    name, rest = line.split(" ", 1)
    fields = dict(item.split("=", 1) for item in rest.split() if "=" in item)
    if not {"allclose", "max_abs", "mean_abs"} <= fields.keys():
        return None
    return CompareRow(
        workload=workload,
        name=name,
        allclose=fields["allclose"],
        max_abs=fields["max_abs"],
        mean_abs=fields["mean_abs"],
    )


def parse_log(path: Path) -> tuple[list[BenchRow], list[CompareRow]]:
    lines = path.read_text().splitlines()
    workload = path.stem
    bench_rows: list[BenchRow] = []
    compare_rows: list[CompareRow] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "backend,ms,rows_per_s,relative_to_aiter":
            for fields in _csv_rows_after(lines, idx):
                bench_rows.append(
                    BenchRow(
                        workload=workload,
                        backend=fields[0],
                        ms=float(fields[1]),
                        throughput=float(fields[2]),
                        relative_to_aiter=float(fields[3]),
                        throughput_name="rows/s",
                    )
                )
        elif stripped == "backend,ms,tokens_per_s,relative_to_aiter":
            for fields in _csv_rows_after(lines, idx):
                bench_rows.append(
                    BenchRow(
                        workload=workload,
                        backend=fields[0],
                        ms=float(fields[1]),
                        throughput=float(fields[2]),
                        relative_to_aiter=float(fields[3]),
                        throughput_name="tokens/s",
                    )
                )
        else:
            parsed = _parse_compare_line(workload, stripped)
            if parsed is not None:
                compare_rows.append(parsed)

    return bench_rows, compare_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def write_summary(bench_rows: list[BenchRow], compare_rows: list[CompareRow]) -> str:
    lines: list[str] = []
    lines.append("# TokenSpeed Native Extend vs AITER Summary")
    lines.append("")
    lines.append(
        "This report compares TokenSpeed native extend paths against "
        "`ROCM_AITER_UNIFIED_ATTN`/AITER unified attention only."
    )
    lines.append(
        "Sliding-window workloads use the corrected convention: workload names "
        "report the vLLM semantic window, TokenSpeed receives `sliding_window - 1`, "
        "and AITER receives `window_size=(sliding_window - 1, 0)`."
    )
    lines.append("")

    if bench_rows:
        lines.append("## Benchmarks")
        lines.append("")
        lines.append(
            "| Workload | Backend/path | ms | Throughput | Relative to AITER |"
        )
        lines.append("|---|---|---:|---:|---:|")
        for row in bench_rows:
            lines.append(
                f"| `{row.workload}` | `{row.backend}` | `{row.ms:.4f}` | "
                f"`{row.throughput:.1f} {row.throughput_name}` | "
                f"`{row.relative_to_aiter:.3f}` |"
            )
        lines.append("")
        lines.append(
            "`relative_to_aiter > 1.000` means slower than AITER for the same "
            "synthetic request shape."
        )
        lines.append("")

    if compare_rows:
        lines.append("## Output Checks")
        lines.append("")
        lines.append("| Workload | Comparison | allclose | max abs | mean abs |")
        lines.append("|---|---|---:|---:|---:|")
        for row in compare_rows:
            lines.append(
                f"| `{row.workload}` | `{row.name}` | `{row.allclose}` | "
                f"`{row.max_abs}` | `{row.mean_abs}` |"
            )
        lines.append("")

    lines.append("## Kernel-Author Target")
    lines.append("")
    lines.append(
        "Optimize TokenSpeed native extend so the TokenSpeed rows approach or beat "
        "`aiter_unified` for the same full-attention and corrected sliding-window "
        "request shapes."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    bench_rows: list[BenchRow] = []
    compare_rows: list[CompareRow] = []
    for path in args.logs:
        parsed_bench, parsed_compare = parse_log(path)
        bench_rows.extend(parsed_bench)
        compare_rows.extend(parsed_compare)

    if not bench_rows:
        raise SystemExit("No benchmark CSV sections found in logs.")

    summary = write_summary(bench_rows, compare_rows)
    if args.out is not None:
        args.out.write_text(summary + "\n")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
