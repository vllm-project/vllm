#!/usr/bin/env python3
"""Summarize the isolated MHA extend kernel-author repro logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtendRow:
    workload: str
    case: str
    ms: float
    rows_per_s: float
    relative_to_native: float


@dataclass(frozen=True)
class MixedRow:
    workload: str
    backend: str
    ms: float
    tokens_per_s: float
    relative_to_aiter: float


@dataclass(frozen=True)
class AccuracyRow:
    workload: str
    max_abs: float
    mean_abs: float


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


def parse_log(path: Path) -> tuple[list[ExtendRow], list[MixedRow], list[AccuracyRow]]:
    lines = path.read_text().splitlines()
    extend_rows: list[ExtendRow] = []
    mixed_rows: list[MixedRow] = []
    accuracy_rows: list[AccuracyRow] = []
    workload = path.stem

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "case,ms,rows_per_s,relative_to_native":
            for fields in _csv_rows_after(lines, idx):
                extend_rows.append(
                    ExtendRow(
                        workload=workload,
                        case=fields[0],
                        ms=float(fields[1]),
                        rows_per_s=float(fields[2]),
                        relative_to_native=float(fields[3]),
                    )
                )
        elif stripped == "backend,ms,tokens_per_s,relative_to_aiter":
            for fields in _csv_rows_after(lines, idx):
                mixed_rows.append(
                    MixedRow(
                        workload=workload,
                        backend=fields[0],
                        ms=float(fields[1]),
                        tokens_per_s=float(fields[2]),
                        relative_to_aiter=float(fields[3]),
                    )
                )
        elif stripped.startswith("native_vs_decomposed_last_run "):
            parts = dict(
                item.split("=", 1)
                for item in stripped.split()[1:]
                if "=" in item
            )
            accuracy_rows.append(
                AccuracyRow(
                    workload=workload,
                    max_abs=float(parts["max_abs_diff"]),
                    mean_abs=float(parts["mean_abs_diff"]),
                )
            )

    return extend_rows, mixed_rows, accuracy_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def write_summary(
    extend_rows: list[ExtendRow],
    mixed_rows: list[MixedRow],
    accuracy_rows: list[AccuracyRow],
) -> str:
    lines: list[str] = []
    lines.append("# MHA Extend Kernel Repro Summary")
    lines.append("")
    lines.append(
        "This repro isolates TokenSpeed MHA native extend behavior without "
        "modifying `tokenspeed_kernel_amd`."
    )
    lines.append("")

    if extend_rows:
        lines.append("## Native Extend vs Decode-Decomposed Extend")
        lines.append("")
        lines.append(
            "| Workload | Case | ms | rows/s | Relative to native extend |"
        )
        lines.append("|---|---|---:|---:|---:|")
        for row in extend_rows:
            lines.append(
                f"| `{row.workload}` | `{row.case}` | `{row.ms:.4f}` | "
                f"`{row.rows_per_s:.1f}` | `{row.relative_to_native:.3f}` |"
            )
        lines.append("")
        lines.append(
            "For this table, `relative_to_native < 1.000` means the "
            "decode-decomposed alternative is faster than native extend."
        )
        lines.append("")

        lines.append("## Native Extend Slowdown Highlights")
        lines.append("")
        lines.append(
            "| Workload | Native extend ms | Best equivalent alternative | "
            "Alternative ms | Native / alternative |"
        )
        lines.append("|---|---:|---|---:|---:|")
        by_workload: dict[str, list[ExtendRow]] = {}
        for row in extend_rows:
            by_workload.setdefault(row.workload, []).append(row)
        for workload, rows in sorted(by_workload.items()):
            native = next((row for row in rows if row.case == "native_extend"), None)
            alternatives = [
                row
                for row in rows
                if row.case.startswith("decode_decomposed_")
            ]
            if native is None or not alternatives:
                continue
            best = min(alternatives, key=lambda row: row.ms)
            lines.append(
                f"| `{workload}` | `{native.ms:.4f}` | `{best.case}` | "
                f"`{best.ms:.4f}` | `{native.ms / best.ms:.3f}` |"
            )
        lines.append("")

    if accuracy_rows:
        lines.append("## Extend Equivalence Checks")
        lines.append("")
        lines.append("| Workload | max abs diff | mean abs diff |")
        lines.append("|---|---:|---:|")
        for row in accuracy_rows:
            lines.append(
                f"| `{row.workload}` | `{row.max_abs:.8g}` | "
                f"`{row.mean_abs:.8g}` |"
            )
        lines.append("")

    if mixed_rows:
        lines.append("## Mixed Batch vs ROCM_AITER_UNIFIED_ATTN")
        lines.append("")
        lines.append(
            "| Workload | Backend/path | ms | tokens/s | Relative to AITER |"
        )
        lines.append("|---|---|---:|---:|---:|")
        interesting = {
            "aiter_unified",
            "tokenspeed_safe",
            "tokenspeed_native_extend",
            "tokenspeed_native_extend_prefill_kernel",
            "tokenspeed_native_extend_all",
        }
        for row in mixed_rows:
            if row.backend not in interesting:
                continue
            lines.append(
                f"| `{row.workload}` | `{row.backend}` | `{row.ms:.4f}` | "
                f"`{row.tokens_per_s:.1f}` | `{row.relative_to_aiter:.3f}` |"
            )
        lines.append("")
        lines.append(
            "For this table, `relative_to_aiter > 1.000` means slower than "
            "`ROCM_AITER_UNIFIED_ATTN` for the same synthetic request mix."
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The `extend_full_q1` workload is the direct kernel-level case to "
        "send upstream when native MHA extend is slower than equivalent "
        "decode-decomposed extend work."
    )
    lines.append(
        "- Multi-token extend rows are included because native extend is not "
        "universally slower in isolation; the important gap is the mixed serving "
        "shape against `ROCM_AITER_UNIFIED_ATTN`."
    )
    lines.append(
        "- The current safe integration keeps native extend disabled by default "
        "because native-extend full-model runs previously regressed GSM8K."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    extend_rows: list[ExtendRow] = []
    mixed_rows: list[MixedRow] = []
    accuracy_rows: list[AccuracyRow] = []
    for path in args.logs:
        parsed_extend, parsed_mixed, parsed_accuracy = parse_log(path)
        extend_rows.extend(parsed_extend)
        mixed_rows.extend(parsed_mixed)
        accuracy_rows.extend(parsed_accuracy)

    if not extend_rows and not mixed_rows:
        raise SystemExit("No benchmark CSV sections found in logs.")

    summary = write_summary(extend_rows, mixed_rows, accuracy_rows)
    if args.out is not None:
        args.out.write_text(summary + "\n")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
