#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive ground-truth ROCm AITER test tolerances (non-circular).

Requires a ROCm GPU with AITER installed. Writes JSONL per path and a summary
JSON with recommended ``atol`` values derived from

    atol_full = max_e( |ref_e - golden_e| + kernel_budget_e - rtol * |ref_e| )

Usage (from the vLLM repo root)::

    python tests/kernels/attention/derive_rocm_aiter_tolerances.py --run all
    python tests/kernels/attention/derive_rocm_aiter_tolerances.py \\
        --aggregate /path/to/apriori_fa_all.jsonl /path/to/apriori_mla_all.jsonl

See ``rocm_aiter_tolerances.py`` for how committed constants relate to the
derived ``atol_full`` values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python tests/kernels/attention/derive_rocm_aiter_tolerances.py``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.kernels.attention.tolerance_derivation.core import (  # noqa: E402
    DEFAULT_MARGIN,
    audit_record,
    load_jsonl,
    print_report,
    summarize_all,
    summaries_to_json,
)


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parent / "tolerance_derivation" / "out"


def _run_paths(out_dir: Path, seeds: int, which: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # Child runners parse ``sys.argv``; isolate them from this CLI.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]

    try:
        if which in ("all", "fa"):
            from tests.kernels.attention.tolerance_derivation.run_fa import main as run_fa

            path = out_dir / "apriori_fa_all.jsonl"
            run_fa(out=str(path), seeds=seeds)
            written.append(path)

        if which in ("all", "mla"):
            from tests.kernels.attention.tolerance_derivation.run_mla import main as run_mla

            path = out_dir / "apriori_mla_all.jsonl"
            run_mla(out=str(path), seeds=seeds, only="all")
            written.append(path)

        if which in ("all", "unified"):
            from tests.kernels.attention.tolerance_derivation.run_unified import (
                main as run_unified,
            )

            path = out_dir / "apriori_unified.jsonl"
            run_unified(out=str(path), seeds=seeds)
            written.append(path)

        if which in ("all", "criterion"):
            from tests.kernels.attention.tolerance_derivation.run_fa_criterion import (
                main as run_criterion,
            )

            path = out_dir / "fa_criterion.jsonl"
            run_criterion(out=str(path), seeds=seeds)
            written.append(path)
    finally:
        sys.argv = saved_argv

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        choices=("all", "fa", "mla", "unified", "criterion"),
        help="Run live derivation on ROCm GPU (requires AITER)",
    )
    ap.add_argument(
        "--aggregate",
        nargs="+",
        type=Path,
        metavar="JSONL",
        help="Aggregate existing JSONL records instead of running kernels",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Directory for JSONL output when --run is used",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Write summary JSON (default: out-dir/summary.json)",
    )
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Safety factor applied to max atol_full",
    )
    args = ap.parse_args()

    jsonl_paths: list[Path] = list(args.aggregate or [])
    if args.run:
        jsonl_paths.extend(_run_paths(args.out_dir, args.seeds, args.run))

    if not jsonl_paths:
        ap.error("Provide --run and/or --aggregate JSONL paths")

    records: list[dict] = []
    for path in jsonl_paths:
        records.extend(load_jsonl(path))

    if not records:
        print("No records loaded.", file=sys.stderr)
        return 1

    summaries = summarize_all(records, margin=args.margin)
    print_report(summaries)

    failed = [
        (r.get("label", r.get("cfg", "?")), audit_record(r).reasons)
        for r in records
        if not audit_record(r).passed
    ]
    if failed:
        print(f"\n=== kernel audit failures: {len(failed)}/{len(records)} ===")
        for label, reasons in failed[:20]:
            print(f"  {label}: {'; '.join(reasons)}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    summary_path = args.summary or (args.out_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries_to_json(summaries), indent=2) + "\n"
    )
    print(f"\nWrote summary to {summary_path}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
