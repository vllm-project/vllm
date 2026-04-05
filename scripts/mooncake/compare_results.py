#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare benchmark results across offloading backends.

Usage:
    python compare_results.py <result_dir> [--prefix PREFIX]

Examples:
    # Compare multi-turn results (mt_baseline.json, mt_mooncake.json, ...)
    python compare_results.py ./bench_results --prefix mt_

    # Compare single-turn results (baseline.json, mooncake.json, ...)
    python compare_results.py ./bench_results
"""

import argparse
import json
import os

ALL_BACKENDS = ["baseline", "native", "simple", "mooncake"]

METRICS = [
    ("Output throughput (tok/s)", "output_throughput", ".1f"),
    ("Request throughput (req/s)", "request_throughput", ".3f"),
    ("Mean TTFT (ms)", "mean_ttft_ms", ".2f"),
    ("Median TTFT (ms)", "median_ttft_ms", ".2f"),
    ("P99 TTFT (ms)", "p99_ttft_ms", ".2f"),
    ("Mean TPOT (ms)", "mean_tpot_ms", ".2f"),
    ("Mean ITL (ms)", "mean_itl_ms", ".2f"),
    ("Mean E2EL (ms)", "mean_e2el_ms", ".2f"),
]

COL_W = 14


def pct(old, new):
    return (new - old) / old * 100 if old else 0.0


def load_results(result_dir, prefix):
    results = {}
    for label in ALL_BACKENDS:
        path = os.path.join(result_dir, f"{prefix}{label}.json")
        if os.path.isfile(path):
            with open(path) as f:
                results[label] = json.load(f)
    return results


def print_comparison(results, title):
    if not results:
        print("\nNo result files found — nothing to compare.\n")
        return

    labels = list(results.keys())

    print()
    print("=" * (32 + COL_W * len(labels)))
    print(f"  {title}  (delta vs baseline: + = regression, - = improvement)")
    print("=" * (32 + COL_W * len(labels)))

    header = f"  {'Metric':<30}"
    for lb in labels:
        header += f"  {lb:>{COL_W - 2}}"
    print(header)

    sep = f"  {'-' * 30}"
    for _ in labels:
        sep += f"  {'-' * (COL_W - 2)}"
    print(sep)

    baseline = results.get("baseline")

    for metric_label, key, fmt in METRICS:
        # Skip metrics not present in any result
        if all(results[lb].get(key) is None for lb in labels):
            continue
        row = f"  {metric_label:<30}"
        for lb in labels:
            val = results[lb].get(key)
            if val is None:
                row += f"  {'N/A':>{COL_W - 2}}"
                continue
            cell = f"{val:{fmt}}"
            if baseline and lb != "baseline":
                bval = baseline.get(key, 0)
                delta = pct(bval, val)
                cell += f" ({delta:+.1f}%)"
            row += f"  {cell:>{COL_W - 2}}"
        print(row)

    print("=" * (32 + COL_W * len(labels)))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results across offloading backends."
    )
    parser.add_argument("result_dir", help="Directory containing result JSONs")
    parser.add_argument(
        "--prefix",
        default="",
        help="Filename prefix (e.g. 'mt_' for multi-turn results)",
    )
    args = parser.parse_args()

    results = load_results(args.result_dir, args.prefix)
    title = "MULTI-TURN COMPARISON" if args.prefix == "mt_" else "COMPARISON"
    print_comparison(results, title)


if __name__ == "__main__":
    main()
