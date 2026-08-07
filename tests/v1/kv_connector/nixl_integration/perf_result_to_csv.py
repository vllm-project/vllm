# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combine a directory of `vllm bench serve --save-result` JSON files
(one per concurrency level) into a single CSV, using the same column
layout as the existing experiments/*/result.csv and */gpu_gpu.csv files
in this project, so plot_benchmark.py can be pointed at any arm.

Usage:
    python perf_result_to_csv.py --results-dir <dir> --glob "label_c*.json" \
        --out <dir>/label.csv
"""

import argparse
import csv
import glob
import json
import os

# Output column order, matching experiments/twoGPUs/shot_1/gpu_gpu.csv
COLUMNS = [
    "Successful requests",
    "Failed requests",
    "Maximum request concurrency",
    "Benchmark duration (s)",
    "Total input tokens",
    "Total generated tokens",
    "Request throughput (req/s)",
    "Output token throughput (tok/s)",
    "Peak output token throughput (tok/s)",
    "Peak concurrent requests",
    "Total token throughput (tok/s)",
    "Mean TTFT (ms)",
    "Median TTFT (ms)",
    "P99 TTFT (ms)",
    "Mean TPOT (ms)",
    "Median TPOT (ms)",
    "P99 TPOT (ms)",
    "Mean ITL (ms)",
    "Median ITL (ms)",
    "P99 ITL (ms)",
]


def row_from_result(result: dict) -> dict:
    return {
        "Successful requests": result.get("completed"),
        "Failed requests": result.get("failed"),
        "Maximum request concurrency": result.get("max_concurrency"),
        "Benchmark duration (s)": result.get("duration"),
        "Total input tokens": result.get("total_input_tokens"),
        "Total generated tokens": result.get("total_output_tokens"),
        "Request throughput (req/s)": result.get("request_throughput"),
        "Output token throughput (tok/s)": result.get("output_throughput"),
        "Peak output token throughput (tok/s)": result.get("max_output_tokens_per_s"),
        "Peak concurrent requests": result.get("max_concurrent_requests"),
        "Total token throughput (tok/s)": result.get("total_token_throughput"),
        "Mean TTFT (ms)": result.get("mean_ttft_ms"),
        "Median TTFT (ms)": result.get("median_ttft_ms"),
        "P99 TTFT (ms)": result.get("p99_ttft_ms"),
        "Mean TPOT (ms)": result.get("mean_tpot_ms"),
        "Median TPOT (ms)": result.get("median_tpot_ms"),
        "P99 TPOT (ms)": result.get("p99_tpot_ms"),
        "Mean ITL (ms)": result.get("mean_itl_ms"),
        "Median ITL (ms)": result.get("median_itl_ms"),
        "P99 ITL (ms)": result.get("p99_itl_ms"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--glob",
        type=str,
        default="*.json",
        help="Glob (relative to --results-dir) matching one JSON result "
        "file per concurrency level.",
    )
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.glob)))
    if not paths:
        raise SystemExit(
            f"No result files matched {args.glob!r} in {args.results_dir!r}"
        )

    rows = []
    for path in paths:
        with open(path) as f:
            result = json.load(f)
        rows.append(row_from_result(result))

    # Sort by concurrency so the CSV reads the same way as the manual ones.
    rows.sort(key=lambda r: r["Maximum request concurrency"] or 0)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
