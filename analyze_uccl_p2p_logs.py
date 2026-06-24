#!/usr/bin/env python3
"""Parse VLLM_UCCL_P2P_TIMING=1 logs and print micro-benchmark summary."""

import argparse
import re
import sys
from collections import defaultdict

TIMING_RE = re.compile(
    r"\[uccl_p2p(?:_worker|_wrapper)?_timing\]\s+([^:]+):\s+([\d.]+)\s*us"
)


def percentile(sorted_vals, p):
    n = len(sorted_vals)
    idx = int(n * p)
    idx = min(idx, n - 1)
    return sorted_vals[idx]


def main():
    parser = argparse.ArgumentParser(description="Summarize UCCL P2P timing logs")
    parser.add_argument("log_files", nargs="+", help="log files to parse")
    parser.add_argument("--top", "-n", type=int, default=30)
    args = parser.parse_args()

    values = defaultdict(list)
    for path in args.log_files:
        try:
            with open(path) as f:
                for line in f:
                    m = TIMING_RE.search(line)
                    if m:
                        values[m.group(1).strip()].append(float(m.group(2)))
        except FileNotFoundError:
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not values:
        print("No [uccl_p2p_timing] lines found.")
        return

    rows = []
    for label, vals in values.items():
        s = sorted(vals)
        n = len(vals)
        p50 = percentile(s, 0.50)
        p99 = percentile(s, 0.99)
        mean = sum(vals) / n
        rows.append((label, n, p50, p99, max(vals), mean))

    # Sort by total estimated time (count * mean)
    rows.sort(key=lambda x: x[1] * x[5], reverse=True)

    print("| label | count | p50 (us) | p99 (us) | max (us) | mean (us) |")
    print("|---|---:|---:|---:|---:|---:|")
    for label, n, p50, p99, mx, mean in rows[: args.top]:
        print(f"| {label} | {n} | {p50:.1f} | {p99:.1f} | {mx:.1f} | {mean:.1f} |")


if __name__ == "__main__":
    main()
