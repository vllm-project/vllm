#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-by-side summary of HiSparse P/D A/B benchmark results.

Reads ``<root>/<arm>/conc*_isl*_osl*.json`` files written by pd_bench.sh,
joins arms on (concurrency, isl, osl), and prints throughput / latency /
completion columns per arm plus the hisparse/gpu-kv throughput ratio.

Usage: summarize_ab.py --root bench_results/hisparse_pd_ab_<ts>
"""

import argparse
import glob
import json
import os


def load_arm(arm_dir):
    points = {}
    for path in glob.glob(os.path.join(arm_dir, "*.json")):
        if path.endswith(".pytorch.json"):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        meta = d.get("metadata", {}) or {}
        try:
            key = (
                int(meta.get("concurrency", 0)),
                int(meta.get("input_len", 0)),
                int(meta.get("output_len", 0)),
            )
        except (TypeError, ValueError):
            continue
        if d.get("completed", 0) == 0:
            continue
        points[key] = {
            "out_tok_s": d.get("output_throughput") or 0.0,
            "ttft_mean": d.get("mean_ttft_ms") or 0.0,
            "ttft_p99": d.get("p99_ttft_ms") or 0.0,
            "itl_med": d.get("median_itl_ms") or 0.0,
            "completed": d.get("completed", 0),
            "failed": d.get("failed", 0),
        }
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="A/B results root directory")
    args = parser.parse_args()

    arms = sorted(
        d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))
    )
    if not arms:
        print(f"No arm directories under {args.root}")
        return
    arm_points = {arm: load_arm(os.path.join(args.root, arm)) for arm in arms}
    keys = sorted(set().union(*[set(p) for p in arm_points.values()]))

    hdr = [("conc", 7), ("isl", 7), ("osl", 7)]
    for arm in arms:
        hdr += [(f"{arm}:tok/s", 14), ("ttft99", 10), ("done/total", 10)]
    print(" ".join(f"{c:>{w}}" for c, w in hdr))

    for key in keys:
        conc, isl, osl = key
        row = [(str(conc), 7), (str(isl), 7), (str(osl), 7)]
        ratio = ""
        for arm in arms:
            p = arm_points[arm].get(key)
            if p is None:
                row += [("-", 14), ("-", 10), ("-", 10)]
                continue
            row += [
                (f"{p['out_tok_s']:.0f}", 14),
                (f"{p['ttft_p99'] / 1000:.1f}s" if p["ttft_p99"] else "-", 10),
                (f"{p['completed']}/{p['completed'] + p['failed']}", 10),
            ]
        if {"gpu-kv", "hisparse"}.issubset(set(arms)):
            g = arm_points["gpu-kv"].get(key)
            h = arm_points["hisparse"].get(key)
            if g and h and g["out_tok_s"] > 0:
                ratio = f"  hisparse/gpu-kv: {h['out_tok_s'] / g['out_tok_s']:.2f}x"
        print(" ".join(f"{c:>{w}}" for c, w in row) + ratio)


if __name__ == "__main__":
    main()
