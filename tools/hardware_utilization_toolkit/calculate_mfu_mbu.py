#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from typing import Dict, List


def load_kernel_summary(path: str) -> List[Dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_mfu_mbu(rows: List[Dict]) -> Dict[str, float]:
    """
    rows: from ncu_kernel_summary_*.csv for a single workload.
    Uses NSys time fraction + NCU MFU/MBU per kernel.
    """
    # Sum time across all kernels (ms)
    total_time_ms = 0.0
    for r in rows:
        t = float(r["nsys_time_ms"])
        total_time_ms += t

    if total_time_ms == 0.0:
        return {"MFU": 0.0, "MBU": 0.0}

    mfu_num = 0.0
    mbu_num = 0.0

    for r in rows:
        t = float(r["nsys_time_ms"])
        sm = r["ncu_sm_pct_of_peak"]
        dram = r["ncu_dram_pct_of_peak"]
        if sm == "" or dram == "":
            continue
        sm = float(sm) / 100.0
        dram = float(dram) / 100.0

        weight = t / total_time_ms
        mfu_num += sm * weight
        mbu_num += dram * weight

    return {"MFU": mfu_num, "MBU": mbu_num}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", help="ncu_kernel_summary_full.csv")
    ap.add_argument("--prefill", help="ncu_kernel_summary_prefill.csv")
    ap.add_argument("--decode", help="ncu_kernel_summary_decode.csv")
    args = ap.parse_args()

    workloads = {}
    if args.full:
        workloads["full"] = args.full
    if args.prefill:
        workloads["prefill"] = args.prefill
    if args.decode:
        workloads["decode"] = args.decode

    if not workloads:
        print("No workloads provided.")
        return

    print("Workload,MFU,MBU")
    for name, path in workloads.items():
        rows = load_kernel_summary(path)
        res = compute_mfu_mbu(rows)
        print(f"{name},{res['MFU']:.4f},{res['MBU']:.4f}")


if __name__ == "__main__":
    main()
