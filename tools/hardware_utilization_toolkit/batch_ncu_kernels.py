#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


# -------------------------------
# Data structure
# -------------------------------

@dataclass
class KernelInfo:
    name: str
    total_time_ms: float
    time_pct: float


# -------------------------------
# Helpers for NSys parsing
# -------------------------------

def _parse_duration_to_ms(s: str) -> float:
    """
    Parse strings like:
      '26.880 μs', '443.106 μs', '3.5 ms', '1.2 s'
    into milliseconds.
    """
    s = str(s).strip()
    if not s:
        return 0.0

    parts = s.split()
    if len(parts) != 2:
        return 0.0

    val_str, unit = parts
    try:
        val = float(val_str.replace(",", ""))
    except ValueError:
        return 0.0

    unit = unit.lower()
    if unit.startswith("μs") or unit.startswith("us"):
        # microseconds -> ms
        return val / 1000.0
    if unit.startswith("ns"):
        # nanoseconds -> ms
        return val / 1e6
    if unit.startswith("ms"):
        # already ms
        return val
    if unit.startswith("s"):
        # seconds -> ms
        return val * 1000.0
    return 0.0


def read_nsys_kernels_from_excel(
    xlsx_path: str,
    min_time_ms: float = 0.0,
    top_k: Optional[int] = 20,
) -> List[KernelInfo]:
    """
    Read an NSys Excel file with per-launch rows and aggregate by kernel name.

    Expected columns:
      - 'Kernel Name'
      - 'Kernel Duration' (e.g. '26.880 μs', '3.5 ms', etc.)
    """
    df = pd.read_excel(xlsx_path)

    if "Kernel Name" not in df.columns or "Kernel Duration" not in df.columns:
        print(
            f"[ERROR] Excel file {xlsx_path} must contain 'Kernel Name' and 'Kernel Duration' columns.",
            file=sys.stderr,
        )
        return []

    # Drop rows without kernel names
    df = df.dropna(subset=["Kernel Name"])

    # Convert duration string -> ms
    df["dur_ms"] = df["Kernel Duration"].apply(_parse_duration_to_ms)

    # Aggregate by kernel name
    grouped = df.groupby("Kernel Name", as_index=False)["dur_ms"].sum()
    grouped = grouped.sort_values("dur_ms", ascending=False)

    total_time_ms = grouped["dur_ms"].sum()
    if total_time_ms <= 0:
        print(f"[WARN] Total kernel time is zero in {xlsx_path}", file=sys.stderr)
        return []

    kernels: List[KernelInfo] = []
    for _, row in grouped.iterrows():
        name = str(row["Kernel Name"]).strip()
        t_ms = float(row["dur_ms"])
        if t_ms < min_time_ms:
            continue
        pct = (t_ms / total_time_ms) * 100.0
        kernels.append(KernelInfo(name=name, total_time_ms=t_ms, time_pct=pct))

    if top_k is not None and top_k > 0:
        kernels = kernels[:top_k]

    return kernels


def read_nsys_kernels_from_csv(
    csv_path: str,
    min_time_ms: float = 0.0,
    top_k: Optional[int] = 20,
) -> List[KernelInfo]:
    """
    Read an NSys cudaKernSummary CSV file (nsys stats --report cudaKernSummary).
    """
    kernels: List[KernelInfo] = []

    with open(csv_path, newline="") as f:
        # Filter out comment lines starting with '#'
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
        if not lines:
            return kernels

        headers = [h.strip() for h in lines[0].strip().split(",")]
        reader = csv.DictReader(lines[1:], fieldnames=headers)

        for row in reader:
            name = row.get("Name") or row.get("Kernel Name") or ""
            name = name.strip()
            if not name:
                continue

            time_str = row.get("Time (ms)") or row.get("Time (ns)") or ""
            pct_str = row.get("Time(%)") or row.get("Time (%)") or ""
            if not time_str or not pct_str:
                continue

            try:
                t_val = float(time_str)
                if "Time (ns)" in headers:
                    total_ms = t_val * 1e-6
                else:
                    total_ms = t_val
                pct = float(pct_str)
            except ValueError:
                continue

            if total_ms < min_time_ms:
                continue

            kernels.append(KernelInfo(name=name, total_time_ms=total_ms, time_pct=pct))

    kernels.sort(key=lambda k: k.time_pct, reverse=True)
    if top_k is not None and top_k > 0:
        kernels = kernels[:top_k]
    return kernels


def read_nsys_kernels(
    path: str,
    min_time_ms: float = 0.0,
    top_k: Optional[int] = 20,
) -> List[KernelInfo]:
    """
    Wrapper that supports:
      - Excel (.xlsx/.xls) from NSys
      - CSV cudaKernSummary
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return read_nsys_kernels_from_excel(path, min_time_ms=min_time_ms, top_k=top_k)
    else:
        return read_nsys_kernels_from_csv(path, min_time_ms=min_time_ms, top_k=top_k)


# -------------------------------
# Helpers for kernel names & filenames
# -------------------------------

def sanitize_for_filename(name: str) -> str:
    """
    Turn a kernel name into a safe (short) filename component.
    """
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    return base[:80]


def to_regex_pattern(full_name: str) -> str:
    """
    Build a robust regex pattern from a full demangled kernel name.

    Heuristics:
      - If it contains known stable substrings (fused_moe_kernel, FlashAttnFwdSm90, etc),
        use those directly.
      - Otherwise:
        * strip the trailing '(...)' parameter list
        * take the first ~120 chars
        * escape for regex
    """
    s = str(full_name)

    # 1) Known important substrings (you can extend this list)
    known_keys = [
        "fused_moe_kernel",
        "FlashAttnFwdSm90",
        "FlashAttnFwdCombine",
        "flash::FlashAttnFwdSm90",
        "nvjet_tst_",
        "nvjet_tss_",
        "triton_per_fused_",
        "triton_poi_fused_",
        "triton_red_fused_",
        "triton_tem_fused_",
        "ncclDevKernel_",
        "vllm::moe::",
    ]
    for key in known_keys:
        if key in s:
            return re.escape(key)

    # 2) Generic fallback: strip param-list and escape prefix
    no_params = s.split("(", 1)[0].strip()  # remove "(T1::Params)" etc.
    if not no_params:
        no_params = s.strip()
    prefix = no_params[:120]
    return re.escape(prefix)


# -------------------------------
# Nsight Compute: run & parse metrics
# -------------------------------

def run_ncu_for_kernel(
    kernel_name: str,
    run_cmd: str,
    out_dir: str,
    workload: str,
    index: int,
    launch_skip: int = 0,
    launch_count: int = 1,
    dry_run: bool = False,
) -> str:
    """
    Run NCU for a single kernel and return the metrics CSV path.

    We:
      - filter by kernel name using regex (robust against template noise)
      - request the MFU/MBU/time metrics directly via --metrics
      - write both a .ncu-rep (for later GUI inspection) and a CSV (for the script)
    """
    os.makedirs(out_dir, exist_ok=True)

    safe = sanitize_for_filename(kernel_name)
    rep_base = f"ncu_{workload}_k{index:03d}_{safe}"
    rep_path = os.path.join(out_dir, rep_base)
    csv_base = rep_path + "_metrics"

    # Metrics we care about for MFU / MBU / duration
    metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__time_duration.sum",
    ]

    pattern = to_regex_pattern(kernel_name)
    regex_arg = f"regex:{pattern}"

    sections = [
        "LaunchStats",
        "Occupancy",
        "SpeedOfLight",
        "SpeedOfLight_HierarchicalTensorRooflineChart",
    ]

    cmd = ["ncu", "-f"]  # -f to overwrite existing .ncu-rep

    for sec in sections:
        cmd.extend(["--section", sec])

    cmd.extend(
        [
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            regex_arg,
            "--launch-skip",
            str(launch_skip),
            "--launch-count",
            str(launch_count),
            "--metrics",
            ",".join(metrics),
            "--csv",
            "--log-file",
            csv_base,
            "-o",
            rep_path,
            "--",
        ]
    )

    cmd.extend(shlex.split(run_cmd))

    print(f"\n[NCU] workload={workload}, kernel #{index}: {kernel_name}")
    print(f"      Regex pattern: {regex_arg}")
    print("      .ncu-rep     :", rep_path + ".ncu-rep")
    print("      Metrics CSV  :", csv_base + ".csv")
    print("      Command      :", " ".join(shlex.quote(c) for c in cmd))

    if dry_run:
        return csv_base + ".csv"

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"[WARN] NCU failed for kernel: {kernel_name} (exit={result.returncode})",
            file=sys.stderr,
        )
    return csv_base + ".csv"


def parse_metrics_csv(csv_path: str):
    """
    Parse the metrics CSV produced by NCU with:
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,
               gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,
               gpu__time_duration.sum
      --csv --log-file <base>

    Returns:
      {
        "sm_pct_of_peak": float or None,
        "dram_pct_of_peak": float or None,
        "duration_ns": float or None,
      }
    """
    metrics = {
        "sm_pct_of_peak": None,
        "dram_pct_of_peak": None,
        "duration_ns": None,
    }

    if not os.path.exists(csv_path):
        print(f"[WARN] Metrics CSV not found: {csv_path}", file=sys.stderr)
        return metrics

    with open(csv_path, newline="") as f:
        # Skip comments and blank lines
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
        if not lines:
            return metrics

        headers = [h.strip() for h in lines[0].strip().split(",")]
        reader = csv.DictReader(lines[1:], fieldnames=headers)

        for row in reader:
            metric_name = (row.get("Metric Name") or row.get("ID") or "").strip()
            val_str = (row.get("Metric Value") or row.get("Value") or "").strip()
            if not metric_name or not val_str:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue

            mn = metric_name
            if "sm__throughput" in mn and "pct_of_peak" in mn:
                metrics["sm_pct_of_peak"] = val
            elif "dram__throughput" in mn and "pct_of_peak" in mn:
                metrics["dram_pct_of_peak"] = val
            elif "gpu__time_duration" in mn and "sum" in mn:
                metrics["duration_ns"] = val

    return metrics


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workload",
        required=True,
        help="Name of workload: full | prefill | decode (used in output filenames).",
    )
    ap.add_argument(
        "--nsys-kern-csv",
        required=True,
        help="Path to NSys kernel summary (CSV or Excel, e.g. Nsys.xlsx).",
    )
    ap.add_argument(
        "--run-cmd",
        required=True,
        help=(
            "Command to run under NCU, e.g. "
            "'python3 run_bench.py --target_length 31744 --output_length 700'"
        ),
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (e.g. outputs/full).",
    )
    ap.add_argument(
        "--min-time-ms",
        type=float,
        default=0.0,
        help="Filter kernels with total time < this (ms).",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Only profile top-K kernels by total time; 0 = all.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print NCU commands without executing them.",
    )
    args = ap.parse_args()

    top_k = None if args.top_k == 0 else args.top_k
    kernels = read_nsys_kernels(
        args.nsys_kern_csv,
        min_time_ms=args.min_time_ms,
        top_k=top_k,
    )

    if not kernels:
        print(
            "No kernels found from NSys summary with given filters.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(kernels)} hot kernels (workload={args.workload}):")
    for k in kernels[:10]:
        print(
            f"  {k.name}  time={k.total_time_ms:.3f} ms  pct={k.time_pct:.2f}%"
        )

    # Run NCU for each kernel
    rep_paths = []
    for idx, kinfo in enumerate(kernels, start=1):
        csv_path = run_ncu_for_kernel(
            kernel_name=kinfo.name,
            run_cmd=args.run_cmd,
            out_dir=args.out_dir,
            workload=args.workload,
            index=idx,
            launch_skip=0,   # tweak if you want to skip warmup launches
            launch_count=1,
            dry_run=args.dry_run,
        )
        rep_paths.append((kinfo, csv_path))

    if args.dry_run:
        print("[DRY-RUN] Skipping aggregation.")
        return

    summary_rows = []
    for kinfo, csv_path in rep_paths:
        m = parse_metrics_csv(csv_path)
        row = {
            "workload": args.workload,
            "kernel_name": kinfo.name,
            "nsys_time_ms": kinfo.total_time_ms,
            "nsys_time_pct": kinfo.time_pct,
            "ncu_sm_pct_of_peak": m["sm_pct_of_peak"],
            "ncu_dram_pct_of_peak": m["dram_pct_of_peak"],
            "ncu_duration_ns": m["duration_ns"],
        }
        summary_rows.append(row)

    out_csv = os.path.join(
        args.out_dir, f"ncu_kernel_summary_{args.workload}.csv"
    )
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"[OK] Wrote per-kernel summary to {out_csv}")
    else:
        print("[WARN] No summary rows collected.")


if __name__ == "__main__":
    main()
