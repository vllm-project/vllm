#!/usr/bin/env python
"""Side-by-side diff of vLLM Qwen3-Next BlockScale zero-init demo runs.

Reads the ``profiler_out_0.txt`` (PyTorch profiler cuda_time_total table)
that each TorchProfilerWrapper writes next to its trace JSON, plus the
``{mode}.json`` produced by ``vllm bench serve``, and prints:

  * top-level serving metrics (TPOT/ITL/throughput) per mode
  * kernel categories of interest (blockscale GEMMs, fill, memset, zero_)
    with #calls / self CUDA time / per-call time, and the splitk_fused vs
    splitk delta where applicable

Usage:
    python benchmarks/parse_zero_init_demo_results.py <results_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass

# (label, regex) - first match wins
KERNEL_BUCKETS = [
    ("aten::zero_                  ", re.compile(r"^aten::zero_$")),
    ("aten::fill_                  ", re.compile(r"^aten::fill_$")),
    ("BF16 fill (vec_elt<4>)       ", re.compile(r"vectorized_elementwise_kernel<4")),
    ("hipMemsetAsync / Memset      ", re.compile(r"hipMemsetAsync|Memset")),
    ("CK BlockScale GEMM           ", re.compile(r"ck::kernel_gemm_xdl_cshuffle")),
    ("CKTile BlockScale GEMM       ", re.compile(r"ck_tile6kentry|_ZN7ck_tile.*kentry")),
    ("ASM BlockScale GEMM          ", re.compile(r"asm_a8w8_blockscale|asm_gemm_a8w8_blockscale_bpreshuffle")),
    ("hipblaslt (Cijk_)            ", re.compile(r"^Cijk_")),
    ("fused_rms_fp8_group_quant    ", re.compile(r"_fused_rms_fp8_group_quant_kernel|fused_rms_fp8_group")),
    ("fused_silu_fp8_group_quant   ", re.compile(r"silu.*fp8_group|fp8_group.*silu")),
]


@dataclass
class KernelStat:
    name: str
    self_cuda_us: float
    n_calls: int


def parse_profiler_out(path: str) -> dict:
    rows: list[KernelStat] = []
    if not os.path.isfile(path):
        return {"rows": []}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            # Header / divider lines have many dashes
            if not line.strip() or line.startswith("---") or "Name" in line[:6]:
                continue
            # The torch profiler table is space-aligned; the last 10
            # columns are numeric counters. Parse by splitting on >=2 spaces.
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 10:
                continue
            name = parts[0]
            try:
                # column layout (after the name): Self CPU %, Self CPU,
                # CPU total %, CPU total, CPU time avg, Self CUDA, ...,
                # # of Calls (last column).
                self_cuda_str = parts[6]
                n_calls = int(parts[-1])
            except (IndexError, ValueError):
                continue
            self_cuda_us = parse_time(self_cuda_str)
            if self_cuda_us is None:
                continue
            rows.append(KernelStat(name=name, self_cuda_us=self_cuda_us, n_calls=n_calls))
    return {"rows": rows}


def parse_time(s: str) -> float | None:
    """Parse "12.345us" / "1.234ms" / "1.234s" / "12.34us " -> microseconds."""
    s = s.strip()
    m = re.match(r"([0-9.]+)\s*(us|ms|s)$", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "us":
        return val
    if unit == "ms":
        return val * 1000
    if unit == "s":
        return val * 1_000_000
    return None


def categorize(rows):
    out = {label: {"self_cuda_us": 0.0, "n_calls": 0} for (label, _) in KERNEL_BUCKETS}
    for r in rows:
        for (label, pat) in KERNEL_BUCKETS:
            if pat.search(r.name):
                out[label]["self_cuda_us"] += r.self_cuda_us
                out[label]["n_calls"] += r.n_calls
                break
    return out


def fmt_us(us: float) -> str:
    if us == 0:
        return "0us"
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s"
    if us >= 1000:
        return f"{us/1000:.2f}ms"
    return f"{us:.2f}us"


def find_profiler_out(mode_dir: str) -> str:
    candidates = [f for f in os.listdir(mode_dir) if f.startswith("profiler_out") and f.endswith(".txt")]
    if not candidates:
        return ""
    return os.path.join(mode_dir, sorted(candidates)[0])


def read_bench_json(p: str) -> dict:
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)


def print_header(title):
    print()
    print("=" * 78)
    print(f"# {title}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--modes", default="splitk,splitk_fused",
                        help="comma-separated mode names to compare")
    args = parser.parse_args()
    modes = args.modes.split(",")

    summaries: dict[str, dict] = {}
    bench_metrics: dict[str, dict] = {}
    for m in modes:
        mode_dir = os.path.join(args.results_dir, m)
        prof_out = find_profiler_out(mode_dir)
        rows = parse_profiler_out(prof_out)["rows"]
        summaries[m] = {"rows": rows, "buckets": categorize(rows)}
        bench_metrics[m] = read_bench_json(os.path.join(args.results_dir, f"{m}.json"))

    # --- serving metrics ---
    print_header("Serving metrics (vllm bench serve)")
    headers = ("metric", *modes)
    rows = []
    for k, label in [
        ("output_throughput", "output tok/s"),
        ("total_token_throughput", "total tok/s"),
        ("mean_tpot_ms", "mean TPOT ms"),
        ("median_tpot_ms", "median TPOT ms"),
        ("p99_tpot_ms", "p99 TPOT ms"),
        ("mean_itl_ms", "mean ITL ms"),
        ("median_itl_ms", "median ITL ms"),
        ("p99_itl_ms", "p99 ITL ms"),
        ("mean_ttft_ms", "mean TTFT ms"),
    ]:
        row = [label]
        for m in modes:
            v = bench_metrics.get(m, {}).get(k)
            row.append(f"{v:>10.3f}" if isinstance(v, (int, float)) else f"{'n/a':>10}")
        rows.append(row)
    col0 = max(len(r[0]) for r in rows)
    print(f"  {'metric':<{col0}}  " + "  ".join(f"{m:>12}" for m in modes))
    for r in rows:
        print(f"  {r[0]:<{col0}}  " + "  ".join(f"{v:>12}" for v in r[1:]))

    # --- kernel buckets ---
    print_header("Kernel-bucket comparison (self CUDA time, # calls)")
    base = modes[0]
    print(f"  {'bucket':<32}  " + "  ".join(f"{m+' time':>14} {m+' #':>10}" for m in modes))
    for label, _ in KERNEL_BUCKETS:
        parts = [f"  {label:<32}"]
        base_us = summaries[base]["buckets"][label]["self_cuda_us"]
        for m in modes:
            b = summaries[m]["buckets"][label]
            parts.append(f"{fmt_us(b['self_cuda_us']):>14} {b['n_calls']:>10d}")
        print("  ".join(parts))

    # --- explicit delta (splitk_fused vs splitk) for fill-shaped kernels ---
    if "splitk" in modes and "splitk_fused" in modes:
        print_header("splitk_fused vs splitk delta")
        for label in [
            "aten::zero_                  ",
            "aten::fill_                  ",
            "BF16 fill (vec_elt<4>)       ",
            "hipMemsetAsync / Memset      ",
            "CK BlockScale GEMM           ",
            "CKTile BlockScale GEMM       ",
            "ASM BlockScale GEMM          ",
        ]:
            sk = summaries["splitk"]["buckets"][label]
            fu = summaries["splitk_fused"]["buckets"][label]
            d_us = fu["self_cuda_us"] - sk["self_cuda_us"]
            d_calls = fu["n_calls"] - sk["n_calls"]
            pct = (d_us / sk["self_cuda_us"] * 100.0) if sk["self_cuda_us"] else 0.0
            print(f"  {label}: dt={fmt_us(d_us):>10}  dcalls={d_calls:+d}  pct={pct:+.1f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
