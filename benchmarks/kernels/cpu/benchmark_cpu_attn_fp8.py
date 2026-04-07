#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Microbenchmark comparing 16-bit and FP8 CPU attention kernel performance.

Runs both paths on identical workloads and reports latency and effective
KV-cache memory bandwidth for each, plus the FP8 speedup ratio.

Usage examples:
  # compare BF16-AMX vs FP8-E4M3-AMX (default on AMX machines):
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py

  # compare BF16-AMX vs FP8-E5M2-AMX:
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py \\
      --fp8-format fp8_e5m2

  # 3-way compare: BF16-AMX, FP8-E4M3-AMX, FP8-E5M2-AMX:
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py \\
      --fp8-format both --isa-fp8 amx

  # E4M3 vs E5M2 on VEC path (no AMX):
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py \\
      --kv-cache-dtype fp8 --fp8-format both --isa-fp8 vec

  # long-decode scenario, MHA heads:
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py \\
      --scenario long-decode --num-query-heads 32 --num-kv-heads 8 --head-size 128 \\
      --fp8-format both

  # custom batch:
  numactl -m 1 -N 1 python benchmarks/kernels/cpu/benchmark_cpu_attn_fp8.py \\
      --scenario custom --batch-size 32 --q-len 1 --kv-len 2048 --fp8-format fp8_e5m2
"""

import ctypes
import ctypes.util
import functools
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# AMX tile-state initialisation (Linux only)
# ---------------------------------------------------------------------------
# Executing AMX tile instructions requires opt-in via arch_prctl.  This must
# happen before the first _tile_loadconfig / _tile_dpbf16ps call.
if sys.platform == "linux" and torch.cpu._is_amx_tile_supported():
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _libc.syscall(
        158, 0x1023, 18
    )  # arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)

from vllm._custom_ops import (
    cpu_attention_with_kv_cache,
    cpu_attention_with_kv_cache_fp8,
    cpu_attn_get_scheduler_metadata,
    cpu_attn_reshape_and_cache,
    cpu_attn_reshape_and_cache_fp8,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _rand_tensor(n: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(n, dtype=dtype)


def _median(times: list[float]) -> float:
    return float(np.median(times))


def _kv_bytes(
    seq_lens: list[tuple[int, int]],
    num_kv_heads: int,
    head_size: int,
    bytes_per_elem: int,
) -> float:
    """Total KV bytes read by the attention kernel (K + V for each context token)."""
    total_kv_tokens = sum(kv for _, kv in seq_lens)
    return 2 * total_kv_tokens * num_kv_heads * head_size * bytes_per_elem


def _bandwidth_gbs(bytes_read: float, elapsed_ms: float) -> float:
    return bytes_read / (elapsed_ms * 1e-3) / 1e9


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _build_fp16_caches(
    seq_lens: list[tuple[int, int]],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    isa: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and pack KV caches in the native (bf16/fp16) format."""
    kv_flat = _rand_tensor(
        2 * num_blocks * block_size * num_kv_heads * head_size, dtype
    ).view(2, num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype
    )
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache(
        kv_flat[0].reshape(-1, num_kv_heads, head_size),
        kv_flat[1].reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        isa,
    )
    return key_cache, value_cache


def _build_fp8_caches(
    seq_lens: list[tuple[int, int]],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    k_scale: float,
    v_scale: float,
    isa: str = "vec",
    kv_cache_dtype: str = "fp8_e4m3",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and pack KV caches in FP8 (uint8) format."""
    kv_flat = _rand_tensor(
        2 * num_blocks * block_size * num_kv_heads * head_size, dtype
    ).view(2, num_blocks, block_size, num_kv_heads, head_size)
    key_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_size, dtype=torch.uint8
    )
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.int64)
    cpu_attn_reshape_and_cache_fp8(
        kv_flat[0].reshape(-1, num_kv_heads, head_size),
        kv_flat[1].reshape(-1, num_kv_heads, head_size),
        key_cache,
        value_cache,
        slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )
    return key_cache, value_cache


def _build_common(
    seq_lens: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    isa: str,
) -> tuple:
    """Build query, block_table, cumulative lens, scheduler metadata."""
    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [kv for _, kv in seq_lens]
    token_num = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    query = _rand_tensor(token_num * num_query_heads * head_size, dtype).view(
        token_num, num_query_heads, head_size
    )
    cu_query = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    block_table = torch.arange(max_blocks_per_seq * num_seqs, dtype=torch.int32).view(
        num_seqs, max_blocks_per_seq
    )

    sched = cpu_attn_get_scheduler_metadata(
        num_reqs=num_seqs,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_size,
        seq_lens=kv_lens_t,
        dtype=dtype,
        query_start_loc=cu_query,
        causal=True,
        sliding_window_size=-1,
        isa=isa,
        enable_kv_split=True,
    )
    return query, cu_query, kv_lens_t, block_table, sched


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


@torch.inference_mode()
def bench_fp16(
    seq_lens: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    isa: str,
    iters: int,
    warmup: int,
) -> dict:
    query, cu_query, kv_lens_t, block_table, sched = _build_common(
        seq_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        isa,
    )
    key_cache, value_cache = _build_fp16_caches(
        seq_lens,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        isa,
    )
    scale = head_size**-0.5
    output = torch.empty_like(query)

    def _run():
        cpu_attention_with_kv_cache(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=cu_query,
            seq_lens=kv_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=sched,
            s_aux=None,
        )

    for _ in range(warmup):
        _run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _run()
        times.append((time.perf_counter_ns() - t0) / 1e6)

    bpp = dtype.itemsize  # bytes per element
    kv_bytes = _kv_bytes(seq_lens, num_kv_heads, head_size, bpp)
    med = _median(times)
    return {
        "times_ms": times,
        "median_ms": med,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "bandwidth_gbs": _bandwidth_gbs(kv_bytes, med),
        "kv_bytes": kv_bytes,
    }


@torch.inference_mode()
def bench_fp8(
    seq_lens: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    isa: str,
    iters: int,
    warmup: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_cache_dtype: str = "fp8_e4m3",
) -> dict:
    query, cu_query, kv_lens_t, block_table, sched = _build_common(
        seq_lens,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        isa,
    )
    key_cache, value_cache = _build_fp8_caches(
        seq_lens,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        dtype,
        k_scale,
        v_scale,
        isa=isa,
        kv_cache_dtype=kv_cache_dtype,
    )
    scale = head_size**-0.5
    output = torch.empty(query.shape, dtype=dtype)

    def _run():
        cpu_attention_with_kv_cache_fp8(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_start_loc=cu_query,
            seq_lens=kv_lens_t,
            scale=scale,
            causal=True,
            alibi_slopes=None,
            sliding_window=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=sched,
            s_aux=None,
            k_scale=k_scale,
            v_scale=v_scale,
            kv_cache_dtype=kv_cache_dtype,
        )

    for _ in range(warmup):
        _run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        _run()
        times.append((time.perf_counter_ns() - t0) / 1e6)

    kv_bytes = _kv_bytes(seq_lens, num_kv_heads, head_size, 1)  # 1 byte/elem for fp8
    med = _median(times)
    return {
        "times_ms": times,
        "median_ms": med,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "bandwidth_gbs": _bandwidth_gbs(kv_bytes, med),
        "kv_bytes": kv_bytes,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_seq_lens(seq_lens: list[tuple[int, int]]) -> str:
    if all(q == 1 for q, _ in seq_lens):
        kv_avg = int(np.mean([kv for _, kv in seq_lens]))
        return f"decode x{len(seq_lens)} (avg kv={kv_avg})"
    if all(q == kv for q, kv in seq_lens):
        q_avg = int(np.mean([q for q, _ in seq_lens]))
        return f"prefill x{len(seq_lens)} (avg len={q_avg})"
    return f"mixed x{len(seq_lens)}"


def print_results(
    label: str,
    seq_lens: list[tuple[int, int]],
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    results: dict[str, dict | None],
    dtype: torch.dtype,
) -> None:
    """Print a comparison table.

    *results* is a dict mapping a display label (e.g. "bf16-amx", "fp8-vec",
    "fp8-amx") to a result dict (or None if that path was not run).
    """
    dtype_str = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype))
    print(f"\n{'=' * 74}")
    print(f"  Scenario : {label}  [{_fmt_seq_lens(seq_lens)}]")
    print(
        f"  Config   : {dtype_str} query, heads={num_query_heads}/{num_kv_heads}, "
        f"head_size={head_size}"
    )
    print(f"{'=' * 74}")
    print(f"  {'path':22s} {'ms median ':>10} {'mean ':>9} {'std ':>8} {'BW GB/s':>9}")
    print(f"  {'-' * 22}  {'-' * 10}  {'-' * 9}  {'-' * 8}  {'-' * 9}")

    # baseline = first non-None result
    baseline_ms: float | None = None
    for row_label, res in results.items():
        if res is None:
            continue
        if baseline_ms is None:
            baseline_ms = res["median_ms"]
        speedup_str = ""
        if baseline_ms is not None and res["median_ms"] > 0:
            sp = baseline_ms / res["median_ms"]
            speedup_str = f"  ({sp:.2f}x vs baseline)"
        print(
            f"  {row_label:22s}  "
            f"{res['median_ms']:>10.3f}  "
            f"{res['mean_ms']:>9.3f}  "
            f"{res['std_ms']:>8.3f}  "
            f"{res['bandwidth_gbs']:>9.2f}"
            f"{speedup_str}"
        )

    # Summary line
    filled = {k: v for k, v in results.items() if v is not None}
    if len(filled) >= 2:
        labels = list(filled.keys())
        vals = list(filled.values())
        sp = vals[0]["median_ms"] / vals[-1]["median_ms"]
        print(
            f"\n  {labels[0]} → {labels[-1]} speedup: {sp:.2f}x  "
            f"(ideal FP8 BW gain = 2x; FP8 = 1 B/elem vs 2 B/elem)"
        )


# ---------------------------------------------------------------------------
# Predefined scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "decode": [(1, 213), (1, 1), (1, 312), (1, 7), (1, 7812)],
    "prefill": [(512, 512), (256, 256), (1024, 1024), (128, 128)],
    "mixed": [(992, 2456), (1, 1234), (98, 1145), (1, 4162), (2345, 2345)],
    "long-decode": [(1, kv) for kv in [1024, 2048, 4096, 8192, 8192, 8192, 8192]],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark 16-bit vs FP8 CPU attention kernel."
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["custom"],
        default="decode",
        help="Predefined sequence-length scenario (default: decode). "
        "Use 'custom' with --batch-size/--q-len/--kv-len.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of sequences (used with --scenario custom).",
    )
    parser.add_argument(
        "--q-len",
        type=int,
        default=1,
        help="Query length per sequence (custom scenario).",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=512,
        help="KV length per sequence (custom scenario).",
    )
    parser.add_argument("--num-query-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument(
        "--head-size", type=int, default=128, choices=[64, 80, 96, 112, 128, 192, 256]
    )
    parser.add_argument("--block-size", type=int, default=32, choices=[32, 64, 128])
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "half", "float"]
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="both",
        choices=["auto", "fp8", "both"],
        help="Which path(s) to benchmark: auto=16-bit only, fp8=fp8 only, "
        "both=compare both 16-bit and FP8 (default).",
    )
    parser.add_argument(
        "--fp8-format",
        type=str,
        default="fp8_e4m3",
        choices=["fp8_e4m3", "fp8_e5m2", "both"],
        help="FP8 format(s) to benchmark: fp8_e4m3 (default), fp8_e5m2, or "
        "both (runs E4M3 and E5M2 as separate rows for direct comparison).",
    )
    parser.add_argument(
        "--isa-fp8",
        type=str,
        default=None,
        choices=["vec", "amx"],
        help="ISA for the FP8 path. Default: 'amx' if hardware supports it "
        "and block-size is compatible, else 'vec'.",
    )
    parser.add_argument("--k-scale", type=float, default=1.0)
    parser.add_argument("--v-scale", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)

    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if args.scenario == "custom":
        seq_lens = [(args.q_len, args.kv_len)] * args.batch_size
        label = "custom"
    else:
        seq_lens = SCENARIOS[args.scenario]
        label = args.scenario

    # Resolve ISAs
    isa_fp16 = _get_attn_isa(dtype, args.block_size)
    # FP8 AMX: requires AMX hardware + BF16 dtype + block_size % 32 == 0
    amx_ok = (
        torch.cpu._is_amx_tile_supported()
        and dtype == torch.bfloat16
        and args.block_size % 32 == 0
    )
    if args.isa_fp8 is not None:
        isa_fp8 = args.isa_fp8
    elif amx_ok:
        isa_fp8 = "amx"
    else:
        isa_fp8 = "vec"

    # Adjust num_blocks to be large enough for the workload.
    max_kv = max(kv for _, kv in seq_lens)
    min_blocks_needed = (max_kv + args.block_size - 1) // args.block_size * len(
        seq_lens
    ) + 1
    num_blocks = max(args.num_blocks, min_blocks_needed)

    print(
        f"ISA (16-bit): {isa_fp16}  |  ISA (fp8): {isa_fp8}  |  "
        f"iters={args.iters}  warmup={args.warmup}"
    )

    run_fp16 = args.kv_cache_dtype in ("auto", "both")
    run_fp8 = args.kv_cache_dtype in ("fp8", "both")

    # Which FP8 formats to run
    fp8_formats: list[str]
    if args.fp8_format == "both":
        fp8_formats = ["fp8_e4m3", "fp8_e5m2"]
    else:
        fp8_formats = [args.fp8_format]

    # Collect results in insertion order for the table
    results: dict[str, dict | None] = {}

    if run_fp16:
        row = f"{args.dtype}-{isa_fp16}"
        print(f"\nRunning {row} ...", flush=True)
        results[row] = bench_fp16(
            seq_lens=seq_lens,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            num_blocks=num_blocks,
            dtype=dtype,
            isa=isa_fp16,
            iters=args.iters,
            warmup=args.warmup,
        )

    if run_fp8:
        for fmt in fp8_formats:
            row = f"{fmt}-{isa_fp8}"
            print(f"Running {row} ...", flush=True)
            results[row] = bench_fp8(
                seq_lens=seq_lens,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_size=args.head_size,
                block_size=args.block_size,
                num_blocks=num_blocks,
                dtype=dtype,
                isa=isa_fp8,
                iters=args.iters,
                warmup=args.warmup,
                k_scale=args.k_scale,
                v_scale=args.v_scale,
                kv_cache_dtype=fmt,
            )

    print_results(
        label=label,
        seq_lens=seq_lens,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        results=results,
        dtype=dtype,
    )
