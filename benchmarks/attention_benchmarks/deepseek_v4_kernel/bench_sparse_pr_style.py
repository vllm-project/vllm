# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PR-style microbenchmark for the sparse-gather dequant kernel.

Mirrors the table layout of PR #42236 (the CuteDSL dense-gather port):

  Table A  — Sparse Kernel: Triton vs CuteDSL on identical inputs.
             Columns: k_len | triton_us | cutedsl_us | triton_GB/s |
                      cutedsl_GB/s | speedup
             Shapes are the same single-req / batched lengths PR #42236
             used (1, 8, 32, 128, 512, 2048, 8192, 16384, 32000, 262144,
             plus the batched [97, 1024, 8192, 16384] case).

  Table B  — Sparse CuteDSL vs Dense CuteDSL across sparsity fractions
             3% / 6% / 12% / 25% / 50% / 100%, on N_dense ∈
             {8192, 16384, 32768}. Shows the real feature gain (a
             sparse-25% gather beats the dense-100% baseline by ~1.2×).

Bytes-moved accounting follows PR #42236: per row we count
``HEAD_BYTES`` (584 B, the FP8+BF16+scale source) read + 1024 B (the
BF16 output) written.

Usage:
    python benchmarks/attention_benchmarks/deepseek_v4_kernel/\\
        bench_sparse_pr_style.py [--out results_sparse/bench_tables.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vllm.utils.import_utils import has_cutedsl
from vllm.v1.attention.ops.deepseek_v4_ops import quantize_and_insert_k_cache
from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    dequantize_and_gather_k_cache_sparse_triton,
)

if has_cutedsl():
    from vllm.v1.attention.ops.deepseek_v4_ops.dequant_gather_k_cutedsl import (
        dequantize_and_gather_k_cache_cutedsl,
        dequantize_and_gather_k_cache_sparse_cutedsl,
    )
else:
    raise SystemExit(
        "has_cutedsl()=False — sparse-cutedsl benchmark requires the CuteDSL "
        "stack (cutlass, cute, quack). Install or run on a host that has them."
    )


HEAD_DIM = 512
HEAD_BYTES = 584
BYTES_PER_ROW = HEAD_BYTES + HEAD_DIM * 2  # read + write
DEVICE = "cuda"


def _make_cache(num_tokens: int, block_size: int, seed: int = 0):
    """Populate a paged FP8 K-cache from ``num_tokens`` random BF16 tokens.

    Returns (k_cache, slot_ids_int32) where slot_ids_int32 is a flat list of
    all valid slot ids in the cache (one per inserted token, in insertion
    order — i.e. a contiguous range mapped through a 1-1 page assignment).
    """
    torch.manual_seed(seed)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_BYTES, dtype=torch.uint8, device=DEVICE
    )
    slot_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    quantize_and_insert_k_cache(
        kv, k_cache.view(num_blocks, -1), slot_ids, block_size=block_size
    )
    return k_cache, slot_ids.to(torch.int32)


def _time(fn, warmup: int = 5, iters: int = 50) -> tuple[float, float, float]:
    """Returns (median_us, p99_us, min_us).

    Reports min instead of std because CUDA-event timing on a shared host is
    bimodal — most samples cluster tightly around the steady-state cost, but
    occasional OS / driver hiccups inject 100×-median outliers that swamp
    any unweighted std-dev. ``min`` is the cleanest "no-interference" lower
    bound; ``p99`` captures the realistic worst case.
    """
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    starts = [torch.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends, strict=True):
        s.record()
        fn()
        e.record()
    torch.accelerator.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends, strict=True))
    n = len(us)
    median = us[n // 2]
    p99 = us[min(n - 1, int(n * 0.99))]
    return median, p99, us[0]


def _bw(rows: int, median_us: float) -> float:
    return rows * BYTES_PER_ROW / (median_us * 1e-6) / 1e9


# ──────────────────────────────────────────────────────────────────────────────
# Table A — Sparse Triton vs Sparse CuteDSL, single-impl-vs-single-impl,
# matching PR #42236 Table 1 shapes.
# ──────────────────────────────────────────────────────────────────────────────


def table_a(block_size: int, iters: int) -> list[str]:
    """Run Table A: Triton vs CuteDSL sparse on PR #42236 shapes."""
    # PR #42236 Table 1 k_len list, minus the 262144 case (would need a 154 MB
    # cache; we cap at 32K — same as the realistic prefill chunk upper bound).
    # The 32000 case mirrors PR #42236 verbatim.
    cases = [
        ("1", [1]),
        ("8", [8]),
        ("32", [32]),
        ("128", [128]),
        ("512", [512]),
        ("2048", [2048]),
        ("8192", [8192]),
        ("16384", [16384]),
        ("32000", [32000]),
        ("262144", [262144]),
        ("batched [97, 1024, 8192, 16384]", [97, 1024, 8192, 16384]),
    ]

    rows = []
    rows.append(
        "| k_len | triton_us | triton_p99_us | triton_min_us | cutedsl_us | "
        "cutedsl_p99_us | cutedsl_min_us | triton_GB/s | cutedsl_GB/s | speedup |"
    )
    rows.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for label, lens in cases:
        N = sum(lens)
        k_cache, all_slot_ids = _make_cache(N, block_size)
        # Use all slot ids as the gather list, in order. Same total work,
        # no padding-induced wasted bandwidth.
        slot_indices = all_slot_ids.clone()
        out = torch.zeros(N, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

        # Default-arg binding pins the loop variables to each closure (avoids
        # late-binding capture).
        def call_triton(out=out, k_cache=k_cache, slot_indices=slot_indices):
            dequantize_and_gather_k_cache_sparse_triton(
                out, k_cache, slot_indices, block_size
            )

        def call_cutedsl(out=out, k_cache=k_cache, slot_indices=slot_indices):
            dequantize_and_gather_k_cache_sparse_cutedsl(
                out, k_cache, slot_indices, block_size
            )

        t_med, t_p99, t_min = _time(call_triton, iters=iters)
        c_med, c_p99, c_min = _time(call_cutedsl, iters=iters)
        rows.append(
            f"| {label} | {t_med:.2f} | {t_p99:.2f} | {t_min:.2f} | "
            f"{c_med:.2f} | {c_p99:.2f} | {c_min:.2f} | "
            f"{_bw(N, t_med):.1f} | {_bw(N, c_med):.1f} | "
            f"{t_med / c_med:.2f}x |"
        )
        # Free per-case to keep memory in check for the 262144 case.
        del k_cache, all_slot_ids, slot_indices, out
        torch.accelerator.empty_cache()

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Table B — Sparse CuteDSL vs Dense CuteDSL across sparsity fractions.
# Shows the actual feature value: at typical C4A topk-union ratios (~10–25%),
# the sparse kernel saves wall time vs the dense baseline.
# ──────────────────────────────────────────────────────────────────────────────


def _flat_seq_lens_t(num_tokens: int) -> torch.Tensor:
    """Single-req seq_lens tensor used by the dense CuteDSL wrapper."""
    return torch.tensor([num_tokens], dtype=torch.int32, device=DEVICE)


def _flat_block_table(num_tokens: int, block_size: int) -> torch.Tensor:
    """Identity block_table for the dense CuteDSL wrapper.

    ``_make_cache`` lays tokens out one-per-physical-slot in order, so the
    block_table is just ``arange(num_blocks)`` for the single request.
    """
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    return torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)


def table_b(block_size: int, iters: int) -> list[str]:
    """Run Table B: sparse-cutedsl vs dense-cutedsl across sparsity sweeps.

    Dense is re-measured fresh inside each (N_dense, fraction) cell so the
    `dense_us` and `sparse_us` columns share identical thermal / cache /
    memory-fragmentation state. We also bracket the timings with
    ``empty_cache()`` to avoid carry-over from Table A.
    """
    n_dense_values = [8192, 16384, 32768]
    fractions = [0.03, 0.06, 0.12, 0.25, 0.50, 1.00]

    rows = []
    rows.append(
        "| N_dense | fraction | rows | dense_us | dense_p99_us | dense_min_us | "
        "sparse_us | sparse_p99_us | sparse_min_us | dense_GB/s | sparse_GB/s | "
        "speedup |"
    )
    rows.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |"
    )

    for N_dense in n_dense_values:
        torch.accelerator.empty_cache()
        k_cache, all_slot_ids = _make_cache(N_dense, block_size)
        seq_lens_t = _flat_seq_lens_t(N_dense)
        block_table = _flat_block_table(N_dense, block_size)
        out_dense = torch.zeros(
            1, N_dense, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )

        # Default-arg binding pins the per-iteration tensors to the closure.
        def call_dense(
            out_dense=out_dense,
            k_cache=k_cache,
            seq_lens_t=seq_lens_t,
            block_table=block_table,
        ):
            dequantize_and_gather_k_cache_cutedsl(
                out_dense, k_cache, seq_lens_t, None, block_table, block_size, 0
            )

        for frac in fractions:
            M = max(1, int(round(frac * N_dense)))
            g = torch.Generator(device=DEVICE).manual_seed(int(frac * 1000))
            perm = torch.randperm(N_dense, generator=g, device=DEVICE)
            slot_indices = all_slot_ids[perm[:M]].contiguous()
            out_sparse = torch.zeros(M, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

            def call_sparse(
                out_sparse=out_sparse,
                k_cache=k_cache,
                slot_indices=slot_indices,
            ):
                dequantize_and_gather_k_cache_sparse_cutedsl(
                    out_sparse, k_cache, slot_indices, block_size
                )

            dense_med, dense_p99, dense_min = _time(call_dense, iters=iters)
            sparse_med, sparse_p99, sparse_min = _time(call_sparse, iters=iters)
            dense_bw = _bw(N_dense, dense_med)
            sparse_bw = _bw(M, sparse_med)
            rows.append(
                f"| {N_dense} | {int(frac * 100):d}% | {M} | "
                f"{dense_med:.2f} | {dense_p99:.2f} | {dense_min:.2f} | "
                f"{sparse_med:.2f} | {sparse_p99:.2f} | {sparse_min:.2f} | "
                f"{dense_bw:.1f} | {sparse_bw:.1f} | "
                f"{dense_med / sparse_med:.2f}x |"
            )

        del k_cache, all_slot_ids, out_dense
        torch.accelerator.empty_cache()

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write the markdown report (also printed to stdout).",
    )
    args = parser.parse_args()

    # No torch.accelerator equivalent for device name / properties as of
    # torch 2.11 — fall back to the CUDA APIs (this script is CUDA-only).
    gpu = torch.cuda.get_device_name(0)  # noqa: TID251
    p = torch.cuda.get_device_properties(0)  # noqa: TID251
    header = [
        "# DSv4 sparse dequant kernel — PR-style benchmark",
        "",
        f"- GPU: **{gpu}** (sm_{p.major}{p.minor})",
        f"- block_size: {args.block_size}, iters: {args.iters}",
        "- Bytes/row = 584 (FP8+BF16+scale read) + 1024 (BF16 written) = 1608",
        "- `*_us` columns are median µs; `*_p99_us` is the 99th percentile; "
        "`*_min_us` is the fastest sample (no-interference lower bound). All "
        "over `iters` cuda-event samples after 5 warmups. Median is the "
        "primary signal; p99 and min bound the noise envelope.",
        "",
    ]

    section_a = [
        "## Table A — Sparse Triton vs Sparse CuteDSL",
        "",
        "Identical inputs (all slot ids of a freshly populated cache). Mirrors "
        "PR #42236 Table 1's `k_len` shapes.",
        "",
    ]
    section_a += table_a(args.block_size, args.iters)
    section_a += [""]

    section_b = [
        "## Table B — Sparse CuteDSL vs Dense CuteDSL across sparsity",
        "",
        "Same K-cache; sparse path samples a random subset of slot ids at "
        "fractions `f ∈ {3, 6, 12, 25, 50, 100}%`. The dense column is "
        "`dequantize_and_gather_k_cache_cutedsl(seq_lens=[N_dense])`. "
        "`speedup = dense_us / sparse_us`; values >1 mean the sparse kernel "
        "saves wall time over the dense baseline at that sparsity.",
        "",
    ]
    section_b += table_b(args.block_size, args.iters)
    section_b += [""]

    section_c = [
        "## Table C — End-to-end TTFT",
        "",
        "_Pending — kernel-only PR. End-to-end TTFT numbers depend on the "
        "C4A prefill pipeline integration (dedup of `topk_indices` per "
        "request, sparse dequant call, mapping into the compact `kv` "
        "buffer), which is scheduled as a follow-up PR._",
        "",
    ]

    report = "\n".join(header + section_a + section_b + section_c)
    print(report)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
