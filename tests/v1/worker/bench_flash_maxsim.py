# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: flash-maxsim vs vanilla MaxSim scoring.

Run:  python tests/v1/worker/test_flash_maxsim_benchmark.py

Three levels of comparison:
  1. Kernel-level: flash_maxsim vs vanilla bmm (docs pre-stacked)
  2. API-level: compute_maxsim_score_batched (includes stacking overhead)
  3. Zero-copy: flash_maxsim_rerank_direct (0 bytes doc memory)
"""
import time

import torch

from vllm.v1.pool.flash_maxsim import (
    flash_maxsim,
    flash_maxsim_rerank_direct,
)
from vllm.v1.pool.late_interaction import (
    _HAS_FLASH_MAXSIM,
    _vanilla_compute_maxsim_score_batched,
    compute_maxsim_score_batched,
)


def _time_fn(fn, *args, warmup=3, repeats=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def _vanilla_bmm_maxsim(Q, D, max_batch=64, max_elems=64_000_000):
    """Vanilla MaxSim: pad + bmm + mask. Same algo as vLLM's original."""
    Lq = Q.shape[0]
    B, Ld, d = D.shape
    scores = []
    start = 0
    while start < B:
        end = min(start + max_batch, B)
        while end - start > 1 and (end - start) * Lq * Ld > max_elems:
            end -= 1
        batch_D = D[start:end]
        bsz = batch_D.shape[0]
        Q_exp = Q.float().unsqueeze(0).expand(bsz, -1, -1)
        token_scores = torch.bmm(Q_exp, batch_D.float().transpose(1, 2))
        scores.append(token_scores.amax(dim=-1).sum(dim=-1))
        start = end
    return torch.cat(scores)


def benchmark_kernel():
    """Direct kernel comparison with docs already stacked."""
    print("=" * 70)
    print("  KERNEL: flash_maxsim vs vanilla bmm (docs pre-stacked)")
    print("=" * 70)
    print(f"{'Config':>30} {'Vanilla (ms)':>14} {'Flash (ms)':>14} "
          f"{'Speedup':>10}")
    print("-" * 70)

    configs = [
        (32, 64, 128, 64, "ColBERT B=64"),
        (32, 128, 128, 256, "ColBERT B=256"),
        (32, 128, 128, 1000, "ColBERT B=1000"),
        (32, 128, 128, 5000, "ColBERT B=5000"),
        (128, 1024, 128, 64, "ColPali B=64"),
        (128, 1024, 128, 256, "ColPali B=256"),
        (128, 1024, 128, 1000, "ColPali B=1000"),
    ]

    for lq, ld, dim, B, label in configs:
        Q = torch.randn(lq, dim, device="cuda", dtype=torch.float16)
        D = torch.randn(B, ld, dim, device="cuda", dtype=torch.float16)

        t_vanilla = _time_fn(_vanilla_bmm_maxsim, Q, D) * 1000
        t_flash = _time_fn(flash_maxsim, Q, D) * 1000
        speedup = t_vanilla / t_flash if t_flash > 0 else float("inf")

        print(f"{label:>30} {t_vanilla:>13.2f} {t_flash:>13.2f} "
              f"{speedup:>9.1f}x")


def benchmark_api():
    """API-level: compute_maxsim_score_batched (1 query, N docs)."""
    print()
    print("=" * 70)
    print("  API: compute_maxsim_score_batched (1 query vs N docs)")
    print("=" * 70)
    print(f"{'Config':>30} {'Vanilla (ms)':>14} {'Flash (ms)':>14} "
          f"{'Speedup':>10}")
    print("-" * 70)

    for n, lq, ld, dim, label in [
        (64, 32, 128, 128, "ColBERT B=64"),
        (256, 32, 128, 128, "ColBERT B=256"),
        (1000, 32, 128, 128, "ColBERT B=1000"),
        (64, 128, 1024, 128, "ColPali B=64"),
        (256, 128, 1024, 128, "ColPali B=256"),
    ]:
        query = torch.randn(lq, dim, device="cuda", dtype=torch.float16)
        qs = [query] * n
        ds = [torch.randn(ld, dim, device="cuda", dtype=torch.float16)
              for _ in range(n)]

        t_vanilla = _time_fn(
            _vanilla_compute_maxsim_score_batched, qs, ds
        ) * 1000
        t_flash = _time_fn(compute_maxsim_score_batched, qs, ds) * 1000
        speedup = t_vanilla / t_flash if t_flash > 0 else float("inf")

        print(f"{label:>30} {t_vanilla:>13.2f} {t_flash:>13.2f} "
              f"{speedup:>9.1f}x")


def benchmark_memory():
    """Peak memory: vanilla materializes [B, Lq, Ld] score matrix."""
    print()
    print("=" * 70)
    print("  MEMORY: kernel-level (Lq=128, Ld=1024, d=128)")
    print("=" * 70)
    print(f"{'B':>10} {'Vanilla (MB)':>14} {'Flash (MB)':>14} "
          f"{'Saved (MB)':>12}")
    print("-" * 52)

    for B in [64, 256, 1000, 5000]:
        Q = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        D = torch.randn(B, 1024, 128, device="cuda", dtype=torch.float16)

        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _vanilla_bmm_maxsim(Q, D)
        mem_vanilla = (torch.cuda.max_memory_allocated() - base) / 1e6

        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        flash_maxsim(Q, D)
        mem_flash = (torch.cuda.max_memory_allocated() - base) / 1e6

        print(f"{B:>10} {mem_vanilla:>13.1f} {mem_flash:>13.1f} "
              f"{mem_vanilla - mem_flash:>11.1f}")


def benchmark_zerocopy():
    """Zero-copy: flash_maxsim_rerank_direct reads from batch tensor.

    Memory for doc scoring: 0 bytes. The kernel reads directly from
    the model's projected output tensor at scattered offsets.
    """
    print()
    print("=" * 70)
    print("  ZERO-COPY: flash_maxsim_rerank_direct")
    print("=" * 70)
    print(f"{'B':>10} {'Vanilla (MB)':>14} {'Zero-copy (MB)':>16} "
          f"{'Time (ms)':>12}")
    print("-" * 55)

    Lq, dim = 128, 128
    Ld = 1024

    for B in [64, 256, 1000, 5000]:
        Q = torch.randn(Lq, dim, device="cuda", dtype=torch.float16)
        # Simulate a batch tensor with all docs contiguous
        total_tokens = B * Ld
        batch_tensor = torch.randn(
            total_tokens, dim, device="cuda", dtype=torch.float16
        )
        offsets = torch.arange(
            0, total_tokens, Ld, device="cuda", dtype=torch.int32
        )
        lengths = torch.full(
            (B,), Ld, device="cuda", dtype=torch.int32
        )

        # Vanilla memory: would need to stack [B, Ld, d]
        vanilla_doc_mem = B * Ld * dim * 4 / 1e6  # fp32

        # Zero-copy: 0 extra bytes for docs
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = _time_fn(
            flash_maxsim_rerank_direct, Q, batch_tensor, offsets, lengths, Ld
        ) * 1000
        zerocopy_mem = (torch.cuda.max_memory_allocated() - base) / 1e6

        print(f"{B:>10} {vanilla_doc_mem:>13.1f} {zerocopy_mem:>15.1f} "
              f"{t:>11.2f}")


def benchmark_correctness():
    """Verify numerical correctness."""
    print()
    print("=" * 70)
    print("  CORRECTNESS")
    print("=" * 70)

    for B, lq, ld, dim, label in [
        (100, 32, 128, 128, "ColBERT"),
        (10, 128, 1024, 128, "ColPali"),
    ]:
        Q = torch.randn(lq, dim, device="cuda", dtype=torch.float16)
        D = torch.randn(B, ld, dim, device="cuda", dtype=torch.float16)

        ref = _vanilla_bmm_maxsim(Q, D)
        got = flash_maxsim(Q, D)
        err = (ref - got).abs().max().item()
        print(f"  {label}: max_err={err:.6f}")

    # Zero-copy correctness
    Q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
    batch = torch.randn(500, 128, device="cuda", dtype=torch.float16)
    offs = torch.tensor([0, 100, 250], device="cuda", dtype=torch.int32)
    lens = torch.tensor([80, 120, 100], device="cuda", dtype=torch.int32)
    scores = flash_maxsim_rerank_direct(Q, batch, offs, lens, 120)
    for i in range(3):
        doc = batch[offs[i]:offs[i]+lens[i]].unsqueeze(0)
        ref = _vanilla_bmm_maxsim(Q, doc)
        err = abs(scores[i].item() - ref.item())
        assert err < 1.0, f"zero-copy doc {i}: err={err}"
    print("  Zero-copy: verified 3 docs, all correct")


def main():
    assert _HAS_FLASH_MAXSIM, "flash-maxsim not available"
    assert torch.cuda.is_available(), "CUDA required"

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    benchmark_kernel()
    benchmark_api()
    benchmark_memory()
    benchmark_zerocopy()
    benchmark_correctness()

    print("\nDone.")


if __name__ == "__main__":
    main()
