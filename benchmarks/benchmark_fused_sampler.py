#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402, UP042, I001
"""Comprehensive Benchmark & Correctness Verification: Fused Top-K/Top-P Sampler.

Directly matches the workload scenarios and Zipf power-law distribution of
vLLM's official cuRITA benchmark (benchmarks/benchmark_topk_topp.py).

Compares:
1. Stock Triton (Baseline): apply_top_k_top_p_triton + gumbel_sample
   (2-pass + HBM roundtrip)
2. Fused Triton (Candidate): fused_topk_topp_sample
   (1-pass in-register, 0 HBM roundtrips)
3. FlashInfer (Reference): flashinfer_sample
   (unseeded C++ reference when k or p is active)
"""

import argparse
from dataclasses import dataclass

import torch

# Compatibility shim for Python 3.10 environments (after torch is loaded)
import enum
import typing

import typing_extensions

if not hasattr(enum, "StrEnum"):

    class StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = StrEnum

for _attr in (
    "Self",
    "assert_never",
    "Never",
    "reveal_type",
    "NotRequired",
    "Required",
    "TypeAlias",
):
    if hasattr(typing_extensions, _attr):
        setattr(typing, _attr, getattr(typing_extensions, _attr))

from vllm.v1.sample.ops.fused_topk_topp_sampler import (
    fused_topk_topp_sample,
    reset_fused_buffer_cache,
)
from vllm.v1.sample.ops.topk_topp_sampler import flashinfer_sample
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    reset_buffer_cache,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@dataclass
class ScenarioConfig:
    name: str
    batch_size: int
    vocab_size: int
    k: torch.Tensor | None
    p: torch.Tensor | None
    temp: torch.Tensor
    seeds: torch.Tensor
    pos: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    description: str


def create_zipf_logits(
    batch_size: int, vocab_size: int, device: str = "cuda"
) -> torch.Tensor:
    """Create Zipf power-law distributed logits mimicking realistic LLM outputs."""
    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
    probs = ranks.pow(-1.1)
    probs = probs / probs.sum()
    base_logits = probs.log()

    logits = base_logits.unsqueeze(0).expand(batch_size, -1).clone()
    for i in range(batch_size):
        logits[i] = logits[i, torch.randperm(vocab_size, device=device)]
    return logits


def flush_l2_cache(device: str = "cuda"):
    """Flush GPU L2 cache to prevent artificial cache hits across iterations."""
    # A100 has 40MB L2, H100 has 50MB L2. Allocating 64MB completely clears L2.
    flush_tensor = torch.empty(
        int(64 * 1024 * 1024 / 4), dtype=torch.int32, device=device
    )
    flush_tensor.zero_()


def build_curita_scenarios(
    batch_size: int,
    vocab_size: int,
    device: str = "cuda",
) -> list[ScenarioConfig]:
    """Build all 6 upstream cuRITA benchmark scenarios plus greedy and unfiltered."""
    scenarios = []

    # Shared request invariants
    torch.manual_seed(42)
    seeds = torch.randint(
        0, 10_000_000, (batch_size,), dtype=torch.int64, device=device
    )
    pos = torch.arange(batch_size, dtype=torch.int64, device=device)
    idx_map = torch.arange(batch_size, dtype=torch.int64, device=device)
    t_default = torch.full((batch_size,), 0.8, dtype=torch.float32, device=device)

    # 1. topk_whole: 100% active top-k (k=50)
    k_all = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
    scenarios.append(
        ScenarioConfig(
            name="topk_whole",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=k_all,
            p=None,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="100% Top-K active (k=50)",
        )
    )

    # 2. topk_partial: 50% active top-k (k=50), 50% no-op (k=vocab_size)
    k_part = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
    k_part[batch_size // 2 :] = vocab_size
    scenarios.append(
        ScenarioConfig(
            name="topk_partial",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=k_part,
            p=None,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="50% Top-K (k=50), 50% No-op",
        )
    )

    # 3. topp_whole: 100% active top-p (p=0.9) - tests standalone top-p path
    p_all = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
    scenarios.append(
        ScenarioConfig(
            name="topp_whole",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=None,
            p=p_all,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="100% Top-P active (p=0.9, standalone)",
        )
    )

    # 4. topp_partial: 50% active top-p (p=0.9), 50% no-op (p=1.0)
    p_part = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
    p_part[batch_size // 2 :] = 1.0
    scenarios.append(
        ScenarioConfig(
            name="topp_partial",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=None,
            p=p_part,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="50% Top-P (p=0.9), 50% No-op",
        )
    )

    # 5. topk_topp_whole: 100% combined top-k + top-p (k=100, p=0.9)
    k_comb = torch.full((batch_size,), 100, dtype=torch.int32, device=device)
    p_comb = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
    scenarios.append(
        ScenarioConfig(
            name="topk_topp_whole",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=k_comb,
            p=p_comb,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="100% Combined (k=100, p=0.9)",
        )
    )

    # 6. mixed_partial: 1/3 k-only, 1/3 p-only, 1/3 both (real production traffic)
    k_mix = torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
    p_mix = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)
    third = max(1, batch_size // 3)
    k_mix[:third] = 50
    p_mix[third : 2 * third] = 0.5
    k_mix[2 * third :] = 100
    p_mix[2 * third :] = 0.9
    scenarios.append(
        ScenarioConfig(
            name="mixed_partial",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=k_mix,
            p=p_mix,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="Mixed Multi-Tenant (1/3 k, 1/3 p, 1/3 both)",
        )
    )

    # 7. greedy: deterministic argmax (temp=0.0)
    t_greedy = torch.full((batch_size,), 0.0, dtype=torch.float32, device=device)
    scenarios.append(
        ScenarioConfig(
            name="greedy",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=k_all,
            p=p_all,
            temp=t_greedy,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="Deterministic Greedy (temp=0.0, k=50, p=0.9)",
        )
    )

    # 8. unfiltered: pure temperature sampling (no k, no p)
    scenarios.append(
        ScenarioConfig(
            name="unfiltered",
            batch_size=batch_size,
            vocab_size=vocab_size,
            k=None,
            p=None,
            temp=t_default,
            seeds=seeds,
            pos=pos,
            expanded_idx_mapping=idx_map,
            description="Unfiltered Sampling (k=None, p=None, temp=0.8)",
        )
    )

    return scenarios


def verify_bit_for_bit(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    device: str = "cuda",
) -> bool:
    """Verify exact bit-for-bit token equivalence between stock and fused samplers."""
    print("=" * 86)
    print("PHASE 1: BIT-FOR-BIT CORRECTNESS VERIFICATION AGAINST STOCK vLLM")
    print("=" * 86)
    all_passed = True
    total_checks = 0

    for v in vocab_sizes:
        for b in batch_sizes:
            scenarios = build_curita_scenarios(b, v, device=device)
            raw_logits = create_zipf_logits(b, v, device=device)

            for sc in scenarios:
                total_checks += 1
                temp_div = torch.where(sc.temp == 0.0, 1.0, sc.temp).unsqueeze(1)
                proc_logits = raw_logits / temp_div

                # Baseline: Stock vLLM Fallback (2-pass: filter + gumbel)
                stock_masked = apply_top_k_top_p_triton(proc_logits.clone(), sc.k, sc.p)
                stock_tokens = gumbel_sample(
                    stock_masked,
                    sc.expanded_idx_mapping,
                    sc.temp,
                    sc.seeds,
                    sc.pos,
                    apply_temperature=False,
                    is_drafting=False,
                    use_fp64=False,
                )

                # Candidate: Fused Sampler (in-register 0 HBM writeback)
                if sc.k is not None or sc.p is not None:
                    fused_tokens = fused_topk_topp_sample(
                        proc_logits.clone(),
                        sc.expanded_idx_mapping,
                        sc.temp,
                        sc.seeds,
                        sc.pos,
                        k=sc.k,
                        p=sc.p,
                    )
                else:
                    fused_tokens = gumbel_sample(
                        proc_logits.clone(),
                        sc.expanded_idx_mapping,
                        sc.temp,
                        sc.seeds,
                        sc.pos,
                        apply_temperature=False,
                        is_drafting=False,
                        use_fp64=False,
                    )

                match = torch.equal(stock_tokens, fused_tokens)
                tag = f"B={b:<3} V={v // 1000:>3}k | {sc.name:<16}"
                if match:
                    print(f"  [PASS] {tag:<30} -> 100% Exact match ({b} tokens)")
                else:
                    all_passed = False
                    diff = (stock_tokens != fused_tokens).sum().item()
                    print(f"  [FAIL] {tag:<30} -> MISMATCH on {diff}/{b} tokens!")
                    print(f"         Stock tokens: {stock_tokens[:4].tolist()}")
                    print(f"         Fused tokens: {fused_tokens[:4].tolist()}")

    print("-" * 86)
    if all_passed:
        print(f"RESULT: ALL {total_checks} SCENARIOS PASSED WITH 100% PARITY!")
    else:
        print("RESULT: VERIFICATION FAILED ON AT LEAST ONE SCENARIO.")
    print("=" * 86 + "\n")
    return all_passed


def run_latency_benchmarks(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    warmup_iters: int = 15,
    bench_iters: int = 40,
    device: str = "cuda",
):
    """Profile latency across Stock Triton, Fused Triton, and FlashInfer."""
    print("=" * 105)
    print("PHASE 2: HARDWARE LATENCY & HBM TRAFFIC PROFILING (NVIDIA A100)")
    print(f"Warmup: {warmup_iters} | Measured: {bench_iters} | Cold L2 Cache: Enabled")
    print("=" * 105)

    header = (
        f"{'Scenario':<18} | {'Batch':<5} | {'Vocab':<6} | "
        f"{'Stock (ms)':<10} | {'Fused (ms)':<10} | {'Speedup':<8} | "
        f"{'FlashInfer':<10} | {'HBM Saved'}"
    )
    print(header)
    print("-" * len(header))

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for v in vocab_sizes:
        for b in batch_sizes:
            scenarios = build_curita_scenarios(b, v, device=device)
            raw_logits = create_zipf_logits(b, v, device=device)

            for sc in scenarios:
                temp_div = torch.where(sc.temp == 0.0, 1.0, sc.temp).unsqueeze(1)
                proc_logits = raw_logits / temp_div

                # --- 1. Stock Triton ---
                reset_buffer_cache()
                for _ in range(warmup_iters):
                    m = apply_top_k_top_p_triton(proc_logits.clone(), sc.k, sc.p)
                    _ = gumbel_sample(
                        m,
                        sc.expanded_idx_mapping,
                        sc.temp,
                        sc.seeds,
                        sc.pos,
                        apply_temperature=False,
                        is_drafting=False,
                        use_fp64=False,
                    )
                torch.cuda.synchronize()

                stock_times = []
                for _ in range(bench_iters):
                    flush_l2_cache(device=device)
                    l_in = proc_logits.clone()
                    start_ev.record()
                    m = apply_top_k_top_p_triton(l_in, sc.k, sc.p)
                    _ = gumbel_sample(
                        m,
                        sc.expanded_idx_mapping,
                        sc.temp,
                        sc.seeds,
                        sc.pos,
                        apply_temperature=False,
                        is_drafting=False,
                        use_fp64=False,
                    )
                    end_ev.record()
                    end_ev.synchronize()
                    stock_times.append(start_ev.elapsed_time(end_ev))

                stock_med = sorted(stock_times)[len(stock_times) // 2]

                # --- 2. Fused Triton ---
                has_filter = sc.k is not None or sc.p is not None
                reset_fused_buffer_cache()
                for _ in range(warmup_iters):
                    if has_filter:
                        _ = fused_topk_topp_sample(
                            proc_logits.clone(),
                            sc.expanded_idx_mapping,
                            sc.temp,
                            sc.seeds,
                            sc.pos,
                            k=sc.k,
                            p=sc.p,
                        )
                    else:
                        _ = gumbel_sample(
                            proc_logits.clone(),
                            sc.expanded_idx_mapping,
                            sc.temp,
                            sc.seeds,
                            sc.pos,
                            apply_temperature=False,
                            is_drafting=False,
                            use_fp64=False,
                        )
                torch.cuda.synchronize()

                fused_times = []
                for _ in range(bench_iters):
                    flush_l2_cache(device=device)
                    l_in = proc_logits.clone()
                    start_ev.record()
                    if has_filter:
                        _ = fused_topk_topp_sample(
                            l_in,
                            sc.expanded_idx_mapping,
                            sc.temp,
                            sc.seeds,
                            sc.pos,
                            k=sc.k,
                            p=sc.p,
                        )
                    else:
                        _ = gumbel_sample(
                            l_in,
                            sc.expanded_idx_mapping,
                            sc.temp,
                            sc.seeds,
                            sc.pos,
                            apply_temperature=False,
                            is_drafting=False,
                            use_fp64=False,
                        )
                    end_ev.record()
                    end_ev.synchronize()
                    fused_times.append(start_ev.elapsed_time(end_ev))

                fused_med = sorted(fused_times)[len(fused_times) // 2]
                speedup = stock_med / max(fused_med, 1e-6)

                # --- 3. FlashInfer (when applicable) ---
                fi_med_str = "N/A"
                if sc.k is not None or sc.p is not None:
                    try:
                        for _ in range(warmup_iters):
                            _ = flashinfer_sample(proc_logits.clone(), sc.k, sc.p)
                        torch.cuda.synchronize()

                        fi_times = []
                        for _ in range(bench_iters):
                            flush_l2_cache(device=device)
                            l_in = proc_logits.clone()
                            start_ev.record()
                            _ = flashinfer_sample(l_in, sc.k, sc.p)
                            end_ev.record()
                            end_ev.synchronize()
                            fi_times.append(start_ev.elapsed_time(end_ev))
                        fi_med = sorted(fi_times)[len(fi_times) // 2]
                        fi_med_str = f"{fi_med:>6.3f} ms"
                    except Exception:
                        fi_med_str = "N/A"

                # Roundtrip HBM bandwidth saved (writing + re-reading logits)
                # 2 passes * (batch_size * vocab_size * 4 bytes)
                bytes_saved = 2 * (b * v * 4)
                mb_saved = bytes_saved / (1024 * 1024)

                s_str = f"{stock_med:>8.3f} ms"
                f_str = f"{fused_med:>8.3f} ms"
                sp_str = f"{speedup:>6.2f}x"
                mb_str = f"{mb_saved:>6.1f} MB"

                print(
                    f"{sc.name:<18} | {b:<5} | {v // 1000:>3}k    | "
                    f"{s_str} | {f_str} | {sp_str} | {fi_med_str:<10} | {mb_str}"
                )

    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Fused Sampler vs Stock cuRITA Benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Batch sizes to evaluate (default: 1 4 16 64)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[32768, 131072, 152064],
        help="Vocab sizes to evaluate (32k Llama 2, 128k Llama 3, 152k Qwen 2.5)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run bit-for-bit equivalence verification only",
    )
    parser.add_argument(
        "--bench-only",
        action="store_true",
        help="Run latency benchmark only",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required. Run this script on an NVIDIA GPU node.")
        return

    dev_name = torch.cuda.get_device_name(0)
    print(f"\n[Hardware Init] GPU: {dev_name} | PyTorch: {torch.__version__}")

    if not args.bench_only:
        passed = verify_bit_for_bit(args.batch_sizes, args.vocab_sizes)
        if not passed and args.verify_only:
            return

    if not args.verify_only:
        run_latency_benchmarks(args.batch_sizes, args.vocab_sizes)


if __name__ == "__main__":
    main()
