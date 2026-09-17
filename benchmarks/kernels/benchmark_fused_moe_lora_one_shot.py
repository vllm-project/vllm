# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused MoE-LoRA fast path (one-shot) vs two-kernel baseline.

The "one_shot" provider goes through `vllm.lora.ops.triton_ops.fused_moe_lora`
which dispatches to the single-kernel one-shot implementation when
fully_sharded=False (the prefill default).

The "two_kernel" provider drives `fused_moe_lora_shrink` + `fused_moe_lora_expand`
directly, bypassing the dispatch and matching the legacy two-kernel path's
work distribution. This isolates the win from kernel fusion.

Run:
    .venv/bin/python -m benchmarks.kernels.benchmark_fused_moe_lora_one_shot
    .venv/bin/python -m benchmarks.kernels.benchmark_fused_moe_lora_one_shot \\
        --model qwen3moe
"""

from __future__ import annotations

import argparse
import os
import random

import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops import (
    fused_moe_lora,
    fused_moe_lora_expand,
    fused_moe_lora_shrink,
)
from vllm.triton_utils import triton

DTYPE = torch.bfloat16
DEVICE = "cuda"


# ----- input fabrication -----------------------------------------------------


def _round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def _ceildiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _assign_loras(num_tokens: int, num_sequences: int, max_loras: int) -> torch.Tensor:
    tokens_per_seq = num_tokens // num_sequences
    rem = num_tokens % num_sequences
    out = torch.empty(num_tokens, dtype=torch.int32)
    start = 0
    for i in range(num_sequences):
        end = start + tokens_per_seq + (1 if i < rem else 0)
        out[start:end] = random.randint(0, max_loras - 1)
        start = end
    return out


def _assign_experts(num_tokens: int, num_experts: int, top_k: int):
    expert_indices = torch.empty((num_tokens, top_k), dtype=torch.int32)
    for i in range(num_tokens):
        expert_indices[i] = torch.randperm(num_experts)[:top_k]
    weights = torch.rand((num_tokens, top_k), dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return expert_indices, weights


def _make_inputs(
    M: int,
    K: int,
    N_per_slice: int,
    rank: int,
    num_experts: int,
    top_k: int,
    max_loras: int,
    num_slices: int,
    block_size_m: int,
):
    """Mirrors the production caller's tensor layout."""
    torch.manual_seed(0)
    random.seed(0)

    num_sequences = max(1, min(M, 8))
    topk_ids_cpu, topk_weights_cpu = _assign_experts(M, num_experts, top_k)
    token_lora_cpu = _assign_loras(M, num_sequences, max_loras)
    lora_ids_cpu = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    uniq = torch.unique(token_lora_cpu, sorted=True)
    lora_ids_cpu[: uniq.size(0)].copy_(uniq)

    topk_ids = topk_ids_cpu.to(DEVICE)
    topk_weights = topk_weights_cpu.to(device=DEVICE, dtype=DTYPE)
    token_lora_mapping = token_lora_cpu.to(DEVICE)
    lora_ids = lora_ids_cpu.to(DEVICE)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=DEVICE)

    lora_a = [
        torch.randn((max_loras, num_experts, rank, K), dtype=DTYPE, device=DEVICE)
        / max(K, 1) ** 0.5
        for _ in range(num_slices)
    ]
    lora_b = [
        torch.randn(
            (max_loras, num_experts, N_per_slice, rank),
            dtype=DTYPE,
            device=DEVICE,
        )
        / max(rank, 1) ** 0.5
        for _ in range(num_slices)
    ]
    hidden = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
    out_template = torch.zeros(
        (M, top_k, num_slices * N_per_slice), dtype=DTYPE, device=DEVICE
    )

    # Sorted-path metadata (the prefill default).
    max_pad = topk_ids.numel() + num_experts * (block_size_m - 1)
    max_pad = _round_up(max_pad, block_size_m)
    max_blocks = _ceildiv(max_pad, block_size_m)
    sorted_token_ids = torch.empty(
        (max_loras * max_pad,), dtype=torch.int32, device=DEVICE
    )
    expert_ids = torch.empty(
        (max_loras * max_blocks,), dtype=torch.int32, device=DEVICE
    )
    num_post = torch.empty((max_loras,), dtype=torch.int32, device=DEVICE)
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size_m,
        max_loras,
        max_pad,
        max_blocks,
        sorted_token_ids,
        expert_ids,
        num_post,
        adapter_enabled,
        lora_ids,
    )
    expert_ids = expert_ids.view(max_loras, -1).contiguous()
    sorted_token_ids = sorted_token_ids.view(max_loras, -1).contiguous()
    num_active = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    return dict(
        hidden=hidden,
        lora_a=lora_a,
        lora_b=lora_b,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_post=num_post,
        token_lora_mapping=token_lora_mapping,
        lora_ids=lora_ids,
        num_active=num_active,
        adapter_enabled=adapter_enabled,
        out_template=out_template,
        # bookkeeping
        M=M,
        K=K,
        N_per_slice=N_per_slice,
        rank=rank,
        num_experts=num_experts,
        top_k=top_k,
        max_loras=max_loras,
        num_slices=num_slices,
        block_size_m=block_size_m,
    )


# ----- providers -------------------------------------------------------------


def _run_one_shot(inp: dict):
    """Drive `fused_moe_lora` with fully_sharded=False -> one-shot fast path."""
    out = inp["out_template"].clone()
    fused_moe_lora(
        out,
        inp["hidden"],
        inp["lora_a"],
        inp["lora_b"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        inp["rank"],
        inp["top_k"],
        inp["lora_ids"],
        inp["num_active"],
        inp["adapter_enabled"],
        inp["block_size_m"],
        64,
        32,
        8,
        4,
        3,
        1,
        inp["block_size_m"],
        64,
        32,
        8,
        4,
        3,
        1,
        False,
        False,
        0,
    )
    return out


def _run_two_kernel(inp: dict):
    """Drive `fused_moe_lora_shrink` + `fused_moe_lora_expand` directly,
    bypassing the dispatch. Matches the legacy two-kernel work distribution.
    """
    M = inp["M"]
    top_k = inp["top_k"]
    rank = inp["rank"]
    num_slices = inp["num_slices"]
    N_per_slice = inp["N_per_slice"]
    K = inp["K"]
    num_experts = inp["num_experts"]
    block_m = inp["block_size_m"]

    intermediate = torch.zeros((num_slices, M, top_k, rank), dtype=DTYPE, device=DEVICE)
    out = inp["out_template"].clone()
    EM = inp["sorted_token_ids"].shape[1]
    num_tokens = M * top_k

    fused_moe_lora_shrink(
        intermediate,
        inp["hidden"],
        inp["lora_a"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        top_k,
        inp["lora_ids"],
        inp["adapter_enabled"],
        torch.device(DEVICE),
        rank,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        block_m,
        64,
        32,
        8,
        4,
        3,
        1,
        inp["num_active"],
        False,
    )
    fused_moe_lora_expand(
        out,
        intermediate,
        inp["lora_b"],
        inp["topk_weights"],
        inp["sorted_token_ids"],
        inp["expert_ids"],
        inp["num_post"],
        inp["token_lora_mapping"],
        top_k,
        inp["lora_ids"],
        inp["adapter_enabled"],
        torch.device(DEVICE),
        rank,
        M,
        EM,
        K,
        num_tokens,
        num_experts,
        num_slices,
        rank,
        N_per_slice,
        block_m,
        64,
        32,
        8,
        4,
        3,
        1,
        inp["num_active"],
        False,
        0,
    )
    return out


PROVIDER_FNS = {
    "one_shot": _run_one_shot,
    "two_kernel": _run_two_kernel,
}


# ----- model presets ---------------------------------------------------------


MODEL_PRESETS: dict[str, dict] = {
    # Mixtral-8x7B style: E=8, top_k=2, hidden=4096, intermediate=14336
    "mixtral": dict(
        K=4096,
        N_per_slice=7168,
        num_experts=8,
        top_k=2,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
    # Qwen3-MoE / DeepSeek-V2 style: E=64, top_k=8, hidden=2048, inter=1408
    "qwen3moe": dict(
        K=2048,
        N_per_slice=1408,
        num_experts=64,
        top_k=8,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
    # GLM-5.1 (zai-org/GLM-5.1-FP8): E=256, top_k=8, hidden=6144,
    # moe_intermediate=2048
    "glm5_1": dict(
        K=6144,
        N_per_slice=2048,
        num_experts=256,
        top_k=8,
        max_loras=4,
        num_slices=2,
        block_size_m=64,
    ),
}


M_RANGE = [16, 64, 256, 1024, 4096, 16384]
RANK_RANGE = [8, 16, 32, 64]


def get_benchmark(model: str, max_loras: int | None = None):
    preset = dict(MODEL_PRESETS[model])
    if max_loras is not None:
        preset["max_loras"] = max_loras

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "rank"],
            x_vals=[(M, R) for M in M_RANGE for R in RANK_RANGE],
            line_arg="provider",
            line_vals=list(PROVIDER_FNS.keys()),
            line_names=["one_shot (fused)", "two_kernel (legacy)"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"fused_moe_lora-{model}-loras{preset['max_loras']}",
            args={"preset": preset},
        )
    )
    def benchmark(M, rank, provider, preset):
        inp = _make_inputs(
            M=M,
            K=preset["K"],
            N_per_slice=preset["N_per_slice"],
            rank=rank,
            num_experts=preset["num_experts"],
            top_k=preset["top_k"],
            max_loras=preset["max_loras"],
            num_slices=preset["num_slices"],
            block_size_m=preset["block_size_m"],
        )
        fn = PROVIDER_FNS[provider]
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: fn(inp), quantiles=quantiles
        )
        return ms, max_ms, min_ms

    return benchmark


# ----- correctness sanity ---------------------------------------------------


def calculate_diff(model: str, M: int, rank: int, max_loras: int | None = None):
    preset = dict(MODEL_PRESETS[model])
    if max_loras is not None:
        preset["max_loras"] = max_loras
    inp = _make_inputs(
        M=M,
        K=preset["K"],
        N_per_slice=preset["N_per_slice"],
        rank=rank,
        num_experts=preset["num_experts"],
        top_k=preset["top_k"],
        max_loras=preset["max_loras"],
        num_slices=preset["num_slices"],
        block_size_m=preset["block_size_m"],
    )
    out_one = _run_one_shot(inp)
    out_two = _run_two_kernel(inp)
    max_abs = (out_one.float() - out_two.float()).abs().max().item()
    print(
        f"  model={model:<9} M={M:<6} rank={rank:<3}  "
        f"max|one_shot - two_kernel|={max_abs:.4g}  "
        f"ref|max|={out_two.float().abs().max().item():.3g}"
    )
    if max_abs <= 5e-2:
        print("  ✅ outputs match within bf16 tolerance")
    else:
        print("  ❌ outputs differ beyond expected bf16 noise")


# ----- main ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mixtral",
        choices=list(MODEL_PRESETS.keys()),
        help="Model preset to sweep",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/fused_moe_lora_one_shot/",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run correctness sanity check only, no perf sweep",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="Override max_loras in the model preset (number of LoRA adapters "
        "active in the batch). Defaults to the preset's value.",
    )
    args = parser.parse_args()

    print(f"Correctness check ({args.model}):")
    calculate_diff(args.model, M=256, rank=32, max_loras=args.max_loras)
    if args.check_only:
        raise SystemExit(0)

    effective_max_loras = (
        args.max_loras
        if args.max_loras is not None
        else MODEL_PRESETS[args.model]["max_loras"]
    )
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Model preset: {args.model}  max_loras={effective_max_loras}\n")
    benchmark = get_benchmark(args.model, max_loras=args.max_loras)
    os.makedirs(args.save_path, exist_ok=True)
    benchmark.run(print_data=True, save_path=args.save_path)
