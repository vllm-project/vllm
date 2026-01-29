#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune Triton WNA16 MoE configs for int4 W4A16 kernels.

Example (GLM-4.5 MoE on V100, TP=2):
  python benchmarks/kernels/tune_moe_wna16_triton.py \
    --model MOE_MODEL_NAME/ \
    --tp-size 2 \
    --group-size 32 \
    --batch-sizes 1 2 4 8 16 32 64 \
    --save-dir vllm/model_executor/layers/fused_moe/configs
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_config_file_name,
    invoke_fused_moe_wna16_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.transformers_utils.config import get_config
from vllm.triton_utils import tl


def _read_model_config(
    model: str | None, trust_remote_code: bool
) -> tuple[int, int, int, int] | None:
    if model is None:
        return None
    cfg = get_config(model, trust_remote_code=trust_remote_code)
    archs = getattr(cfg, "architectures", None) or []

    if any(name.startswith("Glm4Moe") for name in archs):
        return (
            int(cfg.n_routed_experts),
            int(cfg.num_experts_per_tok),
            int(cfg.moe_intermediate_size),
            int(cfg.hidden_size),
        )
    if hasattr(cfg, "num_local_experts") and hasattr(cfg, "num_experts_per_tok"):
        return (
            int(cfg.num_local_experts),
            int(cfg.num_experts_per_tok),
            int(cfg.intermediate_size),
            int(cfg.hidden_size),
        )
    if hasattr(cfg, "num_experts") and hasattr(cfg, "num_experts_per_tok"):
        return (
            int(cfg.num_experts),
            int(cfg.num_experts_per_tok),
            int(cfg.moe_intermediate_size),
            int(cfg.hidden_size),
        )
    return None


def _make_weights(
    device: torch.device,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    group_size: int,
    gate_up_multiplier: int,
    dtype: torch.dtype,
):
    w1_n = gate_up_multiplier * intermediate_size_per_partition
    w2_k = intermediate_size_per_partition

    # Packed int4 weights are uint8 with 2 values per byte.
    w1 = torch.randint(
        0,
        256,
        (num_experts, w1_n, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden_size, w2_k // 2),
        device=device,
        dtype=torch.uint8,
    )

    w1_scale = torch.rand(
        (num_experts, w1_n, hidden_size // group_size),
        device=device,
        dtype=dtype,
    )
    w2_scale = torch.rand(
        (num_experts, hidden_size, w2_k // group_size),
        device=device,
        dtype=dtype,
    )

    return w1, w2, w1_scale, w2_scale


def _get_align_cache(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size_m: int,
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size_m in cache:
        return cache[block_size_m]
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        num_experts,
        expert_map=None,
        ignore_invalid_experts=True,
    )
    cache[block_size_m] = (sorted_token_ids, expert_ids, num_tokens_post_padded)
    return cache[block_size_m]


def _run_once(
    config: dict[str, Any],
    hidden_states: torch.Tensor,
    inter_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    group_size: int,
    compute_type: tl.dtype,
    align_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
):
    block_size_m = config["BLOCK_SIZE_M"]
    sorted_token_ids, expert_ids, num_tokens_post_padded = _get_align_cache(
        topk_ids, w1.size(0), block_size_m, align_cache
    )

    # W1: [M, H] x [E, 2I, H/2] -> [M, topk, 2I]
    out_w1 = torch.empty(
        (hidden_states.size(0), topk, w1.size(1)),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_wna16_triton_kernel(
        hidden_states,
        w1,
        out_w1,
        w1_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        topk,
        config,
        compute_type=compute_type,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=[0, group_size],
    )

    # W2: [M*topk, I] x [E, H, I/2] -> [M*topk, 1, H]
    out_w2 = torch.empty(
        (hidden_states.size(0), topk, w2.size(1)),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_wna16_triton_kernel(
        inter_states,
        w2,
        out_w2,
        w2_scale,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        1,
        config,
        compute_type=compute_type,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=[0, group_size],
    )


def _benchmark_config(
    config: dict[str, Any],
    hidden_states: torch.Tensor,
    inter_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    group_size: int,
    compute_type: tl.dtype,
    warmup: int,
    iters: int,
    align_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    for _ in range(warmup):
        _run_once(
            config,
            hidden_states,
            inter_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            topk,
            group_size,
            compute_type,
            align_cache,
        )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _run_once(
            config,
            hidden_states,
            inter_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            topk,
            group_size,
            compute_type,
            align_cache,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    model_config = _read_model_config(args.model, args.trust_remote_code)
    if model_config is None:
        if None in (
            args.num_experts,
            args.topk,
            args.intermediate_size,
            args.hidden_size,
        ):
            raise ValueError(
                "Provide --model or all of --num-experts, --topk, "
                "--intermediate-size, --hidden-size."
            )
        num_experts = args.num_experts
        topk = args.topk
        intermediate_size = args.intermediate_size
        hidden_size = args.hidden_size
    else:
        num_experts, topk, intermediate_size, hidden_size = model_config

    if intermediate_size % args.tp_size != 0:
        raise ValueError("intermediate_size must be divisible by tp_size.")

    group_size = args.group_size
    inter_size_per_tp = intermediate_size // args.tp_size
    w1_n = args.gate_up_multiplier * inter_size_per_tp
    if hidden_size % 2 != 0 or inter_size_per_tp % 2 != 0:
        raise ValueError("hidden_size and intermediate_size must be even for int4.")
    if hidden_size % group_size != 0 or inter_size_per_tp % group_size != 0:
        raise ValueError(
            "hidden_size and intermediate_size must be divisible by group_size."
        )

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    compute_type = tl.float16 if dtype == torch.float16 else tl.bfloat16

    # Candidate search space
    block_ms = args.block_m
    block_ns = args.block_n
    block_ks = args.block_k
    num_warps_list = args.num_warps
    num_stages_list = args.num_stages

    candidates: list[dict[str, Any]] = []
    for bm, bn, bk, nw, ns in product(
        block_ms, block_ns, block_ks, num_warps_list, num_stages_list
    ):
        if bk % group_size != 0:
            continue
        if hidden_size % bk != 0 or inter_size_per_tp % bk != 0:
            continue
        if bn > w1_n or bn > hidden_size:
            continue
        candidates.append(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 1,
                "SPLIT_K": 1,
                "num_warps": nw,
                "num_stages": ns,
            }
        )

    if not candidates:
        raise RuntimeError("No valid candidates. Adjust block sizes.")

    w1, w2, w1_scale, w2_scale = _make_weights(
        device,
        num_experts,
        hidden_size,
        inter_size_per_tp,
        group_size,
        args.gate_up_multiplier,
        dtype,
    )

    results: dict[int, dict[str, Any]] = {}
    for m in args.batch_sizes:
        print(f"Tuning batch size M={m} with {len(candidates)} candidates...")
        hidden_states = torch.randn((m, hidden_size), device=device, dtype=dtype)
        inter_states = torch.randn(
            (m * topk, inter_size_per_tp), device=device, dtype=dtype
        )
        topk_ids = torch.randint(
            0, num_experts, (m, topk), device=device, dtype=torch.int32
        )
        topk_weights = torch.rand((m, topk), device=device, dtype=dtype)

        align_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        best_time = float("inf")
        best_cfg: dict[str, Any] | None = None
        for cfg in candidates:
            try:
                t = _benchmark_config(
                    cfg,
                    hidden_states,
                    inter_states,
                    w1,
                    w2,
                    w1_scale,
                    w2_scale,
                    topk_weights,
                    topk_ids,
                    topk,
                    group_size,
                    compute_type,
                    args.warmup,
                    args.iters,
                    align_cache,
                )
            except Exception as exc:
                print(f"  skip {cfg} -> {exc}")
                continue
            if t < best_time:
                best_time = t
                best_cfg = cfg
        if best_cfg is None:
            raise RuntimeError(f"No valid config for M={m}.")
        results[m] = best_cfg
        print(f"  best for M={m}: {best_cfg} ({best_time * 1e3:.3f} ms)")

    dtype_str = "int4_w4a16"
    file_name = get_config_file_name(num_experts, inter_size_per_tp, dtype_str)
    output_path = Path(args.save_dir) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    print(f"Saved tuned config to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--gate-up-multiplier", type=int, default=2)
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64]
    )
    parser.add_argument("--block-m", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--block-n", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--block-k", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--num-warps", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--num-stages", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="vllm/model_executor/layers/fused_moe/configs",
    )
    main(parser.parse_args())
