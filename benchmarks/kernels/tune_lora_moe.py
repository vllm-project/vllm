# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tune fused MoE LoRA BF16 Triton kernel configs.

Sweeps kernel configurations for the fused (shrink+expand) LoRA MoE
kernel using coordinate descent on the fused benchmark, and saves the
best config per batch size to a JSON file loadable at runtime via
VLLM_TUNED_CONFIG_FOLDER.

Coordinate descent approach:
  1. For each shared BLOCK_SIZE_M, seed expand with a default config.
  2. Sweep all shrink configs with the fused kernel → pick best shrink.
  3. Sweep all expand configs with best shrink fixed → pick best expand.
  4. Repeat for --num-cd-rounds rounds.

Usage examples:

  # Tune for a Nemotron-3-Nano-like gated MoE model
  python tune_lora_moe.py \\
      --hidden-size 2688 --intermediate-size 1856 \\
      --lora-rank 16 --max-loras 16 --top-k 6 --num-experts 128 \\
      --batch-sizes 1 16 64 256 1024 2048 \\
      --w13-slices 1 --save-dir ./tuned_lora_configs

  # Tune for a Mixtral-like model with 2 gate/up slices
  python tune_lora_moe.py \\
      --hidden-size 4096 --intermediate-size 14336 \\
      --lora-rank 16 --max-loras 4 --top-k 2 --num-experts 8 \\
      --batch-sizes 1 16 64 256 1024 \\
      --save-dir ./tuned_lora_configs

Then set the env var before launching vLLM:
  export VLLM_TUNED_CONFIG_FOLDER=./tuned_lora_configs

Known limitations (benchmark vs real serve discrepancies):

  1. Naive vs sorted routing threshold: The serve uses the same
     _should_use_naive() check as this benchmark. Both w13 and w2 share
     the same routing decision (made once per step). The threshold is:
       num_tokens * top_k * 8 <= num_experts * max_loras
     For small batch sizes (e.g., bs <= 21 for Nemotron-3-Nano with
     128 experts, 8 loras, top_k=6), the serve uses naive routing
     (sorted_token_ids=None). The benchmark correctly handles both paths.

  2. W2 qcurr_hidden_states has M*top_k rows: In the same decode step,
     W13's qcurr has shape (M, hidden_size) but W2's qcurr has shape
     (M*top_k, intermediate_size) — one row per token-expert pair.
     The kernel's M is always topk_weights.shape[0] = M (not M*top_k),
     but qcurr is indexed by sorted_token_ids which range up to M*top_k.
     The benchmark now creates qcurr with M*top_k rows for W2 to match
     the serve's intermediate_cache2 shape.
"""

# Ensure TMA descriptors are importable for TMA-enabled kernels
import contextlib
import gc
import json
import os
import time
from itertools import product

import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops.fused_moe_lora_op import (
    _LORA_PTR_DICT,
    _fused_moe_lora,
)
from vllm.lora.ops.triton_ops.utils import (
    _LORA_A_PTR_DICT,
    _LORA_B_PTR_DICT,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import next_power_of_2, round_up

with contextlib.suppress(ImportError):
    import triton.tools.tensor_descriptor  # noqa: F401

SPARSITY_FACTOR = 8


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _should_use_naive(
    num_tokens: int, top_k: int, num_experts: int, max_loras: int
) -> bool:
    # This matches the serve's decision logic in
    # vllm/lora/layers/fused_moe.py act_decorator (line 213-217).
    # Both w13 and w2 share the same routing decision per step.
    return num_tokens * top_k * SPARSITY_FACTOR <= num_experts * max_loras


def _build_routing(
    batch_size, top_k, num_experts, active_loras, routing="uniform", num_expert_hit=None
):
    """Build token→lora and token→expert routing tables.

    Args:
        routing: "uniform" spreads experts evenly per lora (deterministic).
                 "random" picks top_k random experts per token.
        num_expert_hit: If set, limit expert selection to this many experts
                        (sampled from the full expert pool). This mimics
                        real MoE gate routing where only a subset of experts
                        receive tokens. E.g., num_expert_hit=30 means only
                        30 out of 128 experts get any tokens. Works with
                        both "uniform" and "random" routing modes.
    """
    import random as _random

    token_lora_mapping = []
    topk_experts = []

    # Optionally restrict to a subset of experts
    effective_experts = num_experts
    expert_pool = list(range(num_experts))
    if num_expert_hit is not None and num_expert_hit < num_experts:
        rng_pool = _random.Random(123)
        expert_pool = sorted(rng_pool.sample(expert_pool, num_expert_hit))
        effective_experts = num_expert_hit

    if routing == "uniform":
        tokens_per_lora = [0] * active_loras
        for t in range(batch_size):
            tokens_per_lora[t % active_loras] += 1
        lora_strides = []
        for li in range(active_loras):
            picks = tokens_per_lora[li] * top_k
            if picks == 0 or picks > effective_experts:
                lora_strides.append(1)
            else:
                lora_strides.append(max(1, effective_experts // picks))
        lora_cursors = [0] * active_loras
        for t in range(batch_size):
            lid = t % active_loras
            token_lora_mapping.append(lid)
            experts = []
            for _ in range(top_k):
                experts.append(expert_pool[lora_cursors[lid] % effective_experts])
                lora_cursors[lid] += lora_strides[lid]
            topk_experts.append(experts)
    else:  # random
        rng = _random.Random(42)
        for t in range(batch_size):
            lid = t % active_loras
            token_lora_mapping.append(lid)
            topk_experts.append(rng.sample(expert_pool, min(top_k, effective_experts)))

    return token_lora_mapping, topk_experts


def _make_tensors(
    batch_size,
    hidden_size,
    intermediate_size,
    lora_rank,
    max_loras,
    num_experts,
    top_k,
    num_slices,
    dtype,
    device,
    active_loras=None,
    routing="uniform",
    mul_routed_weight=False,
    num_expert_hit=None,
):
    """Create synthetic tensors for benchmarking."""
    if active_loras is None:
        active_loras = max_loras
    M = batch_size
    lora_map, topk_experts = _build_routing(
        M,
        top_k,
        num_experts,
        active_loras,
        routing=routing,
        num_expert_hit=num_expert_hit,
    )

    token_lora_mapping = torch.tensor(lora_map, dtype=torch.int32, device=device)
    topk_ids = torch.tensor(topk_experts, dtype=torch.int32, device=device)

    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)
    unique_ids = token_lora_mapping.unique()
    lora_ids = torch.full((max_loras,), -1, dtype=torch.int32, device=device)
    lora_ids[: unique_ids.shape[0]] = unique_ids
    # The serve always uses max_loras + 1 as num_active_loras (grid_z dim)
    # regardless of how many loras are actually active, because
    # specialize_active_lora defaults to False (see lora_kernel_metadata.py:84).
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32)

    # W2 qcurr has M*top_k rows (one per token-expert pair from MoE
    # intermediate cache), W13 qcurr has M rows (one per token).
    qcurr_rows = M * top_k if mul_routed_weight else M
    qcurr = torch.randn(qcurr_rows, hidden_size, dtype=dtype, device=device)
    lora_a = [
        torch.randn(
            max_loras, num_experts, lora_rank, hidden_size, dtype=dtype, device=device
        )
        for _ in range(num_slices)
    ]
    lora_b = [
        torch.randn(
            max_loras,
            num_experts,
            intermediate_size,
            lora_rank,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_slices)
    ]
    expand_out = torch.zeros(
        M, top_k, intermediate_size * num_slices, dtype=dtype, device=device
    )

    return {
        "qcurr": qcurr,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "expand_out": expand_out,
        "token_lora_mapping": token_lora_mapping,
        "adapter_enabled": adapter_enabled,
        "lora_ids": lora_ids,
        "num_active_loras": num_active_loras,
        "topk_ids": topk_ids,
        "M": M,
        "num_tokens": M * top_k,
    }


def _prepare_moe_data(tensors, block_size_m, top_k, num_experts, max_loras, device):
    """Run moe_lora_align_block_size to get sorted token assignments."""
    M = tensors["M"]
    topk_ids = tensors["topk_ids"]
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size_m - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size_m)

    sorted_ids = torch.empty(
        max_loras * max_num_tokens_padded, dtype=torch.int32, device=device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size_m)
    expert_ids = torch.empty(
        max_loras * max_num_m_blocks, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty(max_loras, dtype=torch.int32, device=device)

    ops.moe_lora_align_block_size(
        topk_ids,
        tensors["token_lora_mapping"],
        num_experts,
        block_size_m,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        tensors["adapter_enabled"],
        tensors["lora_ids"],
    )

    topk_weights = torch.rand(M, top_k, device=device, dtype=torch.float32)
    sorted_token_ids = sorted_ids.view(max_loras, -1)
    expert_ids = expert_ids.view(max_loras, -1)
    return topk_weights, sorted_token_ids, expert_ids, num_tokens_post_pad


def _clear_caches():
    _LORA_PTR_DICT.clear()
    _LORA_A_PTR_DICT.clear()
    _LORA_B_PTR_DICT.clear()


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def _timed(fn, kwargs, num_warmup=3, num_iters=50):
    """Time fn with CUDA graph capture. Returns us per iteration."""
    for _ in range(num_warmup):
        fn(**kwargs)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(**kwargs)
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        graph.replay()
    end.record()
    end.synchronize()

    del graph
    torch.accelerator.synchronize()

    return start.elapsed_time(end) / num_iters * 1000  # us


def _bench_fused(
    tensors,
    shrink_cfg,
    expand_cfg,
    top_k,
    num_experts,
    max_loras,
    lora_rank,
    num_slices,
    block_size_m,
    num_iters,
    device,
    mul_routed_weight=False,
):
    """Benchmark the fused (shrink+expand) kernel. Returns us."""
    M = tensors["M"]
    naive = _should_use_naive(M, top_k, num_experts, max_loras)

    if naive:
        topk_weights = torch.rand(M, top_k, device=device, dtype=torch.float32)
        expert_ids = tensors["topk_ids"].view(-1)
        sorted_token_ids = None
        ntpp = None
    else:
        topk_weights, sorted_token_ids, expert_ids, ntpp = _prepare_moe_data(
            tensors, block_size_m, top_k, num_experts, max_loras, device
        )

    expand_out = tensors["expand_out"].zero_()
    _clear_caches()

    kwargs = dict(
        output=expand_out,
        qcurr_hidden_states=tensors["qcurr"],
        lora_a_stacked=tensors["lora_a"],
        lora_b_stacked=tensors["lora_b"],
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=ntpp,
        token_lora_mapping=tensors["token_lora_mapping"],
        max_lora_rank=lora_rank,
        top_k_num=top_k,
        lora_ids=tensors["lora_ids"],
        num_active_loras=tensors["num_active_loras"],
        adapter_enabled=tensors["adapter_enabled"],
        shrink_block_size_m=shrink_cfg["block_m"],
        shrink_block_size_n=shrink_cfg["block_n"],
        shrink_block_size_k=shrink_cfg["block_k"],
        shrink_group_size_m=shrink_cfg["group_size_m"],
        shrink_num_warps=shrink_cfg["num_warps"],
        shrink_num_stages=shrink_cfg["num_stages"],
        shrink_split_k=shrink_cfg["split_k"],
        expand_block_size_m=expand_cfg["block_m"],
        expand_block_size_n=expand_cfg["block_n"],
        expand_block_size_k=expand_cfg["block_k"],
        expand_group_size_m=expand_cfg["group_size_m"],
        expand_num_warps=expand_cfg["num_warps"],
        expand_num_stages=expand_cfg["num_stages"],
        expand_split_k=expand_cfg["split_k"],
        mul_routed_weight=mul_routed_weight,
        fully_sharded=False,
        offset=0,
    )

    try:
        return _timed(_fused_moe_lora, kwargs, num_iters=num_iters)
    except Exception as e:
        print(f"    [ERR] {e}")
        with contextlib.suppress(Exception):
            torch.accelerator.synchronize()
        return float("inf")


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def _get_search_space(op_type, lora_rank, hidden_size, intermediate_size):
    block_m_range = [16, 32]
    num_warps_range = [4]
    group_m_range = [1, 4, 8, 16, 32, 64]
    num_stages_range = [2, 3, 4]
    split_k_range = [1]
    rank_po2 = next_power_of_2(lora_rank)

    if op_type == "shrink":
        block_n_range = [16, 32, 64, 128, 256]
        block_k_range = [32, 64, 128]
        split_k_range = [1, 2, 4, 8, 16]
    else:
        block_k_range = [v for v in [16, 32] if v <= rank_po2]
        if not block_k_range:
            block_k_range = [max(16, rank_po2)]
        block_n_range = [64, 128, 256]

    configs = []
    for bm, bn, bk, gm, nw, ns, sk in product(
        block_m_range,
        block_n_range,
        block_k_range,
        group_m_range,
        num_warps_range,
        num_stages_range,
        split_k_range,
    ):
        if bm * bn < 64:
            continue
        lds = bk * bm * 2 + bk * bn * 2
        if lds > 65536:
            continue
        if op_type == "shrink" and sk > 1:
            k_dim = hidden_size
            if k_dim % (sk * bk) != 0:
                continue
        configs.append(
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "group_size_m": gm,
                "num_warps": nw,
                "num_stages": ns,
                "split_k": sk,
            }
        )
    return configs


# ---------------------------------------------------------------------------
# Coordinate descent tuning
# ---------------------------------------------------------------------------
def tune_fused(
    batch_sizes,
    hidden_size,
    intermediate_size,
    lora_rank,
    max_loras,
    top_k,
    num_experts,
    num_slices,
    dtype,
    num_iters,
    num_cd_rounds,
    active_loras,
    prefix="w13",
    routing="uniform",
    num_expert_hit=None,
):
    """Tune shrink+expand together via coordinate descent on fused kernel."""
    is_w13 = prefix == "w13"
    # w2 (down proj) applies routing weights during LoRA; w13 does not
    mul_routed_weight = not is_w13
    # shrink input dim (K): w13 uses hidden_size, w2 uses intermediate_size
    # expand output dim (N): w13 uses intermediate_size * num_slices,
    # w2 uses hidden_size
    if is_w13:
        shrink_K = hidden_size
        expand_N = intermediate_size * num_slices
    else:
        shrink_K = intermediate_size
        expand_N = hidden_size
    device = torch.device("cuda")

    shrink_space = _get_search_space("shrink", lora_rank, shrink_K, expand_N)
    expand_space = _get_search_space("expand", lora_rank, shrink_K, expand_N)
    block_m_values = sorted(
        set(c["block_m"] for c in shrink_space)
        & set(c["block_m"] for c in expand_space)
    )

    print(f"  Shrink search: {len(shrink_space)} configs")
    print(f"  Expand search: {len(expand_space)} configs")
    print(f"  Shared BLOCK_SIZE_M: {block_m_values}")

    shrink_best_all = {}
    expand_best_all = {}

    for bs in batch_sizes:
        print(f"\n  Tuning {prefix}_fused | batch_size={bs}")

        tensors = _make_tensors(
            bs,
            shrink_K,
            expand_N,
            lora_rank,
            max_loras,
            num_experts,
            top_k,
            num_slices,
            dtype,
            device,
            active_loras=active_loras,
            routing=routing,
            mul_routed_weight=mul_routed_weight,
            num_expert_hit=num_expert_hit,
        )

        best_fused = float("inf")
        best_s = None
        best_e = None

        for bm in block_m_values:
            s_cands = [c for c in shrink_space if c["block_m"] == bm]
            e_cands = [c for c in expand_space if c["block_m"] == bm]
            print(f"\n    block_m={bm}: {len(s_cands)} shrink, {len(e_cands)} expand")

            e_sorted = sorted(e_cands, key=lambda c: (c["block_n"], c["block_k"]))
            cur_e = e_sorted[len(e_sorted) // 2]
            cur_s = None
            bm_best = float("inf")

            for rd in range(num_cd_rounds):
                # Sweep shrink
                best_s_t = float("inf")
                for i, sc in enumerate(s_cands):
                    _clear_caches()
                    t = _bench_fused(
                        tensors,
                        sc,
                        cur_e,
                        top_k,
                        num_experts,
                        max_loras,
                        lora_rank,
                        num_slices,
                        bm,
                        num_iters,
                        device,
                        mul_routed_weight=mul_routed_weight,
                    )
                    if t < best_s_t:
                        best_s_t = t
                        cur_s = sc.copy()
                    if t < bm_best:
                        bm_best = t
                    if (i + 1) % 50 == 0 or i == len(s_cands) - 1:
                        print(
                            f"      shrink [{i + 1}/{len(s_cands)}] "
                            f"best={best_s_t:.1f} us"
                        )
                    if (i + 1) % 100 == 0:
                        gc.collect()
                        torch.accelerator.empty_cache()

                # Sweep expand
                best_e_t = float("inf")
                for i, ec in enumerate(e_cands):
                    _clear_caches()
                    t = _bench_fused(
                        tensors,
                        cur_s,
                        ec,
                        top_k,
                        num_experts,
                        max_loras,
                        lora_rank,
                        num_slices,
                        bm,
                        num_iters,
                        device,
                        mul_routed_weight=mul_routed_weight,
                    )
                    if t < best_e_t:
                        best_e_t = t
                        cur_e = ec.copy()
                    if t < bm_best:
                        bm_best = t
                    if (i + 1) % 50 == 0 or i == len(e_cands) - 1:
                        print(
                            f"      expand [{i + 1}/{len(e_cands)}] "
                            f"best={best_e_t:.1f} us"
                        )
                    if (i + 1) % 100 == 0:
                        gc.collect()
                        torch.accelerator.empty_cache()

            if bm_best < best_fused:
                best_fused = bm_best
                best_s = cur_s
                best_e = cur_e

            print(f"    block_m={bm}: {bm_best:.1f} us (global best={best_fused:.1f})")

        print(f"  => bs={bs}: {best_fused:.1f} us")
        print(f"     shrink={best_s}")
        print(f"     expand={best_e}")
        shrink_best_all[bs] = best_s
        expand_best_all[bs] = best_e

        del tensors
        gc.collect()
        torch.accelerator.empty_cache()

    return shrink_best_all, expand_best_all


# ---------------------------------------------------------------------------
# Save configs (compatible with get_lora_op_configs JSON format)
# ---------------------------------------------------------------------------
def _save_configs(
    best_configs,
    op_type,
    max_loras,
    num_slices,
    lora_rank,
    hidden_size,
    intermediate_size,
    save_dir,
):
    loras_key = str(max_loras)
    slices_key = str(num_slices)
    k_key = str(lora_rank)
    n_key = str(hidden_size)
    i_key = str(intermediate_size)

    result = {loras_key: {slices_key: {}}}
    for bs, config in best_configs.items():
        m_key = str(bs)
        result[loras_key][slices_key][m_key] = {
            k_key: {
                n_key: {
                    i_key: {
                        "BLOCK_SIZE_M": config["block_m"],
                        "BLOCK_SIZE_N": config["block_n"],
                        "BLOCK_SIZE_K": config["block_k"],
                        "GROUP_SIZE_M": config["group_size_m"],
                        "num_warps": config["num_warps"],
                        "num_stages": config["num_stages"],
                        "split_k": config["split_k"],
                    }
                }
            }
        }

    file_op_type = f"fused_moe_lora_{op_type}"
    gpu_name = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    filename = f"{gpu_name}_{file_op_type.upper()}.json"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    # Merge with existing file if present
    if os.path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
        if loras_key not in existing:
            existing[loras_key] = {}
        if slices_key not in existing[loras_key]:
            existing[loras_key][slices_key] = {}
        existing[loras_key][slices_key].update(result[loras_key][slices_key])
        result = existing

    with open(filepath, "w") as f:
        json.dump(result, f, indent=4)
        f.write("\n")
    print(f"  Saved to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("=" * 60)
    print("Fused MoE LoRA BF16 Kernel Tuning")
    print("=" * 60)
    print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  hidden_size   : {args.hidden_size}")
    print(f"  intermediate  : {args.intermediate_size}")
    print(f"  lora_rank     : {args.lora_rank}")
    print(f"  max_loras     : {args.max_loras}")
    print(f"  active_loras  : {args.active_loras or args.max_loras}")
    print(f"  routing       : {args.routing}")
    print(f"  top_k         : {args.top_k}")
    print(f"  num_experts   : {args.num_experts}")
    print(f"  expert_hit    : {args.num_expert_hit or args.num_experts}")
    print(f"  w13_slices    : {args.w13_slices}")
    print(f"  batch_sizes   : {args.batch_sizes}")
    print(f"  op_types      : {args.op_types}")
    print(f"  num_iters     : {args.num_iters}")
    print(f"  num_cd_rounds : {args.num_cd_rounds}")
    print(f"  save_dir      : {args.save_dir}")
    print()

    start = time.time()

    for op in args.op_types:
        prefix = op.replace("_fused", "")
        is_w13 = prefix == "w13"
        num_slices = args.w13_slices if is_w13 else 1

        print(f"\n{'=' * 60}")
        print(f"Tuning: {op} (num_slices={num_slices})")
        print(f"{'=' * 60}")

        shrink_best, expand_best = tune_fused(
            batch_sizes=args.batch_sizes,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            lora_rank=args.lora_rank,
            max_loras=args.max_loras,
            top_k=args.top_k,
            num_experts=args.num_experts,
            num_slices=num_slices,
            dtype=torch.bfloat16,
            num_iters=args.num_iters,
            num_cd_rounds=args.num_cd_rounds,
            active_loras=args.active_loras,
            routing=args.routing,
            num_expert_hit=args.num_expert_hit,
            prefix=prefix,
        )

        _save_configs(
            shrink_best,
            f"{prefix}_shrink",
            args.max_loras,
            num_slices,
            args.lora_rank,
            args.hidden_size,
            args.intermediate_size,
            args.save_dir,
        )
        _save_configs(
            expand_best,
            f"{prefix}_expand",
            args.max_loras,
            num_slices,
            args.lora_rank,
            args.hidden_size,
            args.intermediate_size,
            args.save_dir,
        )

    elapsed = time.time() - start
    print(f"\nTotal tuning time: {elapsed:.1f}s")
    print(f"Configs saved to: {os.path.abspath(args.save_dir)}")
    print(f"\nTo use: export VLLM_TUNED_CONFIG_FOLDER={os.path.abspath(args.save_dir)}")


# ---------------------------------------------------------------------------
# Benchmark mode: load saved configs and measure latency
# ---------------------------------------------------------------------------
def _load_config(
    filepath,
    max_loras,
    num_slices,
    lora_rank,
    hidden_size,
    intermediate_size,
    batch_size,
):
    """Load a single config from a tuned JSON file."""
    with open(filepath) as f:
        raw = json.load(f)

    def _nearest(d, target):
        return min(d.keys(), key=lambda x: abs(int(x) - target))

    lk, sk = str(max_loras), str(num_slices)
    mk, kk = str(batch_size), str(lora_rank)
    nk, ik = str(hidden_size), str(intermediate_size)

    data = raw.get(lk, raw[_nearest(raw, max_loras)])
    data = data.get(sk, data[_nearest(data, num_slices)])
    data = data.get(mk, data[_nearest(data, batch_size)])
    data = data.get(kk, data[_nearest(data, lora_rank)])
    data = data.get(nk, data[_nearest(data, hidden_size)])
    cfg = data.get(ik, data[_nearest(data, intermediate_size)])

    _km = {
        "BLOCK_SIZE_M": "block_m",
        "BLOCK_SIZE_N": "block_n",
        "BLOCK_SIZE_K": "block_k",
        "GROUP_SIZE_M": "group_size_m",
    }
    out = {_km.get(k, k): v for k, v in cfg.items()}
    out.setdefault("split_k", 1)
    return out


def benchmark_configs(args):
    """Load tuned configs and benchmark each batch size."""
    device = torch.device("cuda")
    config_dir = args.config_dir or args.save_dir
    print("=" * 60)
    print("Fused MoE LoRA BF16 Kernel Benchmark")
    print("=" * 60)
    print(f"  GPU           : {torch.cuda.get_device_name()}")
    print(f"  config_dir    : {config_dir}")
    print(f"  routing       : {args.routing}")
    print(f"  num_iters     : {args.num_iters}")
    print()

    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")

    for op in args.op_types:
        prefix = op.replace("_fused", "")
        is_w13 = prefix == "w13"
        ns = args.w13_slices if is_w13 else 1
        if is_w13:
            shrink_K = args.hidden_size
            eff_i = args.intermediate_size * ns
        else:
            shrink_K = args.intermediate_size
            eff_i = args.hidden_size

        sf = os.path.join(
            config_dir, f"{gpu}_FUSED_MOE_LORA_{prefix.upper()}_SHRINK.json"
        )
        ef = os.path.join(
            config_dir, f"{gpu}_FUSED_MOE_LORA_{prefix.upper()}_EXPAND.json"
        )

        if not os.path.exists(sf) or not os.path.exists(ef):
            print(f"  [SKIP] {op}: config files not found in {config_dir}")
            continue

        print(f"  {op} (slices={ns})")
        print(f"  {'bs':>6} | {'lat_us':>10} | shrink | expand")
        print("  " + "-" * 70)

        for bs in args.batch_sizes:
            sc = _load_config(
                sf,
                args.max_loras,
                ns,
                args.lora_rank,
                args.hidden_size,
                args.intermediate_size,
                bs,
            )
            ec = _load_config(
                ef,
                args.max_loras,
                ns,
                args.lora_rank,
                args.hidden_size,
                args.intermediate_size,
                bs,
            )
            mul_rw = not is_w13  # w2 applies routing weights
            t = _make_tensors(
                bs,
                shrink_K,
                eff_i,
                args.lora_rank,
                args.max_loras,
                args.num_experts,
                args.top_k,
                ns,
                torch.bfloat16,
                device,
                active_loras=args.active_loras,
                routing=args.routing,
                mul_routed_weight=mul_rw,
                num_expert_hit=args.num_expert_hit,
            )
            lat = _bench_fused(
                t,
                sc,
                ec,
                args.top_k,
                args.num_experts,
                args.max_loras,
                args.lora_rank,
                ns,
                sc["block_m"],
                args.num_iters,
                device,
                mul_routed_weight=mul_rw,
            )
            ss = (
                f"M{sc['block_m']}_N{sc['block_n']}"
                f"_K{sc['block_k']}_s{sc['num_stages']}"
                f"_sk{sc['split_k']}"
            )
            es = (
                f"M{ec['block_m']}_N{ec['block_n']}"
                f"_K{ec['block_k']}_s{ec['num_stages']}"
            )
            print(f"  {bs:>6} | {lat:>10.1f} | {ss} | {es}")
            del t
            gc.collect()
            torch.accelerator.empty_cache()
        print()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Tune fused MoE LoRA BF16 Triton kernel configs"
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument(
        "--intermediate-size",
        type=int,
        required=True,
        help="MoE intermediate size per partition",
    )
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--max-loras", type=int, default=4)
    parser.add_argument(
        "--active-loras",
        type=int,
        default=None,
        help="Active LoRAs (default: max-loras). Set to 1 for prefill simulation.",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--w13-slices",
        type=int,
        default=2,
        choices=[1, 2],
        help="1=non-gated (Nemotron), 2=gated (Mixtral)",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--op-types",
        nargs="+",
        default=["w13_fused"],
        choices=["w13_fused", "w2_fused"],
    )
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--num-cd-rounds", type=int, default=2)
    parser.add_argument(
        "--routing",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
        help="Expert routing: uniform (default) or random",
    )
    parser.add_argument(
        "--num-expert-hit",
        type=int,
        default=None,
        help="Number of experts that receive tokens "
        "(default: all). E.g., --num-expert-hit 30 "
        "for Nemotron-3-Nano real routing sparsity.",
    )
    parser.add_argument("--save-dir", type=str, default="./tuned_lora_configs")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark mode: load configs from --config-dir "
        "(or --save-dir) and measure latency",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory to load tuned configs from (benchmark mode)",
    )
    args = parser.parse_args()
    if args.benchmark:
        benchmark_configs(args)
    else:
        main(args)
