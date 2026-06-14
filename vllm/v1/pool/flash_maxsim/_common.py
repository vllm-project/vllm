# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for flash-maxsim kernels: padding, autotune configs,
and pruning."""

import torch

from vllm.triton_utils import triton


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _get_configs() -> list:
    """Representative autotune grid spanning the (BLOCK_Q, BLOCK_D) space.
    Triton's autotuner picks the best surviving config per (Lq, Ld) bucket;
    `_prune_configs` filters by SMEM budget and Lq at runtime, so the same
    list is safe across GPUs. Per-GPU tuning ships separately as JSON."""
    return [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1),
    ]


def _prune_configs(configs, named_args, **kwargs):
    """Reject configs that exceed the GPU's shared-memory budget.

    BLOCK_Q and BLOCK_D load fp16 tiles of size [BLOCK_*, d]; the
    [BLOCK_Q, BLOCK_D] score tile is fp32.  Total:
        (BLOCK_Q*d + BLOCK_D*d)*2 + BLOCK_Q*BLOCK_D*4 bytes.
    """
    Lq = named_args.get("Lq", 32)
    d = named_args.get("d", 128)
    smem_limit = 200_000  # safe default for A100 (164 KB real)
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 9:  # Hopper+: 228 KB SMEM
            smem_limit = 220_000
    pruned = []
    for cfg in configs:
        bq = cfg.kwargs["BLOCK_Q"]
        bd = cfg.kwargs["BLOCK_D"]
        if bq > Lq * 2:
            continue
        if (bq * d + bd * d) * 2 + bq * bd * 4 > smem_limit:
            continue
        pruned.append(cfg)
    return pruned or configs[:4]
