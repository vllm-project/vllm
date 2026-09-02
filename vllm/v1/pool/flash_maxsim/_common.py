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


def _smem_budget() -> int:
    """Per-block opt-in shared-memory budget of the active device, in bytes.

    Queried from the Triton driver (`max_shared_mem` is
    ``cudaDevAttrMaxSharedMemoryPerBlockOptin``); falls back to a
    conservative capability table when the driver query is unavailable
    (e.g. CPU-only import).
    """
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(
            torch.cuda.current_device()
        )
        return int(props["max_shared_mem"])
    except Exception:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 9:
                return 232_448  # Hopper+: 227 KB opt-in
            if major == 8:
                return 101_376  # worst sm_8x (86/89); A100 has more
        return 98_304  # conservative pre-Ampere default


def _prune_configs(configs, named_args, **kwargs):
    """Reject configs that exceed the GPU's shared-memory budget.

    BLOCK_Q and BLOCK_D load fp16 tiles of size [BLOCK_*, d_pad] — the
    kernels index with ``tl.arange(0, d_pad)``, so the estimate must use
    the padded dimension, not the real one (d=513 pads to 1024 and
    nearly doubles the tile) — and the [BLOCK_Q, BLOCK_D] score tile is
    fp32.  Total:
        (BLOCK_Q*d_pad + BLOCK_D*d_pad)*2 + BLOCK_Q*BLOCK_D*4 bytes.
    """
    Lq = named_args.get("Lq", 32)
    d_pad = named_args.get("d_pad")
    if d_pad is None:
        d_pad = _next_pow2(max(named_args.get("d", 128), 16))
    smem_limit = _smem_budget()

    def _est(cfg):
        bq = cfg.kwargs["BLOCK_Q"]
        bd = cfg.kwargs["BLOCK_D"]
        return (bq * d_pad + bd * d_pad) * 2 + bq * bd * 4

    pruned = []
    for cfg in configs:
        if cfg.kwargs["BLOCK_Q"] > Lq * 2:
            continue
        if _est(cfg) > smem_limit:
            continue
        pruned.append(cfg)
    if pruned:
        return pruned
    # Nothing fits the estimate (large d_pad on a small-SMEM device).
    # Return the single smallest-footprint config — a deterministic
    # least-bad launch attempt — never arbitrary rejected configs.
    return [min(configs, key=_est)]
