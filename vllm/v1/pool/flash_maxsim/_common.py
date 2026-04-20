# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for flash-maxsim kernels: padding, autotune configs,
and pruning."""

import torch

from vllm.triton_utils import triton


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _detect_gpu() -> str:
    """Detect GPU architecture family via compute capability."""
    if not torch.cuda.is_available():
        return "generic"
    major, minor = torch.cuda.get_device_capability()
    if major == 9:
        return "hopper"  # sm_90: H100, H200
    if major >= 10:
        return "blackwell"  # sm_100+: B200, RTX 5090
    if major == 8:
        if minor == 0:
            return "a100"  # sm_80: A100
        if minor >= 9:
            return "ada"  # sm_89: RTX 4090, L40S
        return "ampere"  # sm_86: RTX 3090, A10, A40
    return "generic"  # V100 (sm_70), T4 (sm_75), etc.


def _get_configs(gpu: str | None = None) -> list:
    """Hardware-aware autotune configs.

    Base set (4 small configs) covers every GPU; each family appends
    larger tiles / deeper pipelines where SMEM allows.
    """
    gpu = gpu or _detect_gpu()
    base = [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    ]
    if gpu == "hopper":  # H100: 228 KB SMEM, WGMMA, TMA, stages 3-4
        return base + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        ]
    if gpu == "blackwell":  # B200: 228 KB SMEM + 256 KB TMEM
        return base + [
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        ]
    if gpu == "a100":  # A100: 164 KB SMEM, stages 1-2 typical
        return base + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        ]
    if gpu == "ada":  # RTX 4090 / L40S: less BW than A100, same TC
        return base + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        ]
    if gpu == "ampere":  # RTX 3090 / A10 / A40: less SMEM than A100
        return base + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        ]
    # generic: V100, T4, etc.
    return base + [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
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
