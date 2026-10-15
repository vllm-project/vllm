# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton BF16x3 router GEMM for Hopper (SM90).

Computes the fp32 router logits ``x @ w.T`` for bf16 activations ``x`` with
shape ``[M, K]`` and fp32 router weights ``w`` with shape ``[E, K]``. Without
it, SM90 runs batches above the fp32 GEMV's 32 tokens as an fp32 SGEMM on CUDA
cores.

Each fp32 weight tile is split into three bf16 terms whose sum reproduces it
exactly, so the GEMM runs on bf16 tensor cores and every product with a bf16
activation is exact in fp32. All of the error comes from accumulation: the
tensor-core partial of each K-tile is added into an fp32 register accumulator
instead of being chained across K-tiles. Chaining even two 128-wide K-tiles in
the tensor core is already less accurate than the fp32 SGEMM this replaces.

Small batches split K across CTAs and sum the fp32 partials in a fixed order,
so results do not depend on scheduling.
"""

from functools import cache
from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = [
    "bf16x3_router_gemm",
    "warmup_bf16x3_router_gemm",
]


class _Config(NamedTuple):
    block_e: int
    block_t: int
    block_k: int
    num_warps: int
    num_stages: int
    split_k: int


@triton.jit
def _split_bf16x3(w):
    # Each term keeps the top 16 bits of what is left. The remainders are
    # exact, and three 8-bit significands cover fp32's 24, so the terms sum
    # back to w. Masking is cheaper than round-to-nearest conversions here.
    mask = 0xFFFF0000
    bits = w.to(tl.uint32, bitcast=True)
    rem = w - (bits & mask).to(tl.float32, bitcast=True)
    rem_bits = rem.to(tl.uint32, bitcast=True)
    lo = rem - (rem_bits & mask).to(tl.float32, bitcast=True)
    lo_bits = lo.to(tl.uint32, bitcast=True)
    hi = (bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    mid = (rem_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    lo = (lo_bits >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return hi, mid, lo


@triton.jit(do_not_specialize=["M", "stride_os"])
def _bf16x3_router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    E,
    K,
    stride_xm,
    stride_we,
    stride_os,
    stride_om,
    tiles_per_split,
    BLOCK_E: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_e = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    mask_e = offs_e < E
    mask_t = offs_t < M

    tile_start = pid_s * tiles_per_split
    tile_end = tl.minimum(tile_start + tiles_per_split, K // BLOCK_K)
    k_start = tile_start * BLOCK_K
    x_ptrs = (
        x_ptr + offs_t[:, None].to(tl.int64) * stride_xm + (k_start + offs_k)[None, :]
    )
    w_ptrs = (
        w_ptr + offs_e[None, :].to(tl.int64) * stride_we + (k_start + offs_k)[:, None]
    )

    acc = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
    for _ in range(tile_start, tile_end):
        x = tl.load(x_ptrs, mask=mask_t[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_e[None, :], other=0.0)
        w_hi, w_mid, w_lo = _split_bf16x3(w)
        # Fresh tensor-core accumulator per K-tile, smallest term first.
        part = tl.dot(x, w_lo)
        part = tl.dot(x, w_mid, part)
        part = tl.dot(x, w_hi, part)
        acc += part
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out_ptrs = (
        out_ptr
        + pid_s * stride_os
        + offs_t[:, None].to(tl.int64) * stride_om
        + offs_e[None, :]
    )
    tl.store(out_ptrs, acc, mask=mask_t[:, None] & mask_e[None, :])


# The configs below come from cold-L2 CUDA-graph sweeps on H200. All of them
# keep the MAE against an fp64 reference below that of the fp32 SGEMM.

# (BLOCK_E, BLOCK_T, BLOCK_K, num_warps, num_stages) per token bucket
# next_power_of_2(M); larger buckets reuse the 2048 entry.
_DEFAULT_TILES = {
    64: (32, 16, 128, 4, 3),
    128: (64, 32, 64, 4, 3),
    256: (32, 128, 128, 8, 3),
    512: (64, 128, 128, 8, 3),
    1024: (64, 128, 64, 8, 3),
    2048: (64, 256, 64, 8, 3),
}

# Router shapes (K, E) and buckets where the defaults were >10% slower.
_CONFIG_OVERRIDES = {
    (4096, 128): {  # GLM-4.5-Air
        128: _Config(32, 16, 128, 4, 3, 8),
        512: _Config(32, 128, 128, 8, 3, 8),
    },
    (4096, 192): {  # Hunyuan-V3
        64: _Config(32, 16, 128, 4, 3, 8),
        1024: _Config(64, 256, 64, 8, 3, 8),
        2048: _Config(64, 256, 64, 8, 3, 4),
    },
    (4096, 256): {  # Ling-flash-2.0
        1024: _Config(64, 256, 64, 8, 3, 8),
    },
    (4096, 288): {  # Step-3.5-Flash
        256: _Config(64, 128, 128, 8, 3, 8),
        512: _Config(64, 256, 64, 8, 3, 8),
    },
    (4096, 352): {  # Step-5
        256: _Config(64, 128, 128, 8, 3, 8),
        512: _Config(64, 256, 64, 8, 3, 8),
    },
    (8192, 64): {  # ERNIE-4.5-300B-A47B
        128: _Config(32, 16, 128, 4, 4, 8),
        256: _Config(32, 64, 128, 4, 3, 16),
        512: _Config(32, 128, 128, 8, 3, 16),
    },
}


def _pick_config(m: int, k: int, e: int, num_sms: int) -> _Config:
    bucket = max(triton.next_power_of_2(m), 64)
    cfg = _CONFIG_OVERRIDES.get((k, e), {}).get(bucket)
    if cfg is None:
        block_e, block_t, block_k, num_warps, num_stages = _DEFAULT_TILES[
            min(bucket, 2048)
        ]
        if k % block_k:
            block_k = 64
        # Split K for the fewest waves per unit of work, preferring fewer splits.
        base_ctas = triton.cdiv(e, block_e) * triton.cdiv(m, block_t)
        splits = [
            s
            for s in (1, 2, 4, 8, 16)
            if s == 1 or (k // block_k >= 2 * s and base_ctas * s <= 4 * num_sms)
        ]
        split_k = min(
            splits, key=lambda s: (triton.cdiv(base_ctas * s, num_sms) / s, s)
        )
        cfg = _Config(block_e, block_t, block_k, num_warps, num_stages, split_k)
    elif k % cfg.block_k:
        cfg = cfg._replace(block_k=64)
    return cfg


# Cached per token count: outside CUDA graphs, host time per call matters as
# much as the kernel for mid-size batches.
@cache
def _launch_plan(
    m: int, k: int, e: int, num_sms: int
) -> tuple[tuple[int, int, int], int, dict[str, int]]:
    cfg = _pick_config(m, k, e, num_sms)
    k_tiles = k // cfg.block_k
    tiles_per_split = triton.cdiv(k_tiles, cfg.split_k)
    split_k = triton.cdiv(k_tiles, tiles_per_split)
    grid = (triton.cdiv(e, cfg.block_e), triton.cdiv(m, cfg.block_t), split_k)
    meta = {
        "BLOCK_E": cfg.block_e,
        "BLOCK_T": cfg.block_t,
        "BLOCK_K": cfg.block_k,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    return grid, tiles_per_split, meta


def _bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    e = weight.shape[0]
    if m == 0:
        return x.new_empty((0, e), dtype=torch.float32)
    if x.stride(1) != 1:
        x = x.contiguous()
    grid, tiles_per_split, meta = _launch_plan(
        m, k, e, num_compute_units(x.device.index)
    )
    split_k = grid[2]
    out = x.new_empty((split_k, m, e), dtype=torch.float32)
    _bf16x3_router_gemm_kernel[grid](
        x,
        weight,
        out,
        m,
        e,
        k,
        x.stride(0),
        weight.stride(0),
        m * e,
        e,
        tiles_per_split,
        **meta,
    )
    # Summing the split partials in a fixed order keeps results deterministic.
    return out[0] if split_k == 1 else out.sum(0)


def _bf16x3_router_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="sm90_bf16x3_router_gemm",
    op_func=_bf16x3_router_gemm,
    fake_impl=_bf16x3_router_gemm_fake,
)


def bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``x @ weight.T`` in fp32 using the SM90 BF16x3 router GEMM."""
    return torch.ops.vllm.sm90_bf16x3_router_gemm(x, weight)


def warmup_bf16x3_router_gemm(
    K: int,
    E: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[_Config, ...]:
    """Compile every config reachable for token counts in the given range."""
    device = torch.accelerator.current_device_index()
    num_sms = num_compute_units(device)
    configs: dict[_Config, int] = {}
    for m in range(min_num_tokens, max_num_tokens + 1):
        configs.setdefault(_pick_config(m, K, E, num_sms), m)
    weight = torch.zeros((E, K), dtype=torch.float32, device=device)
    for m in configs.values():
        x = torch.zeros((m, K), dtype=torch.bfloat16, device=device)
        _bf16x3_router_gemm(x, weight)
    return tuple(configs)
