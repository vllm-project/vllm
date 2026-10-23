# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32-exact router GEMM on bf16 matrix cores.

Three bf16 planes sum exactly to the fp32 weight; split-K sums in fixed order.
"""

import functools

import torch

from vllm.triton_utils import tl, triton

# (max tokens, BLOCK_M, BLOCK_N, BLOCK_K, splits, num_warps) for the Muse router
# shape (N=256, K=8192), tuned on MI350X (CDNA4, gfx950).
_CONFIGS_CDNA4 = (
    (16, 16, 32, 128, 32, 2),
    (64, 64, 32, 128, 32, 2),
    (128, 64, 32, 128, 16, 2),
    (256, 64, 64, 128, 16, 4),
    (1024, 128, 64, 128, 16, 8),
    (4096, 128, 128, 64, 8, 8),
    (None, 128, 128, 64, 4, 8),
)
# Tuned on MI300X (CDNA3, gfx942). Its 64 KiB of LDS rejects some of the CDNA4 tiles.
_CONFIGS_CDNA3 = (
    (16, 32, 32, 64, 32, 4),
    (128, 64, 64, 128, 32, 4),
    (256, 64, 64, 128, 16, 4),
    (512, 64, 64, 128, 8, 4),
    (1024, 128, 128, 64, 16, 8),
    (2048, 128, 128, 64, 8, 8),
    (None, 128, 128, 64, 4, 8),
)


@functools.cache
def _device_configs() -> tuple[tuple, ...]:
    from vllm.platforms.rocm import get_cdna_version

    return _CONFIGS_CDNA3 if get_cdna_version() == 3 else _CONFIGS_CDNA4


def _config(num_tokens: int) -> tuple[int, int, int, int, int]:
    for max_tokens, *cfg in _device_configs():
        if max_tokens is None or num_tokens <= max_tokens:
            return tuple(cfg)  # type: ignore[return-value]
    raise AssertionError("unreachable")


def supports_router_bf16x3(num_experts: int, hidden_size: int) -> bool:
    return all(
        num_experts % bn == 0 and hidden_size % (splits * bk) == 0
        for configs in (_CONFIGS_CDNA4, _CONFIGS_CDNA3)
        for _, _, bn, bk, splits, _ in configs
    )


def split_fp32_to_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """``[3, N, K]`` bf16 planes whose sum is ``weight`` (fp32) exactly."""
    assert weight.dtype == torch.float32
    hi = weight.to(torch.bfloat16)
    rest = weight - hi.float()
    mid = rest.to(torch.bfloat16)
    lo = (rest - mid.float()).to(torch.bfloat16)
    return torch.stack([hi, mid, lo]).contiguous()


@triton.jit
def _router_bf16x3_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        a = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        w_tile = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        for p in tl.static_range(3):
            acc = tl.dot(a, tl.load(w_tile + p * N * K), acc)
    tl.store(
        part_ptr + (pid_k * M + offs_m[:, None]) * N + offs_n[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _reduce_splits_kernel(
    part_ptr, out_ptr, MN, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(part_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLITS):
        acc += tl.load(part_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


def router_bf16x3_gemm(x: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """``x @ (planes[0] + planes[1] + planes[2]).T`` in fp32."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    m, k = x.shape
    n = planes.shape[1]
    assert planes.shape == (3, n, k)
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if m == 0:
        return out
    block_m, block_n, block_k, splits, num_warps = _config(m)
    part = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _router_bf16x3_kernel[(n // block_n, splits, triton.cdiv(m, block_m))](
        x,
        planes,
        part,
        m,
        N=n,
        K=k,
        K_PER_SPLIT=k // splits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    block = 1024
    _reduce_splits_kernel[(triton.cdiv(m * n, block),)](
        part, out, m * n, SPLITS=splits, BLOCK=block, num_warps=4
    )
    return out
