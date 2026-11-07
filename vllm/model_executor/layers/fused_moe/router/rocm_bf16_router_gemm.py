# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K BF16 router GEMM for gfx950: ``logits = x @ weight.T`` in fp32.

For a bf16 ``[num_experts, hidden]`` gate whose logits are consumed in fp32,
hipBLASLt serves decode batch sizes with a 32x16 macro tile that walks all of
K in one workgroup and needs ~96 KB of LDS, so it loses its CUs to a shared
expert running on a second stream. This kernel splits K so a small grid still
fills the CUs with ~32 KB of LDS a workgroup, and sums the splits in the same
launch with the arrival counters of the MXFP8 block32 GEMM: the last program
of each output tile adds the partials in split order, so the result does not
depend on scheduling. Every product is a bf16 x bf16 MFMA accumulated in fp32;
only the K summation order differs from hipBLASLt.

Only (N, K) shapes with tuned tiers are served, up to the last tier's M, and
only on gfx950, where the tiers were tuned.
"""

from functools import cache
from typing import NamedTuple

import torch

from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (
    _SPLIT_COUNTERS,
    _XCD_STRIDE,
    _k_partition,
    _split_counters,
    _split_tile,
    _sum_splits,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["M"])
def _bf16_router_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    SPLITS: tl.constexpr,
):
    if SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, SPLITS)
        if pid_m * BLOCK_M >= M:
            return
    else:
        pid_n, pid_m = tl.program_id(0), tl.program_id(1)
        tile = 0
        pid_k = 0
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * K_PER_SPLIT
    offs_k = k0 + tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        k_ok = (offs_k + kk) < K
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0.0)
        acc = tl.dot(x, w.T, acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLITS > 1:
        _sum_splits(
            acc,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            SPLITS,
        )
    else:
        tl.store(out_ptrs, acc, mask=out_mask)


class _Config(NamedTuple):
    block_m: int
    block_n: int
    block_k: int
    splits: int
    num_warps: int
    num_stages: int
    waves_per_eu: int
    matrix_instr_nonkdim: int = 16


# (N, K) -> [(max M, config), ...], tuned on MI350X under HIP graphs with the
# weight cold in the LLC and the TP1 shared expert's MXFP8 GEMMs on a second
# stream, as in a DP-attention decode layer.
_TUNED: dict[tuple[int, int], list[tuple[int, _Config]]] = {
    # DeepSeek-V4.1-Flash gate: 384 experts, hidden 5120.
    (384, 5120): [
        (64, _Config(64, 16, 256, 8, 4, 3, 0)),
        (128, _Config(32, 32, 128, 8, 4, 3, 0)),
        (256, _Config(32, 32, 256, 8, 4, 1, 0)),
    ],
}


def _config(M: int, N: int, K: int) -> _Config | None:
    for max_m, cfg in _TUNED.get((N, K), ()):
        if max_m >= M:
            return cfg
    return None


def _launch(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, cfg: _Config
) -> None:
    M, K = x.shape
    N = weight.shape[0]
    grid_m, grid_n = triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n)
    tiles = grid_m * grid_n
    k_per_split, splits = _k_partition(K, cfg.splits, cfg.block_k)
    counters = None
    if splits > 1 and tiles <= _SPLIT_COUNTERS:
        counters = _split_counters(out.device)
    if splits > 1 and counters is None:
        # No counters inside an un-warmed capture: one split, still correct.
        k_per_split, splits = _k_partition(K, 1, cfg.block_k)
    if splits > 1:
        slots = torch.empty(
            tiles * splits * cfg.block_m * cfg.block_n,
            dtype=torch.float32,
            device=out.device,
        )
        grid: tuple[int, ...] = (
            triton.cdiv(tiles, _XCD_STRIDE) * _XCD_STRIDE * splits,
        )
    else:
        slots = counters = out
        grid = (grid_n, grid_m)
    _bf16_router_gemm_kernel[grid](
        x,
        weight,
        out,
        slots,
        counters,
        M,
        N,
        K,
        x.stride(0),
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        K_PER_SPLIT=k_per_split,
        SPLITS=splits,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        waves_per_eu=cfg.waves_per_eu,
        matrix_instr_nonkdim=cfg.matrix_instr_nonkdim,
    )


@cache
def platform_supported() -> bool:
    """Whether the tiers apply to this device: gfx950 only, as tuned."""
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def rocm_bf16_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """``x @ weight.T`` in fp32 for bf16 ``x`` [M, K] and ``weight`` [N, K].

    Returns None off gfx950, when (N, K) has no tier covering M, or when the
    layout does not fit, so the caller keeps its own GEMM.
    """
    M = x.shape[0]
    if (
        M == 0
        or not platform_supported()
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.dim() != 2
        or weight.dim() != 2
        or x.stride(1) != 1
        or not weight.is_contiguous()
        or x.shape[1] != weight.shape[1]
        or x.device != weight.device
    ):
        return None
    cfg = _config(M, weight.shape[0], weight.shape[1])
    if cfg is None:
        return None
    out = torch.empty((M, weight.shape[0]), dtype=torch.float32, device=x.device)
    _launch(x, weight, out, cfg)
    return out
