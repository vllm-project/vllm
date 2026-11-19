# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 expert GEMMs for Hopper (Triton), in place of `trtllm_bmm.bmm`.

They read the same tile-padded expert rows and block scales (R8c4 / R128c4)
as the TRT-LLM-gen GEMMs, and row-major (unshuffled) expert weights. Both
operands are dequantized exactly (NVFP4 to fp16, MXFP8 to bf16: an e2m1 or
e4m3 value times its block scale needs at most 8 significant bits), so the
products match the block-scaled MMA; the fp32 row and expert scales apply to
the accumulator.
"""

import torch

from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets
from .trtllm_bmm import ExpertPlan


@triton.jit
def _dequant_e2m1(codes, scales):
    """fp16 values of the e2m1 codes in the low nibbles of `codes` times
    their e4m3 `scales` (same shape), exact: |products| lie in
    `[2^-10, 2688]` with at most 6 significant bits. With its 3 magnitude
    bits at fp16 bits 9-11, a code reads as an fp16 of its value times
    2^-14."""
    codes = codes.to(tl.uint16)
    bits = ((codes & 8) << 12) | ((codes & 7) << 9)
    values = bits.to(tl.float16, bitcast=True) * 16384.0
    return values * scales.to(tl.float16)


@triton.jit
def _dequant_e4m3(values, exponents):
    """bf16 e4m3 `values` times 2 ** (ue8m0 `exponents` - 127)."""
    exponents = exponents.to(tl.uint32)
    scales = tl.where(exponents == 0, 0x00400000, exponents << 23)
    scales = scales.to(tl.float32, bitcast=True)
    return (values.to(tl.float32) * scales).to(tl.bfloat16)


@triton.jit
def _expand_scales(scales, REPEAT: tl.constexpr):
    """`[R, G]` block scales -> `[R, G * REPEAT]`, each repeated in place."""
    rows: tl.constexpr = scales.shape[0]
    groups: tl.constexpr = scales.shape[1]
    scales = tl.broadcast_to(scales[:, :, None], (rows, groups, REPEAT))
    return tl.reshape(scales, (rows, groups * REPEAT))


@triton.jit
def _expert_gemm_kernel(
    Act,
    ActScale,
    RowScale,
    Weight,
    WeightScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    NVFP4: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """One `[BN, BT]` block of the transposed output: `BN` weight rows (the
    MMA's M, dequantized in registers) by `BT` rows of one token tile."""
    PACK: tl.constexpr = 2 if NVFP4 else 1
    GROUP: tl.constexpr = 16 if NVFP4 else 32
    T_BLOCKS: tl.constexpr = (TILE + BT - 1) // BT
    tl.extra.cuda.gdc_wait()
    pid_t = tl.program_id(1)
    tile = pid_t // T_BLOCKS
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + (pid_t % T_BLOCKS) * BT + tl.arange(0, BT)
    row_mask = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_mask = cols < N

    weight = Weight + expert * N * (K // PACK)
    weight_scale = WeightScale + expert * N * (K // GROUP)
    acc = tl.zeros((BN, BT), tl.float32)
    for k in range(0, K, BK):
        # NVFP4 bytes hold columns 2j (low nibble) and 2j + 1: the low and
        # high nibbles of both operands are multiplied separately.
        kp = k // PACK + tl.arange(0, BK // PACK)
        kg = k // GROUP + tl.arange(0, BK // GROUP)
        # Masked rows and columns are never stored.
        w = tl.load(
            weight + cols[:, None].to(tl.int64) * (K // PACK) + kp[None, :],
            col_mask[:, None],
        )
        w_scale = _expand_scales(
            tl.load(
                weight_scale + cols[:, None] * (K // GROUP) + kg[None, :],
                col_mask[:, None],
            ),
            GROUP // PACK,
        )
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * (K // PACK) + kp[None, :],
            row_mask[:, None],
        )
        a_scale = _expand_scales(
            tl.load(
                ActScale + _scale_offsets(rows[:, None], kg[None, :], K // GROUP, TILE),
                row_mask[:, None],
            ),
            GROUP // PACK,
        )
        if NVFP4:
            acc = tl.dot(
                _dequant_e2m1(w, w_scale),
                tl.trans(_dequant_e2m1(a, a_scale)),
                acc,
            )
            acc = tl.dot(
                _dequant_e2m1(w >> 4, w_scale),
                tl.trans(_dequant_e2m1(a >> 4, a_scale)),
                acc,
            )
        else:
            acc = tl.dot(
                _dequant_e4m3(w, w_scale), tl.trans(_dequant_e4m3(a, a_scale)), acc
            )
    acc = acc * tl.load(ExpertScale + expert)
    if NVFP4:
        acc = acc * tl.load(RowScale + rows, row_mask, other=0)[None, :]
    tl.store(
        Out + rows[None, :].to(tl.int64) * N + cols[:, None],
        acc.to(tl.bfloat16),
        row_mask[None, :] & col_mask[:, None],
    )


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
    projection: int,
) -> torch.Tensor:
    """`trtllm_bmm.bmm` (bf16 `[plan.rows, N]`) on row-major weights."""
    del projection
    nvfp4 = weight.dtype == torch.uint8
    num_experts, n = weight.shape[:2]
    k = weight.shape[2] * (2 if nvfp4 else 1)
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    # (weight rows, token rows, warps, stages) by token tile, from H200 sweeps.
    if plan.tile <= 16:
        block_n, block_t, num_warps, num_stages = 128, 16, 4, 3
    elif plan.tile <= 64:
        block_n, block_t, num_warps, num_stages = 128, 64, 4, 3
    else:
        block_n, block_t, num_warps, num_stages = 256, 128, 8, 2 if nvfp4 else 3
    block_k = 128
    assert k % block_k == 0
    grid = (
        triton.cdiv(n, block_n),
        plan.max_ctas * triton.cdiv(plan.tile, block_t),
    )
    _expert_gemm_kernel[grid](
        act,
        act_scale,
        row_scale,
        weight,
        weight_scale,
        ones,
        out,
        tiles,
        MAX_CTAS=plan.max_ctas,
        N=n,
        K=k,
        TILE=plan.tile,
        NVFP4=nvfp4,
        BN=block_n,
        BT=block_t,
        BK=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=True,
    )
    return out
