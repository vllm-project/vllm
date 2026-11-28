# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120/121 expert GEMMs with block-scaled NVFP4 and MXFP8 operands.

Weights remain row-major, while activations retain Inkling2's tile-padded
R8c4/R128c4 layout and its four-or-six quantization and outer row scales.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets
from .trtllm_bmm import ExpertPlan


@triton.jit
def _expert_gemm(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
            valid[:, None] & (packed[None, :] < K // 2),
            other=0,
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
            valid[:, None] & (groups[None, :] < K // 16),
            other=0.0,
        )
        w = tl.load(
            Weight + (expert * N + cols[None, :]) * (K // 2) + packed[:, None],
            (cols[None, :] < N) & (packed[:, None] < K // 2),
            other=0,
        )
        w_scale = tl.load(
            WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :],
            (cols[:, None] < N) & (groups[None, :] < K // 16),
            other=0.0,
        )
        acc = tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1", acc)
    acc *= tl.load(ExpertScale + expert)
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    tl.store(
        Out + rows[:, None].to(tl.int64) * N + cols[None, :],
        acc,
        valid[:, None] & (cols[None, :] < N),
    )


@triton.jit
def _dequant_e4m3(values, exponents):
    exponents = exponents.to(tl.uint32)
    scales = tl.where(exponents == 0, 0x00400000, exponents << 23)
    scales = scales.to(tl.float32, bitcast=True)
    return (values.to(tl.float32) * scales).to(tl.bfloat16)


@triton.jit
def _expand_scales(scales):
    rows: tl.constexpr = scales.shape[0]
    groups: tl.constexpr = scales.shape[1]
    scales = tl.broadcast_to(scales[:, :, None], (rows, groups, 32))
    return tl.reshape(scales, (rows, groups * 32))


@triton.jit
def _mxfp8_gemm(
    Act,
    ActScale,
    Weight,
    WeightScale,
    ExpertScale,
    Out,
    Tiles,
    MAX_CTAS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NATIVE_MXFP8: tl.constexpr,
):
    T_BLOCKS: tl.constexpr = tl.cdiv(TILE, BT)
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
    weight = Weight + expert * N * K
    weight_scale = WeightScale + expert * N * (K // 32)
    acc = tl.zeros((BN, BT), tl.float32)
    for k in range(0, K, BK):
        kp = k + tl.arange(0, BK)
        kg = k // 32 + tl.arange(0, BK // 32)
        w = tl.load(
            weight + cols[:, None].to(tl.int64) * K + kp[None, :],
            col_mask[:, None],
        )
        w_scale = tl.load(
            weight_scale + cols[:, None] * (K // 32) + kg[None, :],
            col_mask[:, None],
        )
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * K + kp[None, :],
            row_mask[:, None],
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], kg[None, :], K // 32, TILE),
            row_mask[:, None],
        )
        if NATIVE_MXFP8:
            acc = tl.dot_scaled(w, w_scale, "e4m3", tl.trans(a), a_scale, "e4m3", acc)
        else:
            acc = tl.dot(
                _dequant_e4m3(w, _expand_scales(w_scale)),
                tl.trans(_dequant_e4m3(a, _expand_scales(a_scale))),
                acc,
            )
    acc = acc * tl.load(ExpertScale + expert)
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
    """Multiply quantized expert rows, returning bf16 `[plan.rows, N]`."""
    del projection
    if weight.dtype == torch.float8_e4m3fn:
        n, k = weight.shape[1:]
        assert k % 128 == 0
        native_mxfp8 = (
            plan.tile == 128
            and n == 2304
            and k in (1152, 2304)
            and current_platform.is_device_capability_family(120)
        )
        if native_mxfp8:
            block_n, block_t, num_warps = 128, 64, 4
        elif plan.tile <= 16:
            block_n, block_t, num_warps = 128, 16, 4
        elif plan.tile <= 64:
            block_n, block_t, num_warps = 128, 64, 4
        else:
            block_n, block_t, num_warps = 256, 128, 8
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
        _mxfp8_gemm[
            (triton.cdiv(n, block_n), plan.max_ctas * triton.cdiv(plan.tile, block_t))
        ](
            act,
            act_scale,
            weight,
            weight_scale,
            ones,
            out,
            tiles,
            MAX_CTAS=plan.max_ctas,
            N=n,
            K=k,
            TILE=plan.tile,
            BN=block_n,
            BT=block_t,
            BK=128,
            NATIVE_MXFP8=native_mxfp8,
            num_warps=num_warps,
            num_stages=2 if native_mxfp8 else 3,
            launch_pdl=True,
        )
        return out
    assert weight.dtype == torch.uint8
    n, k = weight.shape[1], weight.shape[2] * 2
    block_m, block_n, num_warps = 32, 64, 4
    if plan.tile >= 128 and n == 2304 and k in (1152, 2304):
        # Mini's prefill tiles reuse each weight load across more routed rows.
        block_m = 128
        block_n = 128 if plan.tile >= 256 else 64
        num_warps = 8
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    _expert_gemm[
        (plan.max_ctas, triton.cdiv(plan.tile, block_m), triton.cdiv(n, block_n))
    ](
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
        BM=block_m,
        BN=block_n,
        BK=512,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
