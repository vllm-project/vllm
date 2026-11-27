# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120/121 expert GEMMs with native NVFP4 MMA and dequantized MXFP8.

Weights remain row-major, while activations retain Inkling2's tile-padded
R8c4/R128c4 layout and its four-or-six quantization and outer row scales.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .block_scales import _scale_offsets
from .trtllm_bmm import ExpertPlan

# Cap early dependent launch to decode/verify batches; prefill saturates SMs.
LAUNCH_DEPENDENTS_MAX_ROWS = 2048


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
    launch_pdl: tl.constexpr,
    PREFETCH_TILES: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
    MASK_N: tl.constexpr,
    MASK_K: tl.constexpr,
    EVICT_W: tl.constexpr,
):
    tile = tl.program_id(0)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    if launch_pdl and PREFETCH_TILES:
        # w2: Tiles and ExpertScale were written before w13, so load before wait.
        num_tiles = tl.load(Tiles + 2 * MAX_CTAS)
        valid_tile = tile < num_tiles
        expert = tl.load(Tiles + tile, mask=valid_tile, other=0).to(tl.int64)
        row_end = tl.load(Tiles + MAX_CTAS + tile, mask=valid_tile, other=0)
        expert_scale = tl.load(ExpertScale + expert, mask=valid_tile, other=0.0)
        tl.extra.cuda.gdc_wait()
        if LAUNCH_DEPENDENTS:
            tl.extra.cuda.gdc_launch_dependents()
        if not valid_tile:
            return
    else:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
            if LAUNCH_DEPENDENTS:
                tl.extra.cuda.gdc_launch_dependents()
        if tile >= tl.load(Tiles + 2 * MAX_CTAS):
            return
        expert = tl.load(Tiles + tile).to(tl.int64)
        row_end = tl.load(Tiles + MAX_CTAS + tile)
        expert_scale = tl.load(ExpertScale + expert)

    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    acc = tl.zeros((BM, BN), tl.float32)
    w_evict: tl.constexpr = "evict_first" if EVICT_W else ""
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a_mask = (
            valid[:, None] & (packed[None, :] < K // 2) if MASK_K else valid[:, None]
        )
        as_mask = (
            valid[:, None] & (groups[None, :] < K // 16) if MASK_K else valid[:, None]
        )
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
            a_mask,
            other=0,
        )
        a_scale = tl.load(
            ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
            as_mask,
            other=0.0,
        )
        w_ptr = Weight + (expert * N + cols[None, :]) * (K // 2) + packed[:, None]
        ws_ptr = (
            WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :]
        )
        if MASK_N or MASK_K:
            w_mask = (cols[None, :] < N) if MASK_N else True
            ws_mask = (cols[:, None] < N) if MASK_N else True
            if MASK_K:
                w_mask = w_mask & (packed[:, None] < K // 2)
                ws_mask = ws_mask & (groups[None, :] < K // 16)
            w = tl.load(w_ptr, mask=w_mask, other=0, eviction_policy=w_evict)
            w_scale = tl.load(ws_ptr, mask=ws_mask, other=0.0, eviction_policy=w_evict)
        else:
            w = tl.load(w_ptr, eviction_policy=w_evict)
            w_scale = tl.load(ws_ptr, eviction_policy=w_evict)
        acc = tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1", acc)
    acc *= expert_scale
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    out_mask = valid[:, None] & (cols[None, :] < N) if MASK_N else valid[:, None]
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc, out_mask)


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
    launch_pdl: tl.constexpr,
    PREFETCH_TILES: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    T_BLOCKS: tl.constexpr = tl.cdiv(TILE, BT)
    pid_t = tl.program_id(1)
    tile = pid_t // T_BLOCKS
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    if launch_pdl and PREFETCH_TILES:
        num_tiles = tl.load(Tiles + 2 * MAX_CTAS)
        valid_tile = tile < num_tiles
        expert = tl.load(Tiles + tile, mask=valid_tile, other=0).to(tl.int64)
        row_end = tl.load(Tiles + MAX_CTAS + tile, mask=valid_tile, other=0)
        expert_scale = tl.load(ExpertScale + expert, mask=valid_tile, other=0.0)
        tl.extra.cuda.gdc_wait()
        if LAUNCH_DEPENDENTS:
            tl.extra.cuda.gdc_launch_dependents()
        if not valid_tile:
            return
    else:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
            if LAUNCH_DEPENDENTS:
                tl.extra.cuda.gdc_launch_dependents()
        if tile >= tl.load(Tiles + 2 * MAX_CTAS):
            return
        expert = tl.load(Tiles + tile).to(tl.int64)
        row_end = tl.load(Tiles + MAX_CTAS + tile)
        expert_scale = tl.load(ExpertScale + expert)
    rows = tile * TILE + (pid_t % T_BLOCKS) * BT + tl.arange(0, BT)
    row_mask = rows < tl.minimum(row_end, (tile + 1) * TILE)
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
        w_scale = _expand_scales(
            tl.load(
                weight_scale + cols[:, None] * (K // 32) + kg[None, :],
                col_mask[:, None],
            )
        )
        a = tl.load(
            Act + rows[:, None].to(tl.int64) * K + kp[None, :],
            row_mask[:, None],
        )
        a_scale = _expand_scales(
            tl.load(
                ActScale + _scale_offsets(rows[:, None], kg[None, :], K // 32, TILE),
                row_mask[:, None],
            )
        )
        acc = tl.dot(
            _dequant_e4m3(w, w_scale), tl.trans(_dequant_e4m3(a, a_scale)), acc
        )
    acc = acc * expert_scale
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
    use_pdl = current_platform.is_arch_support_pdl()
    if weight.dtype == torch.float8_e4m3fn:
        n, k = weight.shape[1:]
        assert k % 128 == 0
        if plan.tile <= 16:
            block_n = 64 if n == 2304 else 128
            block_t, num_warps = 16, 4
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
            PREFETCH_TILES=(projection == 1),
            LAUNCH_DEPENDENTS=(plan.rows <= LAUNCH_DEPENDENTS_MAX_ROWS),
            num_warps=num_warps,
            num_stages=3,
            launch_pdl=use_pdl,
        )
        return out
    assert weight.dtype == torch.uint8
    n, k = weight.shape[1], weight.shape[2] * 2
    block_m, block_n, block_k, num_warps, num_stages = 32, 64, 512, 4, 2
    if plan.tile >= 128 and n == 2304 and k in (1152, 2304):
        # Mini's prefill tiles reuse each weight load across more routed rows.
        block_m = 128
        block_n = 128 if plan.tile >= 256 else 64
        num_warps = 8
    elif plan.tile < 128 and n == 2304 and k in (1152, 2304):
        if plan.rows <= 8:
            # Single-wave N-tiling at bs=1 decode (144 vs 288 CTAs).
            block_m, block_n, block_k, num_warps, num_stages = 32, 128, 256, 4, 3
        elif k == 2304 and plan.tile <= 16 and plan.rows > 64:
            # Multi-token decode w13 benefits from BM=16 with BK=512.
            block_m, block_k, num_stages = 16, 512, 2
        else:
            # Smaller BK removes K=2304 tail masking and enables deeper pipelining.
            block_k = 256
            num_stages = 4 if k == 2304 else 3
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
        BK=block_k,
        PREFETCH_TILES=(projection == 1),
        LAUNCH_DEPENDENTS=(plan.rows <= LAUNCH_DEPENDENTS_MAX_ROWS),
        MASK_N=(n % block_n != 0),
        MASK_K=(k % block_k != 0),
        EVICT_W=(plan.tile <= 16),
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=use_pdl,
    )
    return out
