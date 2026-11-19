# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 MoE GEMM kernel for gfx942 using tl.dot_scaled with fp8 scales.

W layout: [E, K//2, N] packed e2m1 uint8 (K-major)
W scale:  [E, N, K//16] fp8_e4m3fn (NVFP4 per-16 scales)
X scale:  [M, K//16] fp8_e4m3fn (all 1.0 for non-microscaled X)
"""

import torch
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton._triton_kernels.moe.quant_moe import (
    _compute_static_fp8_quant,
)
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import should_upcast_indices
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils.device_info import get_num_xcds

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    return group * pids_per_group + min(group, extra_pid_groups) + local_pid


@triton.jit
def _moe_gemm_nvfp4_dot_scaled(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WScale,
    stride_ws_e,
    stride_ws_n,
    stride_ws_k,
    GlobalScale,
    X_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,
    Gammas,
    N,
    K,
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    grid_m,
    grid_n,
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    """NVFP4 MoE GEMM using tl.dot_scaled with fp8_e4m3fn scales."""
    NVFP4_SCALE_FACTOR: tl.constexpr = 16
    NVFP4_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // NVFP4_SCALE_FACTOR
    PACKED_BLOCK_K: tl.constexpr = BLOCK_K // 2

    tl.static_assert(BLOCK_K % NVFP4_SCALE_FACTOR == 0)

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    tl.assume(stride_ws_e >= 0)
    tl.assume(stride_ws_n >= 0)
    tl.assume(stride_ws_k >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0  # type: ignore[no-redef]

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = _xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)

    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
        # Advance X_static_scale too: offs_x_m is expert-local, so without
        # this offset, experts with nonzero start_m read wrong per-row scales.
        if X_static_scale is not None:
            X_static_scale += start_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    # FNUZ bytes are decoded as OCP e4m3fn (2x); the caller halves the
    # global scale to compensate.
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
    offs_w_k = PACKED_BLOCK_K * pid_k + tl.arange(0, PACKED_BLOCK_K)
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    offs_ws_k = NVFP4_SCALE_BLOCK_K * pid_k + tl.arange(0, NVFP4_SCALE_BLOCK_K)
    WScale += expt_id * stride_ws_e
    WScalePtrs = WScale + (
        offs_w_n.to(index_type)[:, None] * stride_ws_n
        + offs_ws_k.to(index_type)[None, :] * stride_ws_k
    )

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        ws = tl.load(WScalePtrs, cache_modifier=W_CACHE_MODIFIER)

        # Identity X scales: OCP fp8 1.0 = 0x38
        xs = tl.full((BLOCK_M, NVFP4_SCALE_BLOCK_K), 0x38, dtype=tl.uint8)
        xs = tl.cast(xs, tl.float8e4nv, bitcast=True)

        acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e2m1", acc=acc, fast_math=True)

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K * SPLIT_K) * stride_w_k
        WScalePtrs += (NVFP4_SCALE_BLOCK_K * SPLIT_K) * stride_ws_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < (MASK_K_LIMIT // 2)
        mask_ws_k = offs_ws_k * NVFP4_SCALE_FACTOR < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )
        ws = tl.load(
            WScalePtrs,
            mask=mask_ws_k[None, :],
            other=0.0,
            cache_modifier=W_CACHE_MODIFIER,
        )

        xs = tl.full((BLOCK_M, NVFP4_SCALE_BLOCK_K), 0x38, dtype=tl.uint8)
        xs = tl.cast(xs, tl.float8e4nv, bitcast=True)

        acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e2m1", acc=acc, fast_math=True)

    global_scale = tl.load(GlobalScale + expt_id).to(tl.float32)
    acc *= global_scale

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M

    if X_static_scale is not None:
        x_scales_row = tl.load(X_static_scale + offs_x_m, mask=mask_m, other=1.0)
        acc *= x_scales_row[:, None]

    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]

    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
        tl.static_assert(out.shape[1] == OUT_BLOCK_N)
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(ACTIVATION_REDUCTION_N == 1)
        out = acc

    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))

    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)


@triton.jit
def _e2m1_to_f16_scaled(nib, s16):
    # e2m1 nibble -> fp16 equal to e2m1 * 2^-14 (exact, subnormal codes too), times
    # the fp16 block scale. Every product is exactly representable in fp16.
    bits = ((nib & 0x8) << 12) | ((nib & 0x7) << 9)
    return bits.to(tl.float16, bitcast=True) * s16


@triton.jit
def _moe_gemm_nvfp4_dequant_f16(
    Y,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WScale,
    stride_ws_e,
    stride_ws_n,
    stride_ws_k,
    GlobalScale,
    X_static_scale,
    Gammas,
    N,
    K,
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    grid_m,
    grid_n,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    X_FNUZ: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    """NVFP4 MoE GEMM without tl.dot_scaled: exact in-register NVFP4 -> fp16
    dequant (bit placement + one fp16 multiply by the block scale), fp16 MFMA
    with fp32 accumulation. Bitwise-equal products to the dot_scaled path but
    ~2x fewer VALU ops per weight and no LDS byte-shuffle (see WaveScope ATT).
    Activations are fp8 e4m3fnuz decoded exactly (no FNUZ/OCP 2x), so the
    PR's 0.5-corrected global scale is multiplied back by 2 here."""
    HALF_K: tl.constexpr = BLOCK_K // 2
    SCALE_K: tl.constexpr = BLOCK_K // 16
    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0  # type: ignore[no-redef]
    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n
    if padding_m > 0 and pid >= total_actual_tiles:
        return
    pid_mn = pid
    if XCD_SWIZZLE != 1:
        pid_mn = _xcd_swizzle(pid_mn, total_actual_tiles, XCD_SWIZZLE)
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n = pid_n.to(index_type)

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
        if X_static_scale is not None:
            X_static_scale += start_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT

    offs_k = tl.arange(0, BLOCK_K)
    offs_hk = tl.arange(0, HALF_K)
    offs_sk = tl.arange(0, SCALE_K)
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_k.to(index_type)[None, :] * stride_x_k
    )
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
    W += expt_id * stride_w_e
    WPtrs = (
        W
        + offs_w_n.to(index_type)[:, None] * stride_w_n
        + offs_hk.to(index_type)[None, :] * stride_w_k
    )
    WScale += expt_id * stride_ws_e
    WSPtrs = (
        WScale
        + offs_w_n.to(index_type)[:, None] * stride_ws_n
        + offs_sk.to(index_type)[None, :] * stride_ws_k
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            x = tl.load(XPtrs)
            w = tl.load(WPtrs)
            ws = tl.load(WSPtrs)
        else:
            krem = K - ki * BLOCK_K
            x = tl.load(XPtrs, mask=(offs_k < krem)[None, :], other=0)
            w = tl.load(WPtrs, mask=(2 * offs_hk < krem)[None, :], other=0)
            ws = tl.load(WSPtrs, mask=(offs_sk * 16 < krem)[None, :], other=0)
        # x: e4m3fnuz -> fp16 * 2^-7 ; ws: e4m3fn -> fp16 * 2^-8 (scaled back below)
        if X_FNUZ:  # hardware cvt on gfx942
            xf = x.to(tl.float8e4b8, bitcast=True).to(tl.float16)
        else:
            xf = x.to(tl.float8e4nv, bitcast=True).to(tl.float16)
        # e4m3fn scale bits read as e4m3fnuz decode to exactly value / 2 via the
        # gfx942 hardware cvt (incl. subnormals); x2 restores it exactly in fp16.
        # NVFP4 block scales are non-negative; & 0x7F maps -0 (fnuz NaN) to 0.
        s16 = (ws & 0x7F).to(tl.float8e4b8, bitcast=True).to(tl.float16) * 2.0
        s16 = tl.reshape(
            tl.broadcast_to(s16[:, :, None], (BLOCK_N, SCALE_K, 8)), (BLOCK_N, HALF_K)
        )
        wi = w.to(tl.int16)
        w_lo = _e2m1_to_f16_scaled(wi & 0xF, s16)  # k = 2j
        w_hi = _e2m1_to_f16_scaled((wi >> 4) & 0xF, s16)  # k = 2j + 1
        wf = tl.reshape(tl.join(w_lo, w_hi), (BLOCK_N, BLOCK_K))
        acc = tl.dot(xf, tl.trans(wf), acc)
        XPtrs += BLOCK_K * stride_x_k
        WPtrs += HALF_K * stride_w_k
        WSPtrs += SCALE_K * stride_ws_k

    # 2^14 (e2m1 placement) * 2 (undo PR's 0.5 correction, see docstring)
    global_scale = tl.load(GlobalScale + expt_id).to(tl.float32) * 32768.0
    acc *= global_scale
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    if X_static_scale is not None:
        x_scales_row = tl.load(X_static_scale + offs_x_m, mask=mask_m, other=1.0)
        acc *= x_scales_row[:, None]
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        acc *= gammas[:, None]
    Y += start_m * stride_y_m
    YPtrs = (
        Y
        + offs_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    tl.store(YPtrs, acc, mask=mask_m[:, None] & (offs_y_n < N)[None, :])


# Launch configs, measured on MI300X (gfx942) for MiniMax-M3.1-Flash TP2
# (GEMM1 K=6144 N=3072, GEMM2 K=1536 N=6144), L2/MALL-flushed timing.
# Decode/small tiles (block_m <= 32) keep tl.dot_scaled (split-K fills the GPU);
# prefill tiles (block_m >= 64) use the fp16-dequant kernel.
def _dot_scaled_config(block_m: int, M: int, K: int) -> dict:
    cfg = dict(block_n=128, block_k=128, num_warps=4, split_k=1)
    if K >= 4096:
        if block_m <= 16:
            cfg.update(split_k=4, num_warps=8 if M <= 128 else 4)
        elif block_m == 32:
            cfg.update(block_n=256, block_k=256, num_warps=8)
        else:
            cfg.update(block_n=256)
    else:
        if block_m <= 16 and M <= 8:
            cfg.update(split_k=4, num_warps=8)
        elif block_m == 32:
            cfg.update(block_n=256, num_warps=8)
        elif block_m >= 64:
            cfg.update(block_n=256)
    while cfg["split_k"] > 1 and K % (cfg["block_k"] * cfg["split_k"]):
        cfg["split_k"] //= 2
    return cfg


def _dequant_f16_config(block_m: int, K: int) -> dict:
    if K >= 4096:
        return dict(block_n=128, block_k=64, num_warps=8 if block_m < 128 else 4)
    if block_m < 128:
        return dict(block_n=64, block_k=64, num_warps=4)
    return dict(block_n=256, block_k=64, num_warps=8)


DEQUANT_F16_MIN_BLOCK_M = 64


def moe_gemm_a8w4_nvfp4(
    x,
    w,
    w_scale,
    w_global_scale,
    x_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data=None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
):
    """MoE GEMM with NVFP4 weights using tl.dot_scaled with fp8 scales."""
    assert w.dtype == torch.uint8
    assert w.ndim == 3
    assert w_global_scale.dtype == torch.float32

    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    E, K_packed, N = w.shape
    K = K_packed * 2

    if w_scale.dtype == torch.uint8:
        w_scale = w_scale.view(torch.float8_e4m3fn)
    block_m = routing_data.block_m
    num_stages = 2
    group_m = 4
    xcd_swizzle = get_num_xcds()
    w_cache_modifier = ".cg" if block_m <= 32 else None

    if (
        block_m >= DEQUANT_F16_MIN_BLOCK_M
        and not apply_swiglu
        and bias is None
        and quant_static_scale is None
        and routing_data.expt_data is not None
    ):
        return _moe_gemm_nvfp4_dequant_f16_launch(
            x,
            w,
            w_scale,
            w_global_scale,
            x_static_scale,
            routing_data,
            gather_indx,
            scatter_indx,
            gammas,
            out_dtype,
            M,
            N,
            K,
        )

    cfg = _dot_scaled_config(block_m, M, K)
    block_n, block_k, num_warps = cfg["block_n"], cfg["block_k"], cfg["num_warps"]
    split_k = cfg["split_k"] if not apply_swiglu else 1

    if apply_swiglu and split_k > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
    reduction_n_reduction = 2 if (apply_swiglu and split_k > 1) else 1

    matmul_shape = (split_k, M, N // reduction_n_matmul)
    # split-K partials stay fp32; reduce_grouped sums them and casts once.
    y_dtype = out_dtype if split_k == 1 else torch.float32
    y = torch.empty(matmul_shape, device=x.device, dtype=y_dtype)

    stride_bias = None if bias is None else bias.stride(0)
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    grid = grid_m * grid_n * split_k

    _moe_gemm_nvfp4_dot_scaled[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x.view(torch.uint8),
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale,
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        w_global_scale,
        x_static_scale,
        quant_static_scale,
        bias,
        stride_bias,
        gammas,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        swiglu_add_residual,
        routing_data.n_expts_act,
        block_m,
        block_n,
        block_k,
        group_m,
        XCD_SWIZZLE=xcd_swizzle,
        EVEN_K=K % block_k == 0,
        MASK_K_LIMIT=K % block_k,
        SPLIT_K=split_k,
        W_CACHE_MODIFIER=w_cache_modifier,
        num_warps=num_warps,
        num_stages=num_stages,
        UPCAST_INDICES=should_upcast_indices(x.view(torch.uint8), w, w_scale, y),
    )

    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    if scatter_indx is not None or split_k > 1:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
            if scatter_indx is not None
            else M
        )
        y_final = torch.empty(
            (y_rows, N // reduction_n_matmul // reduction_n_reduction),
            device=x.device,
            dtype=out_dtype,
        )
    else:
        y_final = None

    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu and split_k > 1,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )
    return y_final


def _moe_gemm_nvfp4_dequant_f16_launch(
    x,
    w,
    w_scale,
    w_global_scale,
    x_static_scale,
    routing_data,
    gather_indx,
    scatter_indx,
    gammas,
    out_dtype,
    M,
    N,
    K,
):
    block_m = routing_data.block_m
    cfg = _dequant_f16_config(block_m, K)
    expt_data = routing_data.expt_data
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, cfg["block_n"])
    xv = x.view(torch.uint8)
    wsv = w_scale.view(torch.uint8)
    assert x.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.uint8)
    _moe_gemm_nvfp4_dequant_f16[(grid_m * grid_n,)](
        y,
        y.stride(0),
        y.stride(1),
        xv,
        xv.stride(0),
        xv.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        wsv,
        wsv.stride(0),
        wsv.stride(1),
        wsv.stride(2),
        w_global_scale,
        x_static_scale,
        gammas,
        N,
        K,
        gather_indx,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.token_offs_pad[-1],
        expt_data.block_pid_map,
        grid_m,
        grid_n,
        routing_data.n_expts_act,
        block_m,
        cfg["block_n"],
        cfg["block_k"],
        4,
        XCD_SWIZZLE=get_num_xcds(),
        EVEN_K=K % cfg["block_k"] == 0,
        X_FNUZ=x.dtype == torch.float8_e4m3fnuz,
        num_warps=cfg["num_warps"],
        num_stages=2,
        UPCAST_INDICES=should_upcast_indices(xv, w, wsv, y),
    )
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    y_final = None
    if scatter_indx is not None:
        y_final = torch.empty(
            (scatter_indx.shape[0] // routing_data.n_expts_act, N),
            device=x.device,
            dtype=out_dtype,
        )
    return reduce_grouped(
        y.unsqueeze(0), group_indx, y_final, False, 1.0, 1.0, 1, out_dtype=out_dtype
    )


# Fused per-token fp8 activation quant: bit-identical to
# aiter_nvfp4_moe._static_fp8_quant_torch (bf16 op rounding emulated).
@triton.jit
def _per_token_fp8_quant_kernel(
    X, Y, S, stride_xm, stride_ym, K, BLOCK: tl.constexpr, EMULATE_BF16: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        m = (k0 + offs) < K
        x = tl.load(X + row * stride_xm + k0 + offs, mask=m, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    a = tl.max(amax, 0)
    a = tl.maximum(a, 1e-12)
    if EMULATE_BF16:
        a = a.to(tl.bfloat16).to(tl.float32)  # clamp result stored as bf16
        s = (a / 240.0).to(tl.bfloat16).to(tl.float32)
    else:
        s = a / 240.0
    tl.store(S + row, s.to(S.dtype.element_ty))
    for k0 in range(0, K, BLOCK):
        m = (k0 + offs) < K
        x = tl.load(X + row * stride_xm + k0 + offs, mask=m, other=0.0).to(tl.float32)
        q = x / s
        if EMULATE_BF16:
            q = q.to(tl.bfloat16).to(tl.float32)
        q = tl.minimum(tl.maximum(q, -240.0), 240.0)
        tl.store(Y + row * stride_ym + k0 + offs, q.to(Y.dtype.element_ty), mask=m)


def per_token_fp8_quant(x: torch.Tensor):
    assert x.ndim == 2 and x.stride(-1) == 1
    M, K = x.shape
    fp8 = torch.float8_e4m3fnuz if current_platform.is_rocm() else torch.float8_e4m3fn
    y = torch.empty((M, K), dtype=fp8, device=x.device)
    s = torch.empty((M,), dtype=x.dtype, device=x.device)
    if M == 0:
        return y, s
    BLOCK = min(triton.next_power_of_2(K), 8192)
    _per_token_fp8_quant_kernel[(M,)](
        x,
        y,
        s,
        x.stride(0),
        y.stride(0),
        K,
        BLOCK=BLOCK,
        EMULATE_BF16=x.dtype == torch.bfloat16,
        num_warps=8 if BLOCK >= 4096 else 4,
    )
    return y, s
