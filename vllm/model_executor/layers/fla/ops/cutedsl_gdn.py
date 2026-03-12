# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CuTe DSL GDN decode kernel (ported from sglang)."""

import operator
import warnings
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack

from vllm.logger import init_logger

logger = init_logger(__name__)


def _to_cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach(), assumed_align=16)


def _supports_f32x2_intrinsics() -> bool:
    # CuTe packed f32x2 intrinsics are only supported on newer architectures.
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


_USE_PACKED_F32X2 = _supports_f32x2_intrinsics()

if _USE_PACKED_F32X2:
    fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
    mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
    add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
    sub_packed_f32x2 = partial(
        cute.arch.calc_packed_f32x2_op,
        src_c=None,
        calc_func=nvvm.sub_packed_f32x2,
        rnd=nvvm.RoundingModeKind.RN,
    )
else:
    # For SM<100, define scalar operations that work with single values
    # The packed functions return tuples, so we mimic that interface
    def fma_packed_f32x2(a, b, c):
        return a[0] * b[0] + c[0], a[1] * b[1] + c[1]

    def mul_packed_f32x2(a, b):
        return a[0] * b[0], a[1] * b[1]

    def add_packed_f32x2(a, b):
        return a[0] + b[0], a[1] + b[1]

    def sub_packed_f32x2(a, b):
        return a[0] - b[0], a[1] - b[1]


torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


@cute.jit
def reduce_dim0(
    input_r: cute.Tensor,
    output: cute.Tensor,
    DIM0: cutlass.Constexpr[int],
    DIM1: cutlass.Constexpr[int],
):
    """Reduce `input_r` along dim0 into `output`."""
    assert output.shape[0] == DIM1
    assert input_r.shape[0][0] == DIM0
    assert input_r.shape[0][1] == DIM1

    input_r_ = cute.make_rmem_tensor_like(output, input_r.element_type)

    # Initialize accumulators
    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
        input_r_[reg_H_idx_y] = input_r[reg_H_idx_y * DIM0]

    # Accumulate along dim0 using scalar operations
    for reg_H_idx_x in cutlass.range_constexpr(1, DIM0, 1, unroll=(DIM0 - 1)):
        for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
            input_r_[reg_H_idx_y] = (
                input_r_[reg_H_idx_y] + input_r[reg_H_idx_y * DIM0 + reg_H_idx_x]
            )

    # Warp reduction
    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
        input_r_[reg_H_idx_y] = cute.arch.warp_reduction(
            input_r_[reg_H_idx_y], operator.add, threads_in_group=cute.arch.WARP_SIZE
        )

    for reg_H_idx_y in cutlass.range_constexpr(0, DIM1, 1, unroll=DIM1):
        output[reg_H_idx_y] = input_r_[reg_H_idx_y].to(output.element_type)


@cute.jit
def L2Norm(
    X: cute.Tensor,
    elem_per_thread: cutlass.Constexpr[int],
):
    """Compute row-wise L2 normalization for one thread fragment."""
    thrX_r = X.load().to(cute.Float32)
    thrX_norm = cute.make_rmem_tensor_like(X, cute.Float32)
    thrX_sum = 0.0
    # Use scalar operations instead of packed
    for reg_X_idx in cutlass.range_constexpr(0, elem_per_thread, 1, unroll=4):
        val = thrX_r[reg_X_idx]
        thrX_norm[reg_X_idx] = val * val
        thrX_sum += thrX_norm[reg_X_idx]

    thrX_sum = cute.arch.warp_reduction(
        thrX_sum, operator.add, threads_in_group=cute.arch.WARP_SIZE
    )

    thrX_rsqrt = cute.rsqrt(thrX_sum + 1e-6)
    for reg_X_idx in cutlass.range_constexpr(0, elem_per_thread, 1, unroll=4):
        thrX_norm[reg_X_idx] = thrX_r[reg_X_idx] * thrX_rsqrt
    return thrX_norm


@cute.kernel
def fused_recurrent_sigmoid_update_kernel_128x32_col(
    gA: cute.Tensor,
    ga: cute.Tensor,
    gdt_bias: cute.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gH: cute.Tensor,
    gO: cute.Tensor,
    gB: cute.Tensor,
    gIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: float,
    tv_layout_k: cute.Layout,
    tv_layout_v: cute.Layout,
    tv_layout_h: cute.Layout,
    T: cutlass.Constexpr[int],
    HK: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    x_threads: int,
    y_threads: int,
    ELEM_H_X: cutlass.Constexpr[int],
    ELEM_H_Y: cutlass.Constexpr[int],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
):
    batch_idx, head_idx, bidx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    if const_expr(cu_seqlens is not None):
        bos, eos = cu_seqlens[batch_idx], cu_seqlens[batch_idx + 1]
        _ = eos - bos  # sequence length, unused
        state_idx = gIndices[batch_idx]
        batch_idx = 0
    else:
        bos = 0
        state_idx = gIndices[batch_idx]

    if state_idx >= 0:
        blk_coord_H = ((None, None, None, None), (state_idx, head_idx, None, bidx))
        blkH = gH[blk_coord_H]

        tidfrgH = cute.composition(blkH, tv_layout_h)

        tArA = gA[head_idx].to(cute.Float32)
        tDrD = gdt_bias[head_idx].to(cute.Float32)

        thrQ_coord = (tidx % x_threads, None)
        thrK_coord = (tidx % x_threads, None)
        thrV_coord = (bidx * y_threads + tidx // x_threads, None)
        thrH_coord = (tidx, None)
        thrO_coord = (bidx * y_threads + tidx // x_threads, None)

        tHgH = tidfrgH[thrH_coord]

        tHrH_i = tHgH.load().to(cute.Float32)
        tHrH_g = cute.make_rmem_tensor_like(tHgH, cute.Float32)
        tHrHk = cute.make_rmem_tensor_like(tHgH, cute.Float32)

        for t_idx in cutlass.range(0, 1, 1, unroll=1):
            blk_coord_a = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
            blk_coord_B = ((None, None, None), (batch_idx, bos + t_idx, head_idx))
            blk_coord_Q = (
                (None, None, None, None),
                (batch_idx, bos + t_idx, head_idx // (HV // HK), None),
            )
            blk_coord_K = (
                (None, None, None, None),
                (batch_idx, bos + t_idx, head_idx // (HV // HK), None),
            )
            blk_coord_V = (
                (None, None, None, None),
                (batch_idx, bos + t_idx, head_idx, None),
            )
            blk_coord_O = (
                (None, None, None, None),
                (batch_idx, bos + t_idx, head_idx, None),
            )

            blka = ga[blk_coord_a]
            blkQ = gQ[blk_coord_Q]
            blkK = gK[blk_coord_K]
            blkV = gV[blk_coord_V]
            blkO = gO[blk_coord_O]
            blkB = gB[blk_coord_B]

            tidfrgQ = cute.composition(blkQ, tv_layout_k)
            tidfrgK = cute.composition(blkK, tv_layout_k)
            tidfrgV = cute.composition(blkV, tv_layout_v)
            tidfrgO = cute.composition(blkO, tv_layout_v)

            tQgQ = tidfrgQ[thrQ_coord]
            tKgK = tidfrgK[thrK_coord]
            tVgV = tidfrgV[thrV_coord]
            tOgO = tidfrgO[thrO_coord]

            tBrBeta = blkB.load()[0].to(cute.Float32)
            tMrMa = blka.load().to(cute.Float32)
            tVrU = cute.make_rmem_tensor_like(tVgV, cute.Float32)
            tVrV = tVgV.load().to(cute.Float32)

            x = tMrMa + tDrD
            beta_x = softplus_beta * x
            softplux_x = cute.where(
                beta_x <= softplus_threshold,
                (1.0 / softplus_beta) * cute.math.log(1.0 + cute.math.exp(beta_x)),
                x,
            )

            tGrG = -cute.math.exp(tArA) * softplux_x
            tBrB = 1.0 / (1.0 + cute.math.exp(-tBrBeta))

            if const_expr(USE_QK_L2NORM_IN_KERNEL):
                tQrQ_norm = L2Norm(tQgQ, ELEM_H_X)
                tKrK = L2Norm(tKgK, ELEM_H_X)
            else:
                tQrQ_norm = tQgQ.load().to(cute.Float32)
                tKrK = tKgK.load().to(cute.Float32)

            # Create a new tensor for scaled Q
            tQrQ = cute.make_rmem_tensor_like(tQgQ, cute.Float32)
            for reg_Q_idx in cutlass.range_constexpr(0, ELEM_H_X, 1, unroll=4):
                tQrQ[reg_Q_idx] = tQrQ_norm[reg_Q_idx] * scale

            tGrGexp = cute.math.exp(tGrG, fastmath=True)[0]
            for reg_H_idx in cutlass.range_constexpr(
                0, ELEM_H_X * ELEM_H_Y, 1, unroll=4
            ):
                tHrH_g[reg_H_idx] = tHrH_i[reg_H_idx] * tGrGexp

            for reg_H_idx in cutlass.range_constexpr(
                0, ELEM_H_X * ELEM_H_Y, 1, unroll=4
            ):
                tHrHk[reg_H_idx] = tHrH_g[reg_H_idx] * tKrK[reg_H_idx % ELEM_H_X]

            reduce_dim0(tHrHk, tVrU, ELEM_H_X, ELEM_H_Y)

            for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_Y, 1, unroll=4):
                tVrU[reg_V_idx] = (tVrV[reg_V_idx] - tVrU[reg_V_idx]) * tBrB

            for reg_K_idx in cutlass.range_constexpr(0, ELEM_H_X, 1, unroll=4):
                for reg_V_idx in cutlass.range_constexpr(0, ELEM_H_Y, 1, unroll=4):
                    idx = reg_V_idx * ELEM_H_X + reg_K_idx
                    tHrHk[idx] = tKrK[reg_K_idx] * tVrU[reg_V_idx] + tHrH_g[idx]

            for reg_H_idx in cutlass.range_constexpr(
                0, ELEM_H_X * ELEM_H_Y, 1, unroll=4
            ):
                tHrH_g[reg_H_idx] = tHrHk[reg_H_idx] * tQrQ[reg_H_idx % ELEM_H_X]

            reduce_dim0(tHrH_g, tOgO, ELEM_H_X, ELEM_H_Y)

        for reg_H_idx in cutlass.range_constexpr(0, ELEM_H_X * ELEM_H_Y, 1, unroll=4):
            tHgH[reg_H_idx] = tHrHk[reg_H_idx].to(tHgH.element_type)


@cute.jit
def fused_recurrent_sigmoid_update_128x32_col(
    mA: cute.Tensor,
    ma: cute.Tensor,
    mdt_bias: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mB: cute.Tensor,
    mIndices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    BK: cutlass.Constexpr[int],
    BV: cutlass.Constexpr[int],
    DIM: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    USE_QK_L2NORM_IN_KERNEL: cutlass.Constexpr[bool],
    stream=None,
):
    h_dtype = mH.element_type
    k_dtype = mK.element_type

    x_threads = 32
    y_threads = 4
    if const_expr(DIM == 256):
        x_threads = 32
        y_threads = 16

    elem_per_thread_k = BK // x_threads
    k_thr_layout = cute.make_ordered_layout((1, x_threads), order=(1, 0))
    k_val_layout = cute.make_ordered_layout(
        (1, elem_per_thread_k * k_dtype.width // 8), order=(1, 0)
    )
    k_val_layout_recast = cute.recast_layout(k_dtype.width, 8, k_val_layout)
    k_tiler_mn, tv_layout_k = cute.make_layout_tv(k_thr_layout, k_val_layout_recast)

    elem_per_thread_v = BV // y_threads
    v_thr_layout = cute.make_ordered_layout((1, y_threads), order=(1, 0))
    v_val_layout = cute.make_ordered_layout(
        (1, elem_per_thread_v * k_dtype.width // 8), order=(1, 0)
    )
    v_val_layout_recast = cute.recast_layout(k_dtype.width, 8, v_val_layout)
    v_tiler_mn, tv_layout_v = cute.make_layout_tv(v_thr_layout, v_val_layout_recast)

    coalesced_bytesl_h_x = BK * h_dtype.width // 8 // x_threads
    elem_per_thread_h_y = BV // y_threads
    thr_h_layout = cute.make_ordered_layout((x_threads, y_threads), order=(0, 1))
    h_val_layout = cute.make_ordered_layout(
        (coalesced_bytesl_h_x, elem_per_thread_h_y), order=(0, 1)
    )
    h_val_layout = cute.recast_layout(h_dtype.width, 8, h_val_layout)
    tiler_mn_h, tv_layout_h = cute.make_layout_tv(thr_h_layout, h_val_layout)
    elem_h_x, elem_h_y = h_val_layout.shape[0], h_val_layout.shape[1]

    gQ = cute.zipped_divide(mQ, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))
    gK = cute.zipped_divide(mK, (1, 1, k_tiler_mn[0], k_tiler_mn[1]))
    gV = cute.zipped_divide(mV, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))
    gO = cute.zipped_divide(mO, (1, 1, v_tiler_mn[0], v_tiler_mn[1]))
    gH = cute.zipped_divide(mH, (1, 1, tiler_mn_h[0], tiler_mn_h[1]))
    gB = cute.zipped_divide(mB, (1, 1, 1))
    gA = mA
    ga = cute.zipped_divide(ma, (1, 1, 1))
    gdt_bias = mdt_bias

    B = mQ.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    T = mK.shape[1]
    HK = mK.shape[2]
    HV = mV.shape[2]
    blocks_per_head = mK.shape[-1] // BV

    fused_recurrent_sigmoid_update_kernel_128x32_col(
        gA,
        ga,
        gdt_bias,
        softplus_beta,
        softplus_threshold,
        gQ,
        gK,
        gV,
        gH,
        gO,
        gB,
        mIndices,
        cu_seqlens,
        scale,
        tv_layout_k,
        tv_layout_v,
        tv_layout_h,
        T,
        HK,
        HV,
        x_threads,
        y_threads,
        elem_h_x,
        elem_h_y,
        USE_QK_L2NORM_IN_KERNEL,
    ).launch(
        grid=[B, HV, blocks_per_head],
        block=[cute.size(tv_layout_h, mode=[0]), 1, 1],
        stream=stream,
    )


def cutedsl_fused_sigmoid_gated_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    stream=None,
):
    del g, gk, gv

    K_input = k.shape[-1]
    V_input = v.shape[-1]

    if initial_state is None or initial_state_indices is None:
        raise ValueError("initial_state and initial_state_indices are required")

    # Kernel expects K-contiguous view (stride[-2] == 1), i.e. [N, HV, K, V]
    # for best performance.
    if initial_state.stride()[-2] != 1:
        warnings.warn(
            "Expected initial_state with K-contiguous layout (stride[-2] == 1).",
            RuntimeWarning,
            stacklevel=2,
        )

    assert K_input == V_input and (K_input == 128 or K_input == 256), (
        "Current cutedsl decode only supports K=V in {128, 256}"
    )

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    if b is None:
        b = torch.ones_like(q[..., 0])

    if b.dim() == 2:
        b = b.unsqueeze(0)
    if a.dim() == 2:
        a = a.unsqueeze(0)

    # Decode fast path for unit-length varlen batches:
    # original shape [1, T, ...] + cu_seqlens can be mapped to [T, 1, ...]
    # and launched without cu_seqlens bookkeeping.
    use_decode_fastpath = (
        cu_seqlens is not None
        and q.shape[0] == 1
        and q.shape[1] == initial_state_indices.numel()
    )
    if use_decode_fastpath:
        q_launch = q.transpose(0, 1)
        k_launch = k.transpose(0, 1)
        v_launch = v.transpose(0, 1)
        a_launch = a.transpose(0, 1)
        b_launch = b.transpose(0, 1)
        cu_seqlens_launch = None
    else:
        q_launch = q
        k_launch = k
        v_launch = v
        a_launch = a
        b_launch = b
        cu_seqlens_launch = cu_seqlens

    B, T, H, K, V = *k_launch.shape, v_launch.shape[-1]
    HV = v_launch.shape[-2]

    q_ = _to_cute_tensor(q_launch)
    k_ = _to_cute_tensor(k_launch)
    v_ = _to_cute_tensor(v_launch)
    h_ = _to_cute_tensor(initial_state)
    b_ = _to_cute_tensor(b_launch)
    ind_ = _to_cute_tensor(initial_state_indices)
    if cu_seqlens_launch is not None:
        cu_seqlens_ = _to_cute_tensor(cu_seqlens_launch)
    else:
        cu_seqlens_ = None
    A_log_ = _to_cute_tensor(A_log)
    a_ = _to_cute_tensor(a_launch)
    dt_bias_ = _to_cute_tensor(dt_bias)

    o = torch.empty_like(v_launch)
    o_ = from_dlpack(o.detach(), assumed_align=16)

    BK = K
    BV = V // 4

    dtype = torch2cute_dtype_map[initial_state.dtype]

    compile_key = (dtype, B, T, H, HV, BV, use_qk_l2norm_in_kernel)

    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compile_key not in cutedsl_fused_sigmoid_gated_delta_rule_update.compile_cache:
        cutedsl_fused_sigmoid_gated_delta_rule_update.compile_cache[compile_key] = (
            cute.compile(
                fused_recurrent_sigmoid_update_128x32_col,
                A_log_,
                a_,
                dt_bias_,
                softplus_beta,
                softplus_threshold,
                q_,
                k_,
                v_,
                h_,
                o_,
                b_,
                ind_,
                cu_seqlens_,
                BK,
                BV,
                DIM=K,
                scale=scale,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                stream=stream,
            )
        )

    cutedsl_fused_sigmoid_gated_delta_rule_update.compile_cache[compile_key](
        A_log_,
        a_,
        dt_bias_,
        q_,
        k_,
        v_,
        h_,
        o_,
        b_,
        ind_,
        cu_seqlens_,
        stream=stream,
    )
    if use_decode_fastpath:
        return o.squeeze(1)
    return o.squeeze(0)


cutedsl_fused_sigmoid_gated_delta_rule_update.compile_cache = {}
