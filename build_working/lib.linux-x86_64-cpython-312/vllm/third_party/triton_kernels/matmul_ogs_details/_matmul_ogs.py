# isort: off
# fmt: off
import triton
import triton.language as tl
from triton_kernels.tensor_details.layout_details.blackwell_scale import unswizzle_mx_scale_bw
from triton_kernels.tensor_details.layout_details.hopper_scale import unswizzle_mxfp4_scale_hopper
from triton_kernels.tensor_details.layout_details.hopper_value import mxfp4_to_bf16_triton
from triton_kernels.tensor_details.layout_details.cdna4_scale import unswizzle_mx_scale_cdna4
from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE
from ._common import make_matmul_repr, matmul_launch_metadata, swizzle2d, xcd_swizzle, get_scaled_dot_format_string


@triton.jit
def _zero_masked_rows(
        pid_m, pid_n,
        Y, stride_y_m, stride_y_n,
        N,
        ScatterSrcIndx, num_idxs,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    offs_m = BLOCK_M * pid_m.to(tl.int64) + tl.arange(0, BLOCK_M)
    offs_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    src_idx = tl.load(ScatterSrcIndx + offs_m, mask=offs_m < num_idxs, other=0)
    YPtrs = Y + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    mask_n = offs_n < N
    mask = (src_idx == -1)[:, None] & mask_n[None, :]
    tl.store(YPtrs, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), mask=mask)


_matmul_ogs_repr = make_matmul_repr("_matmul_ogs", [0, 1, 2])
@triton.jit(do_not_specialize=["TOKENS_PER_EXPT_FOR_ANNOTATION"],
            repr=_matmul_ogs_repr, launch_metadata=matmul_launch_metadata)
def _matmul_ogs(
             Y, YPtr, stride_y_k, stride_y_z, stride_y_m, stride_y_n,
             YExpectedScale, YActualScale, YChecksumScale,
             stride_y_mx_z, stride_y_mx_m, stride_y_mx_n,
             X, XPtr, stride_x_z, stride_x_m, stride_x_k,
             XScale,
             XMxScale, stride_x_mx_z, stride_x_mx_m, stride_x_mx_k,
             W, WPtr, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: tl.constexpr,
             WScale,
             WMxScale, stride_w_mx_e, stride_w_mx_k, stride_w_mx_n,
             B, stride_b_e, # Bias
             NRows, M, N, K, # shapes
             # expt data
             Betas, Gammas,
             GatherIndx,
             ScatterSrcIndx, num_idxs,
             WriteBackIndx, writeback_size,
             ExptHist, ExptOffs, ExptOffsSum, ExptData,
             # true grid size
             batch_size, grid_m, grid_n,
             # Out scale
             out_alpha,
             # fused activation function
             ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
             # epilogue transform
             EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
             # MoE config
             N_EXPTS_TOT: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
             # precision config
             MAX_NUM_IMPRECISE_ACC: tl.constexpr, ALLOW_TF32: tl.constexpr,
             FLEXPOINT_SATURATE_INF: tl.constexpr,
             PER_BATCH_SCALE: tl.constexpr,
             # optimization config
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr,
             # One of ["HOPPER", "BLACKWELL", None]
             SWIZZLE_MX_VALUE: tl.constexpr,
             # One of ["HOPPER", "BLACKWELL", None]
             SWIZZLE_MX_SCALE: tl.constexpr,
             EPILOGUE_SUBTILE: tl.constexpr,
             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
             W_CACHE_MODIFIER: tl.constexpr,
             NUM_SMS: tl.constexpr,
             X_TMA_MODE: tl.constexpr,
             Y_TMA_MODE: tl.constexpr,
             TOKENS_PER_EXPT_FOR_ANNOTATION=None,
             UPCAST_INDICES: tl.constexpr = False,
             SWAP_XW: tl.constexpr = False,
             IS_EPILOGUE_QUANT_MXFP8: tl.constexpr = False):

    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_z >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_z >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if stride_w_mx_e is not None:
        tl.assume(stride_w_mx_e >= 0)
    if stride_w_mx_k is not None:
        tl.assume(stride_w_mx_k >= 0)
    if stride_w_mx_n is not None:
        tl.assume(stride_w_mx_n >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(batch_size >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    is_w_microscaled: tl.constexpr = WMxScale is not None
    MX_PACK_DIVISOR: tl.constexpr = MXFP_BLOCK_SIZE
    if is_w_microscaled:
        w_type: tl.constexpr = W.dtype.element_ty
        is_mxfp4: tl.constexpr = w_type == tl.uint8
        tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                         "mx_weight_ptr must be uint8 or fp8")
        tl.static_assert(WMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")
        tl.static_assert(SWIZZLE_MX_VALUE == "HOPPER_VALUE" or SWIZZLE_MX_VALUE is None, "Only Hopper swizzling is supported for values")
    else:
        tl.static_assert(SWIZZLE_MX_VALUE is None)
        tl.static_assert(SWIZZLE_MX_SCALE is None)
    is_x_microscaled: tl.constexpr = XMxScale is not None
    if is_x_microscaled:
        x_type: tl.constexpr = X.dtype.element_ty
        tl.static_assert(is_w_microscaled)
        tl.static_assert(x_type == tl.float8e4nv, "mx_act_ptr must be float8e4nv")
        tl.static_assert(XMxScale.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")
    is_out_microscaled: tl.constexpr = stride_y_mx_z is not None

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    HAS_FUSED_SCATTER: tl.constexpr = WriteBackIndx is not None
    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = batch_size * unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        tl.device_assert(batch_size == 0)
        pid_mn = pid - total_actual_tiles
        if pid_mn < padding_m * grid_n:
            pid_m, pid_n = swizzle2d(pid_mn, padding_m, grid_n, GROUP_M)

            # set masked out rows to 0
            if HAS_FUSED_SCATTER and N_EXPTS_ACT == 1:
                _zero_masked_rows(pid_m, pid_n, Y, stride_y_m, stride_y_n, yN, ScatterSrcIndx, num_idxs, BLOCK_M, OUT_BLOCK_N)
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = swizzle2d(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to( index_type) * stride_y_k
        if is_out_microscaled:
            YActualScale += pid_k.to(index_type) * stride_x_mx_k
    # set masked out rows to 0
    if HAS_FUSED_SCATTER and N_EXPTS_ACT == 1:
        _zero_masked_rows(pid_m, pid_n, Y, stride_y_m, stride_y_n, yN, ScatterSrcIndx, num_idxs, BLOCK_M, OUT_BLOCK_N)
    # unpack expert data
    if ExptData is None:
        tl.static_assert(M is not None)
        expt_id, start_z, start_m, block_id = pid_e, pid_e, 0, pid_m
    else:
        tl.static_assert(M is None)
        expt_data = tl.load(ExptData + pid_m)
        if expt_data == -1:
            return
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        M = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        start_z = 0
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m, start_z = start_m.to(index_type), start_z.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)
    # A pointers
    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    X += start_z * stride_x_z
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m.to(index_type)[:, None] * stride_x_m + offs_k.to(index_type)[None, :] * stride_x_k

    # TODO: refactor if/else when triton front end improves
    if is_w_microscaled:
        if SWIZZLE_MX_VALUE == "HOPPER_VALUE":
            tl.static_assert(is_mxfp4, "Only mxfp4 is supported for HOPPER swizzling")
            tl.static_assert(not is_x_microscaled)
            # We have pack 2 fp4 values in a byte but we divide the dimension by 2
            # when swizzling
            W_K_DIVISOR: tl.constexpr = 1
            W_K_MULTIPLIER: tl.constexpr = 2
            W_N_DIVISOR: tl.constexpr = 4
        else:
            # We have pack 2 fp4 values in a  byte
            W_K_DIVISOR: tl.constexpr = 2 if is_mxfp4 else 1
            W_K_MULTIPLIER: tl.constexpr = 1
            W_N_DIVISOR: tl.constexpr = 1

        if W_TRANSPOSE:
            # When weight is transposed, 2 fp4 values are packed per Byte along
            # the contiguous dimension, K.
            PACKED_BLOCK_K_W: tl.constexpr = (BLOCK_K // W_K_DIVISOR) * W_K_MULTIPLIER
            PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_N_DIVISOR
        else:
            # When weight is not transposed, fp4 values are *not* packed along
            # the contiguous dimension, N.
            PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
            PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N // W_K_DIVISOR
        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR

        WMxScale += expt_id * stride_w_mx_e

        if SWIZZLE_MX_SCALE == "BLACKWELL_SCALE":
            # TODO: support non W_TRANSPOSE with blackwell swizzling
            tl.static_assert(W_TRANSPOSE)
            tl.static_assert(BLOCK_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K % 4 == 0)
            PACKED_MX_BLOCK: tl.constexpr = (MX_SCALE_BLOCK_K // 4) * 32 * 4 * 4
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // 128
            stride_scale_k: tl.constexpr = 1
        elif SWIZZLE_MX_SCALE == "HOPPER_SCALE":
            # TODO: support non W_TRANSPOSE with Hopper swizzling
            tl.static_assert(W_TRANSPOSE)
            n_warps: tl.constexpr = tl.extra.cuda.num_warps()
            tl.static_assert(BLOCK_N % (2 * n_warps * 2 * 8) == 0)
            tl.static_assert(MX_SCALE_BLOCK_K % 2 == 0)
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * 32
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // 32
            stride_scale_k = stride_w_mx_k
        elif SWIZZLE_MX_SCALE == "CDNA4_SCALE":
            tl.static_assert(stride_w_mx_k is not None)
            tl.static_assert(stride_w_mx_n is not None)
            NON_K_PRESHUFFLE_BLOCK_SIZE: tl.constexpr = 32
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N // NON_K_PRESHUFFLE_BLOCK_SIZE
            stride_scale_k = stride_w_mx_k
        else:
            PACKED_MX_BLOCK: tl.constexpr = MX_SCALE_BLOCK_K
            SCALE_BLOCK_N: tl.constexpr = BLOCK_N
            stride_scale_k = stride_w_mx_k
        offs_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)) % N
        offs_n_scale = tl.max_contiguous(tl.multiple_of(offs_n_scale, SCALE_BLOCK_N), SCALE_BLOCK_N)
        # K dimension must be the last dimension for the scales
        offs_k_scale = PACKED_MX_BLOCK * pid_k + tl.arange(0, PACKED_MX_BLOCK)
        WMxScalePtrs = WMxScale + offs_k_scale.to(index_type)[None, :] * stride_scale_k + offs_n_scale.to(index_type)[:, None] * stride_w_mx_n
    else:
        WMxScalePtrs = None
        offs_k_scale = None
        W_K_DIVISOR: tl.constexpr = 1
        W_K_MULTIPLIER: tl.constexpr = 1
        W_N_DIVISOR: tl.constexpr = 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
        PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N

    # B pointers
    offs_w_n = pid_n * PACKED_BLOCK_N_W + tl.arange(0, PACKED_BLOCK_N_W)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % (N // W_N_DIVISOR), PACKED_BLOCK_N_W), PACKED_BLOCK_N_W)

    if is_x_microscaled:
        XMxScale += start_z.to(index_type) * stride_x_mx_z
        if GatherIndx is None:
            XMxScale += start_m * stride_x_mx_m
        offs_x_k_scale = MX_SCALE_BLOCK_K * pid_k + tl.arange(0, MX_SCALE_BLOCK_K)
        XMxScalePtrs = XMxScale + offs_x_m.to(index_type)[:, None] * stride_x_mx_m + offs_x_k_scale.to(index_type)[None, :] * stride_x_mx_k
    else:
        XMxScalePtrs = None

    offs_w_k = PACKED_BLOCK_K_W * pid_k + tl.arange(0, PACKED_BLOCK_K_W)
    W += expt_id * stride_w_e
    WPtrs = W + (offs_w_k.to(index_type)[:, None] * stride_w_k + offs_w_n.to(index_type)[None, :] * stride_w_n)
    # compute output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, BLOCK_K * pid_k, -(BLOCK_K * SPLIT_K)):
        if EVEN_K:
            mask_k = tl.full([BLOCK_K], True, dtype=tl.int1)
            mask_k_w = tl.full([PACKED_BLOCK_K_W], True, dtype=tl.int1)
            if is_w_microscaled and SWIZZLE_MX_SCALE is None:
                mask_k_scale = tl.full([PACKED_MX_BLOCK], True, dtype=tl.int1)
            if is_x_microscaled:
                mask_x_k_scale = tl.full([MX_SCALE_BLOCK_K], True, dtype=tl.int1)
        else:
            mask_k = offs_k < k
            mask_k_w = offs_w_k < ((k // (W_K_DIVISOR if W_TRANSPOSE else 1)) * W_K_MULTIPLIER)
            if is_w_microscaled and SWIZZLE_MX_SCALE is None:
                mask_k_scale = offs_k_scale * MX_PACK_DIVISOR < k
            if is_x_microscaled:
                mask_x_k_scale = offs_x_k_scale * MX_PACK_DIVISOR < k

        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_k_w[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER)
        if is_w_microscaled:
            x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
            w_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)

            if is_x_microscaled:
                x_scales = tl.load(XMxScalePtrs, mask=mask_x_k_scale[None, :])
            elif x_format == "fp16" or x_format == "bf16":
                x_scales: tl.constexpr = None
            else:
                # Scale of 1 in E8M0 format
                x_scales = tl.full((BLOCK_M, MX_SCALE_BLOCK_K), 127, dtype=tl.uint8)

            if SWIZZLE_MX_SCALE == "BLACKWELL_SCALE":
                w_scales = unswizzle_mx_scale_bw(tl.load(WMxScalePtrs))
            elif SWIZZLE_MX_SCALE == "HOPPER_SCALE":
                # Handshake with the swizzling code
                num_warps: tl.constexpr = tl.extra.cuda.num_warps()
                w_scales = unswizzle_mxfp4_scale_hopper(tl.load(WMxScalePtrs), mx_axis=1, num_warps=num_warps)
            elif SWIZZLE_MX_SCALE == "CDNA4_SCALE":
                w_scales = unswizzle_mx_scale_cdna4(tl.load(WMxScalePtrs), BLOCK_N, MX_SCALE_BLOCK_K)
            else:
                w_scales = tl.load(WMxScalePtrs, mask=mask_k_scale[None, :])

            if SWIZZLE_MX_VALUE == "HOPPER_VALUE":
                # Handshake with the swizzling code
                tl.static_assert(x_format == "bf16")
                tl.static_assert(w_format == "e2m1")
                w = mxfp4_to_bf16_triton(w.trans(), w_scales, 1)
                tl.static_assert(w.dtype == tl.bfloat16)
                acc = acc.trans()
                x = x.trans()
                # w = w.trans()
                acc = tl.dot(w, x, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                acc = acc.trans()
            else:
                rhs_k_pack: tl.constexpr = W_TRANSPOSE or not is_w_microscaled or W_K_DIVISOR != 2
                acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, w_format, acc=acc, fast_math=True, rhs_k_pack=rhs_k_pack)
            if SWIZZLE_MX_SCALE == "BLACKWELL_SCALE":
                WMxScalePtrs += (MX_SCALE_BLOCK_K // 4 * SPLIT_K) * stride_w_mx_k
            else:
                WMxScalePtrs += (PACKED_MX_BLOCK * SPLIT_K) * stride_w_mx_k
            if is_x_microscaled:
                XMxScalePtrs += (MX_SCALE_BLOCK_K * SPLIT_K) * stride_x_mx_k
        else:
            acc = tl.dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K_W * SPLIT_K) * stride_w_k
    # bias + scale
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
    else:
        bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
    if Betas is not None:
        betas = tl.load(Betas + start_m + offs_m, mask=mask_m, other=0.0)
    else:
        betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
    else:
        gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    # flexpoint
    x_scale = load_scale(XScale)
    if PER_BATCH_SCALE:
        w_scale = load_scale(WScale + expt_id)
    else:
        w_scale = load_scale(WScale)
    acc *= x_scale * w_scale
    acc = acc + bias[None, :] * betas[:, None]
    if out_alpha is not None:
        acc *= out_alpha
    if ACTIVATION_FN is not None:
        out = ACTIVATION_FN(acc, *activation_fn_args)
        tl.static_assert(out.shape[1] == OUT_BLOCK_N, f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})")
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(ACTIVATION_REDUCTION_N == 1, "Activation reduction must be 1 if no activation fn is provided")
        out = acc
    out *= gammas[:, None]
    # write-back
    Y += start_z.to(index_type) * stride_y_z
    if WriteBackIndx is not None:
        WriteBackIndx += start_m
        dst_idx = tl.load(WriteBackIndx + offs_m, mask=start_m + offs_m < writeback_size, other=-1)
        mask_m = mask_m & (dst_idx != -1)
        offs_y_m = dst_idx
    else:
        Y += start_m * stride_y_m
        offs_y_m = offs_m

    YPtrs = Y + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    if is_out_microscaled:
        MX_SCALE_BLOCK_N: tl.constexpr = BLOCK_N // MXFP_BLOCK_SIZE
        N_MX_BLOCK: tl.constexpr = tl.cdiv(N, MXFP_BLOCK_SIZE)
        tl.static_assert(EPILOGUE_FN is not None)
        out, out_scale = EPILOGUE_FN(out, mask, *epilogue_fn_args)
        tl.static_assert(BLOCK_N % MX_SCALE_BLOCK_N == 0, "")
        offs_y_n_scale = MX_SCALE_BLOCK_N * pid_n + tl.arange(0, MX_SCALE_BLOCK_N)
        mask_n_scale = offs_y_n_scale < N_MX_BLOCK
        YActualScale += start_z.to(index_type) * stride_y_mx_z
        if WriteBackIndx is None:
            YActualScale += start_m * stride_y_mx_m
            YActualScalePtrs = YActualScale + offs_y_m.to(index_type)[:, None] * stride_y_mx_m + offs_y_n_scale.to(index_type)[None, :] * stride_y_mx_n
        else:
            YActualScalePtrs = YActualScale + (offs_y_m - NRows).to(index_type)[:, None] * stride_y_mx_m + offs_y_n_scale.to(index_type)[None, :] * stride_y_mx_n
        tl.store(YActualScalePtrs, out_scale, mask=mask_m[:, None] & mask_n_scale[None, :])
    else:
        out = float_to_flex(out, YExpectedScale, YActualScale, YChecksumScale, mask, Y, FLEXPOINT_SATURATE_INF)
        if EPILOGUE_FN is not None and not IS_EPILOGUE_QUANT_MXFP8:
            out = EPILOGUE_FN(out, *epilogue_fn_args, target_dtype=YPtrs.dtype.element_ty)
    tl.store(YPtrs, out, mask=mask)
