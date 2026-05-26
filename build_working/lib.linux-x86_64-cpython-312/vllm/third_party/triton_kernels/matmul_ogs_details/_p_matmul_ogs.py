# isort: off
# fmt: off
import torch
import triton
import triton.language as tl
from triton.tools.ragged_tma import load_ragged, store_ragged
from triton_kernels import target_info
from triton_kernels.tensor_details.layout_details.blackwell_scale import unswizzle_mx_scale_bw
from triton_kernels.numerics_details.flexpoint import (
    float_to_flex,
    load_scale,
    nan_propagating_absmax_reduce,
    compute_scale,
)
from triton_kernels.numerics_details.mxfp_details._downcast_to_mxfp import MXFP_BLOCK_SIZE
from ._common import make_matmul_repr, matmul_launch_metadata, swizzle2d, xcd_swizzle, get_scaled_dot_format_string


@triton.constexpr_function
def cuda_capability_geq(major, minor):
    return target_info.cuda_capability_geq(major, minor)

@triton.constexpr_function
def get_dtype(tensor_or_desc: tl.tensor | tl.tensor_descriptor) -> tl.dtype:
    if isinstance(tensor_or_desc, tl.tensor):
        return tensor_or_desc.dtype.element_ty
    elif isinstance(tensor_or_desc, tl.tensor_descriptor):
        return tensor_or_desc.dtype
    else:
        raise ValueError(f"Invalid type: {type(tensor_or_desc)}")

@triton.jit
def _load_tile_attrs(
    tile_id, num_tiles, grid_m, grid_n, padding_m,
    M, ExptData, ExptHist, ExptOffs,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr):
    # unpack and swizzle program ids
    pid_emnk = tile_id
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, num_tiles // SPLIT_K, XCD_SWIZZLE)
    pid_e = pid_emnk // ((grid_m - padding_m) * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % ((grid_m - padding_m) * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, (grid_m - padding_m), grid_n, GROUP_M)

    # unpack expert data
    if ExptData is None:
        tl.static_assert(M is not None)
        expt_id, start_z, start_m, block_id, eM = pid_e, pid_e, 0, pid_m, -1
    else:
        tl.static_assert(M is None)
        expt_data = tl.load(ExptData + pid_m)
        expt_id = expt_data & 0x0000FFFF
        block_id = expt_data >> 16
        eM = tl.load(ExptHist + expt_id)
        start_m = tl.load(ExptOffs + expt_id)
        start_z = 0

    off_m = BLOCK_M * block_id
    off_n = BLOCK_N * pid_n

    return expt_id, start_z, start_m, eM, off_m, off_n, pid_k

@triton.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = tl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)


_matmul_ogs_repr = make_matmul_repr("_p_matmul_ogs", [0, 1, 2])
@triton.jit(do_not_specialize=["TOKENS_PER_EXPT_FOR_ANNOTATION"],
            repr=_matmul_ogs_repr, launch_metadata=matmul_launch_metadata)
def _p_matmul_ogs(
             Y, YPtr, stride_y_k, stride_y_z, stride_y_m, stride_y_n,
             YExpectedScale, YActualScale, YChecksumScale,
             stride_y_mx_z, stride_y_mx_m, stride_y_mx_n,
             X, XPtr, stride_x_z, stride_x_m, stride_x_k,
             XScale,
             XMxScale, stride_x_mx_z, stride_x_mx_m, stride_x_mx_k,
             W, WPtr, stride_w_e, stride_w_k, stride_w_n, W_TRANSPOSE: tl.constexpr,
             WScale,
             MxScale, stride_mx_e, stride_mx_k, stride_mx_n,
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
             # NYI: Must be None
             SWIZZLE_MX_VALUE: tl.constexpr,
             # One of ["BLACKWELL", None]
             SWIZZLE_MX_SCALE: tl.constexpr,
             EPILOGUE_SUBTILE: tl.constexpr,
             EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
             W_CACHE_MODIFIER: tl.constexpr,
             NUM_SMS: tl.constexpr,
             X_TMA_MODE: tl.constexpr,
             Y_TMA_MODE: tl.constexpr,
             TOKENS_PER_EXPT_FOR_ANNOTATION=None,
             UPCAST_INDICES:tl.constexpr=False,
             SWAP_XW: tl.constexpr = False,
             IS_EPILOGUE_QUANT_MXFP8: tl.constexpr = False):
    # tl.static_assert(SWIZZLE_MX_VALUE is None, "NYI. Value swizzling")

    # why is this faster than using host-side tensor descriptor?!
    if Y_TMA_MODE is not None:
        Y = tl.make_tensor_descriptor(YPtr, Y.shape, Y.strides[:-1] + (1,), Y.block_shape)

    is_microscaled_format: tl.constexpr = MxScale is not None
    tl.static_assert(not is_microscaled_format or W_TRANSPOSE, "NYI. Non-transposed mxfp4 weights")
    MX_PACK_DIVISOR: tl.constexpr = MXFP_BLOCK_SIZE
    if is_microscaled_format:
        w_type: tl.constexpr = get_dtype(W)
        tl.static_assert(w_type == tl.uint8 or (w_type == tl.float8e4nv or w_type == tl.float8e5),
                         "mx_weight_ptr must be uint8")
        tl.static_assert(get_dtype(MxScale) == tl.uint8, "mx_scale_ptr must be uint8")
        tl.static_assert(BLOCK_K % MX_PACK_DIVISOR == 0, "BLOCK_K must be a multiple of MX_PACK_DIVISOR")
        tl.static_assert(SWIZZLE_MX_SCALE == "BLACKWELL_SCALE" or SWIZZLE_MX_SCALE is None, "Only Blackwell swizzling is supported for scales")

        # We have pack 2 fp4 values in a byte
        W_PACK_DIVISOR: tl.constexpr = 2 if w_type == tl.uint8 else 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K // W_PACK_DIVISOR
        MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR
    else:
        W_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K: tl.constexpr = 1
        PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
        tl.static_assert(SWIZZLE_MX_SCALE is None)

    if ExptOffsSum is not None:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64

    USE_FLEXPOINT_SCALE: tl.constexpr = YActualScale is not None or YChecksumScale is not None
    HAS_SCATTER: tl.constexpr = WriteBackIndx is not None
    HAS_GATHER: tl.constexpr = GatherIndx is not None
    USE_GATHER_TMA: tl.constexpr = HAS_GATHER and X_TMA_MODE == "dense"
    USE_SCATTER_TMA: tl.constexpr = HAS_SCATTER and Y_TMA_MODE == "dense"

    if EPILOGUE_SUBTILE is None:
        SUBTILE_FACTOR: tl.constexpr = 1
    else:
        SUBTILE_FACTOR: tl.constexpr = EPILOGUE_SUBTILE
    EPILOGUE_BLOCK_N: tl.constexpr = BLOCK_N // SUBTILE_FACTOR
    OUT_BLOCK_N: tl.constexpr = EPILOGUE_BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    # set masked out rows to 0
    if HAS_SCATTER and N_EXPTS_ACT == 1:
        # Iterate with reversed pids so that later pids will get more tiles if the number of
        # tiles isn't evenly divisible by the number of SMs.
        # The main loop after this iterates in the forward direction such that earlier
        # pids get more tiles if the number of tiles isn't evenly divisible.
        # This helps balance the work across the SMs.
        for pid_mnk in range(NUM_SMS - tl.program_id(0) - 1, batch_size * grid_m * grid_n * SPLIT_K, NUM_SMS):
            pid_k = pid_mnk % SPLIT_K
            pid_mn = pid_mnk // SPLIT_K
            pid_m, pid_n = swizzle2d(pid_mn, grid_m, grid_n, GROUP_M)

            z = tl.zeros([BLOCK_M, BLOCK_N // ACTIVATION_REDUCTION_N], dtype=tl.float32)
            offs_m = z.shape[0] * pid_m + tl.arange(0, z.shape[0])
            offs_n = z.shape[1] * pid_n + tl.arange(0, z.shape[1])
            src_idx = tl.load(ScatterSrcIndx + offs_m, mask=offs_m < num_idxs, other=0)
            YPtrs = YPtr + offs_m.to(index_type)[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
            mask_n = offs_n < yN
            mask = (src_idx == -1)[:, None] & mask_n[None, :]
            tl.store(YPtrs + pid_k * stride_y_k, z, mask=mask)


    k_tiles = tl.cdiv(K, BLOCK_K * SPLIT_K)
    num_tiles = batch_size * (grid_m - padding_m) * grid_n * SPLIT_K

    # If true, do not share loop-carried variables between the prologue and the
    # epilogue to enable better pipelining with mmav5
    INDEPENDENT_EPILOGUE: tl.constexpr = cuda_capability_geq(10, 0)

    # start negative; will be incremented at the top of the loop
    if INDEPENDENT_EPILOGUE:
        tile_id1 = tl.program_id(0) - NUM_SMS

    # Keep track of local max for updating flexpoint scales.
    THREADS_PER_BLOCK: tl.constexpr = tl.extra.cuda.num_threads()
    local_absmax = tl.full([THREADS_PER_BLOCK], 0.0, tl.uint32)

    DISALLOW_ACC_MULTI_BUFFER: tl.constexpr = is_microscaled_format and BLOCK_M * BLOCK_N >= 128 * 256

    for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER, warp_specialize=True):
        expt_id, start_z, start_m, eM, off_m, off_n, pid_k = _load_tile_attrs(
            tile_id, num_tiles, grid_m, grid_n, padding_m,
            M, ExptData, ExptHist, ExptOffs,
            BLOCK_M, BLOCK_N, SPLIT_K,
            GROUP_M, XCD_SWIZZLE)

        # Base pointers and offsets.
        if X_TMA_MODE is None:
            XBase = X + start_z.to(index_type) * stride_x_z
            offs_x_k = tl.arange(0, BLOCK_K)[None, :] * stride_x_k
            if SPLIT_K > 1:
                offs_x_k += pid_k.to(index_type) * BLOCK_K * stride_x_k

        if USE_GATHER_TMA:
            offs_m = off_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < (M if M is not None else eM)
            if ExptData is None:
                offs_x_m = tl.load(GatherIndx + start_m.to(index_type) + offs_m, mask=mask_m)
                # Bump rows to account for the Z offset.
                offs_x_m += start_z * (stride_x_z // stride_x_m)
                offs_x_m = tl.where(mask_m, offs_x_m, -1)
            else:
                offs_x_m = tl.load(GatherIndx + start_m.to(index_type) + offs_m,
                                    mask=mask_m, other=-N_EXPTS_ACT) // N_EXPTS_ACT
        elif X_TMA_MODE is None:
            tl.static_assert(HAS_GATHER)
            offs_m = off_m + tl.arange(0, BLOCK_M)
            if M is not None:
                offs_m = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
            else:
                offs_m = tl.max_contiguous(tl.multiple_of(offs_m % eM, BLOCK_M), BLOCK_M)
            # no needs to bounds-check here because `offs_m` wraps around M dim
            offs_m = tl.load(GatherIndx + start_m.to(index_type) + offs_m) // N_EXPTS_ACT
            offs_x_m = offs_m.to(index_type)[:, None] * stride_x_m


        acc = tl.zeros((BLOCK_N, BLOCK_M) if SWAP_XW else (BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in tl.range(k_tiles, disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER):
            off_k = pid_k * BLOCK_K + ki * BLOCK_K * SPLIT_K
            off_k_w = pid_k * PACKED_BLOCK_K_W + ki * PACKED_BLOCK_K_W * SPLIT_K
            off_k_mx = pid_k * MX_SCALE_BLOCK_K + ki * MX_SCALE_BLOCK_K * SPLIT_K

            # --- load x ---
            if USE_GATHER_TMA:
                x = X.gather(offs_x_m, off_k)
            elif X_TMA_MODE == "dense":
                x = X.load([start_z, start_m + off_m, off_k])
                x = x.reshape(BLOCK_M, BLOCK_K)
            elif X_TMA_MODE == "ragged":
                x = load_ragged(X, start_m, eM, [start_z, off_m, off_k], ragged_dim=1)
                x = x.reshape(BLOCK_M, BLOCK_K)
            else:
                tl.static_assert(X_TMA_MODE is None)
                XPtrs = XBase + offs_x_m + offs_x_k
                XBase += BLOCK_K * SPLIT_K * stride_x_k
                mask_k = tl.arange(0, BLOCK_K) < K - off_k
                if EVEN_K:
                    if SPLIT_K > 1:
                        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
                    else:
                        x = tl.load(XPtrs)
                else:
                    x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)

            # --- load w ---
            if W_TRANSPOSE:
                w = tl.reshape(W.load([expt_id, off_n, off_k_w]), W.block_shape[1:]).T
            else:
                w = tl.reshape(W.load([expt_id, off_k_w, off_n]), W.block_shape[1:])

            # --- load w_scale ---
            if is_microscaled_format:
                x_format: tl.constexpr = get_scaled_dot_format_string(x.dtype)
                mx_format: tl.constexpr = get_scaled_dot_format_string(w.dtype)
                if x_format == "fp16" or x_format == "bf16":
                    x_scales: tl.constexpr = None
                else:
                    x_scales = tl.full((BLOCK_M, BLOCK_K // MX_PACK_DIVISOR), 127, dtype=tl.uint8)
                if SWIZZLE_MX_SCALE == "BLACKWELL_SCALE":
                    flattened_expt_n_idx = expt_id * ((N + 127) // 128) + (off_n // 128)
                    w_scales = MxScale.load([0, flattened_expt_n_idx, pid_k * MX_SCALE_BLOCK_K // 4 + ki * (MX_SCALE_BLOCK_K // 4 * SPLIT_K), 0, 0])
                    w_scales = w_scales.reshape((w_scales.shape[1], w_scales.shape[2] * w_scales.shape[-2] * w_scales.shape[-1]))
                    w_scales = unswizzle_mx_scale_bw(w_scales)
                else:
                    w_scales = MxScale.load([expt_id, off_k_mx, off_n])
                    w_scales = tl.reshape(w_scales, *w_scales.shape[1:]).T

            # --- update accumulator ---
            if is_microscaled_format:
                if SWAP_XW:
                    acc = tl.dot_scaled(w.T, w_scales, mx_format, x.T, x_scales, x_format, acc=acc, fast_math=True)
                else:
                    acc = tl.dot_scaled(x, x_scales, x_format, w, w_scales, mx_format, acc=acc, fast_math=True)
            else:
                if SWAP_XW:
                    acc = tl.dot(w.T, x.T, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)
                else:
                    acc = tl.dot(x, w, acc, max_num_imprecise_acc=MAX_NUM_IMPRECISE_ACC, allow_tf32=ALLOW_TF32)

        if INDEPENDENT_EPILOGUE:
            tile_id1 += NUM_SMS
            expt_id1, start_z1, start_m1, eM1, off_m1, off_n1, pid_k1 = _load_tile_attrs(
                tile_id1, num_tiles, grid_m, grid_n, padding_m,
                M, ExptData, ExptHist, ExptOffs,
                BLOCK_M, BLOCK_N, SPLIT_K,
                GROUP_M, XCD_SWIZZLE)
        else:
            tile_id1, expt_id1, start_z1, start_m1, eM1 = tile_id, expt_id, start_z, start_m, eM
            off_m1, off_n1, pid_k1 = off_m, off_n, pid_k

        offs_m = off_m1 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < (M if M is not None else eM1)
        if USE_SCATTER_TMA:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE
            if SPLIT_K > 1:
                # Compute the split k offset in number of rows, and add it to offs_y_m.
                # This allows us to write to the correct slice in the output tensor while using
                # a 2D TMA scatter.
                tl.device_assert(stride_y_k // stride_y_m == tl.cdiv(stride_y_k, stride_y_m))
                split_k_row_offs = pid_k1 * (stride_y_k // stride_y_m)
                offs_y_m = tl.where(mask_m, offs_y_m + split_k_row_offs, offs_y_m)
        elif Y_TMA_MODE is None:
            tl.static_assert(HAS_SCATTER)
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            MASK_ACC: tl.constexpr = USE_FLEXPOINT_SCALE
        else:
            offs_y_m = start_m1 + offs_m
            MASK_ACC = False if USE_GATHER_TMA else USE_FLEXPOINT_SCALE

        # bias + scale
        offs_y_n = off_n1 + tl.arange(0, BLOCK_N)
        mask_n = offs_y_n < N
        if B is not None:
            BPtrs = B + expt_id1 * stride_b_e + offs_y_n
            if pid_k1 == 0:
                bias = tl.load(BPtrs, mask=mask_n, other=0)
            else:
                bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        if Betas is not None:
            betas = tl.load(Betas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        if Gammas is not None:
            gammas = tl.load(Gammas + start_m1 + offs_m, mask=mask_m, other=0.0)
        else:
            gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        x_scale = load_scale(XScale)
        if PER_BATCH_SCALE:
            w_scale = load_scale(WScale + expt_id1)
        else:
            w_scale = load_scale(WScale)

        accs = (acc,)
        biases = (bias,)

        if SUBTILE_FACTOR >= 2:
            acc0, acc1 = acc.reshape(BLOCK_M, 2, BLOCK_N // 2).permute(0, 2, 1).split()
            accs = (acc0, acc1)
            bias0, bias1 = bias.reshape(2, BLOCK_N // 2).permute(1, 0).split()
            biases = (bias0, bias1)

        if SUBTILE_FACTOR >= 4:
            acc00, acc01 = acc0.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1).split()
            acc10, acc11 = acc1.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1).split()
            accs = (acc00, acc01, acc10, acc11)
            bias00, bias01 = bias0.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            bias10, bias11 = bias1.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            biases = (bias00, bias01, bias10, bias11)

        tl.static_assert(EPILOGUE_BLOCK_N == BLOCK_N // SUBTILE_FACTOR)
        tl.static_assert(len(accs) == SUBTILE_FACTOR)

        for a_i in tl.static_range(len(accs)):
            acc_tile = accs[a_i]
            acc_tile *= x_scale * w_scale

            if SWAP_XW:
                acc_tile = acc_tile.T

            acc_tile = acc_tile + biases[a_i][None, :] * betas[:, None]
            if out_alpha is not None:
                acc_tile *= out_alpha

            if ACTIVATION_FN is not None:
                out = ACTIVATION_FN(acc_tile, *activation_fn_args)
                tl.static_assert(out.shape[1] == OUT_BLOCK_N, f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})")
            else:
                tl.static_assert(ACTIVATION_REDUCTION_N == 1, "Activation reduction must be 1 if no activation fn is provided")
                out = acc_tile

            out *= gammas[:, None]

            if MASK_ACC:
                out = tl.where(mask_m[:, None], out, 0.0)
            # Flexpoint
            out_view = tl.reshape(out, [out.numel // THREADS_PER_BLOCK, THREADS_PER_BLOCK], can_reorder=True)
            local_absmax = tl.maximum(local_absmax, nan_propagating_absmax_reduce(out_view, axis=0))
            out = float_to_flex(
                out, YExpectedScale,
                None, # ActualScale: local absmax is tracked and updated after the loop
                YChecksumScale,
                None, # mask: out is manually masked to 0
                YPtr, FLEXPOINT_SATURATE_INF
            )
            if EPILOGUE_FN is not None:
                out = EPILOGUE_FN(out, *epilogue_fn_args, target_dtype=YPtr.dtype.element_ty, pid=len(accs)*tile_id1 + a_i)

            out_off_n = off_n1 // ACTIVATION_REDUCTION_N + a_i * OUT_BLOCK_N
            out = out.to(YPtr.dtype.element_ty)
            if USE_SCATTER_TMA:
                # Convert -1 offsets to INT_MAX. We do this by clearing the leading bit. Note that
                # there shouldn't be any other negative values.
                offs_y_m = (offs_y_m.to(tl.uint32, bitcast=True) & 0x7FFFFFFF).to(tl.int32, bitcast=True)
                Y.scatter(out, offs_y_m, out_off_n)
            elif Y_TMA_MODE == "dense":
                out = tl.reshape(out, [1] + out.shape)
                off_kz = pid_k * batch_size + start_z1
                Y.store([off_kz, off_m1, out_off_n], out)
            elif Y_TMA_MODE == "ragged":
                out = tl.reshape(out, [1] + out.shape)
                store_ragged(Y, start_m1, eM1, [pid_k, off_m1, out_off_n], out, ragged_dim=1)
            else:
                tl.static_assert(Y_TMA_MODE is None)
                offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN

                YPtrs = YPtr + pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_n
                mask = mask_m[:, None] & mask_n[None, :]
                tl.store(YPtrs, out, mask=mask)


    # Update the flexpoint scales
    if YActualScale is not None:
        tl.atomic_max(YActualScale, compute_scale(local_absmax.to(tl.float32, bitcast=True), YPtr), sem="relaxed")


_per_device_alloc_fns = {}
def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = {}
        def alloc_fn(size: int, alignment: int, stream):
            assert alignment == 128
            if stream not in _per_stream_tensors or _per_stream_tensors[stream].numel() < size:
                _per_stream_tensors[stream] = torch.empty(size, device=device, dtype=torch.int8)
                _per_stream_tensors[stream].__hibernate__ = {"type": "ignore"}
            return _per_stream_tensors[stream]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]
