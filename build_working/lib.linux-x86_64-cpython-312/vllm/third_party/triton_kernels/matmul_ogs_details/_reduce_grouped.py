from triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from triton_kernels.numerics_details.mxfp import quantize_mxfp8_fn
import triton
import triton.language as tl


@triton.jit
def _reduce_grouped(X, stride_xb: tl.uint64, stride_xm: tl.uint64, stride_xn,  #
                    XScale,  # input scalar flex scale
                    Out, stride_om: tl.uint64, stride_on,  # output tensor
                    OutExpectedScale, OutActualScale, OutChecksumScale,  # output scalar flex scales
                    InIndx, B, N,  #
                    XMxScale, stride_mxb: tl.uint64,
                    stride_mxs: tl.uint64,  # optional per-32-col output MXFP scales (uint8)
                    OutMxScale, stride_omxs: tl.uint64,  # optional per-32-col output MXFP scales (uint8)
                    # fused activation function
                    ACTIVATION_FN: tl.constexpr, activation_fn_args, ACTIVATION_REDUCTION_N: tl.constexpr,
                    # epilogue transform
                    EPILOGUE_FN: tl.constexpr, epilogue_fn_args,
                    #
                    HAS_IN_MX_SCALE: tl.constexpr, HAS_OUT_MX_SCALE: tl.constexpr, FLEXPOINT_SATURATE_INF: tl.constexpr,
                    K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_t = tl.program_id(0)
    BLOCK_N_OUT: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    # persistent along N: single program on N, iterate tiles of size BLOCK_N
    start = pid_t * K
    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t, )
    else:
        indxs = ()
        for i in tl.static_range(0, K):
            indxs = indxs + (tl.load(InIndx + start + i), )
    # determine first valid topk row
    fi = indxs[(K - 1)]
    for i in tl.static_range(K - 2, -1, -1):
        fi = tl.where(indxs[i] != -1, indxs[i], fi)
    # record overwritten row index (may be -1 if none)
    XPtrs = X + tl.arange(0, BLOCK_N) * stride_xn
    OutPtrs = Out + tl.arange(0, BLOCK_N_OUT) * stride_on
    if HAS_IN_MX_SCALE:
        XScalePtrs = XMxScale + tl.arange(0, BLOCK_N // 32) * stride_xn
    if HAS_OUT_MX_SCALE:
        OutScalePtrs = OutMxScale + tl.arange(0, BLOCK_N_OUT // 32) * stride_on
    x_scale = load_scale(XScale)
    for n_curr in tl.range(0, N, BLOCK_N, num_stages=4):
        acc = tl.zeros([BLOCK_N_OUT], dtype=tl.float32)
        x_n_mask = tl.arange(0, BLOCK_N) < N - n_curr
        x_n_mask_scale = tl.arange(0, BLOCK_N // 32) < tl.cdiv(N - n_curr, 32)
        # accumulate contributions for this tile
        for i in tl.static_range(0, K):
            curr = tl.zeros([BLOCK_N], dtype=tl.float32)
            # iterate over split_k partial values
            for b in tl.range(0, B):
                is_valid = indxs[i] != -1
                x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb
                vals = tl.load(x_row_ptr, mask=x_n_mask & is_valid, other=0.0)
                vals = vals.to(tl.float32)
                if HAS_IN_MX_SCALE:
                    scale_row_ptr = XScalePtrs + indxs[i] * stride_mxs + b * stride_mxb
                    scale = tl.load(scale_row_ptr, mask=x_n_mask_scale & is_valid, other=0.)
                    scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
                    vals = vals.reshape([BLOCK_N // 32, 32])
                    vals = (scale[:, None] * vals).reshape([BLOCK_N])
                curr += vals
            # apply nonlinearity to split-k output
            if ACTIVATION_FN is not None:
                curr = ACTIVATION_FN(curr[None, :], *activation_fn_args)
            curr = tl.reshape(curr, [curr.shape[-1]])
            # update final accumulator
            acc += curr
        acc *= x_scale
        # Compute per-32-col MXFP scales for this tile if requested
        Nrem = (N - n_curr) // ACTIVATION_REDUCTION_N
        out_n_mask = tl.arange(0, BLOCK_N_OUT) < Nrem
        out_n_mask_scale = tl.arange(0, BLOCK_N_OUT // 32) < tl.cdiv(Nrem, 32)
        if HAS_OUT_MX_SCALE:
            acc, acc_scale = quantize_mxfp8_fn(acc[None, :], out_n_mask[None, :])
            acc = tl.reshape(acc, [acc.shape[-1]])
            acc_scale = tl.reshape(acc_scale, [acc_scale.shape[-1]])
        # Convert to flexpoint output if configured (scalar scales)
        acc = float_to_flex(acc, OutExpectedScale, OutActualScale, OutChecksumScale, None, Out, FLEXPOINT_SATURATE_INF)
        # write-back for this tile
        out_ptr = OutPtrs + pid_t * stride_om
        tl.store(out_ptr, acc, mask=out_n_mask)
        if HAS_OUT_MX_SCALE:
            out_scale_ptr = OutScalePtrs + pid_t * stride_omxs
            tl.store(out_scale_ptr, acc_scale, mask=out_n_mask_scale)
        XPtrs += BLOCK_N * stride_xn
        OutPtrs += BLOCK_N_OUT * stride_on
        if HAS_IN_MX_SCALE:
            XScalePtrs += BLOCK_N // 32 * stride_xn
        if HAS_OUT_MX_SCALE:
            OutScalePtrs += BLOCK_N_OUT // 32 * stride_xn
