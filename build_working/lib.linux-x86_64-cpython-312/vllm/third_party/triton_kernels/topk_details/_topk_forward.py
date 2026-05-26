import triton
import triton.language as tl


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# stable top-k tie-breaks to value with smaller index
@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        # this ensures that we leave at least 16 bits for expert index
        # even if the input dtype is smaller than 16 bits:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    # subtract 1 from loop iterations because we peel the first (masked) iteration:
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # first iteration:
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    # subsequent iterations:
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    # rotate expert index into upper 16 bits:
    # 0000vvvvvvvviiii --> iiii0000vvvvvvvv
    acc = (acc << (y_nbits - 16)) | (acc >> 16)
    # sort in ascending order of expert (descending order of key)
    acc = tl.sort(acc, dim=1, descending=True)
    # iiii0000vvvvvvvv --> 0000iiii:
    y_indices_raw = (acc >> (y_nbits - 16)).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    # iiii0000vvvvvvvv --> vvvvvvvv:
    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    return y_values, y_indices


@triton.jit
def _topk_forward(X, stride_xm,  # inputs
                  Yv, Yi, stride_ym,  # topk values/indices
                  USE_PROVIDED_INDX: tl.constexpr, Bits, stride_rm: tl.constexpr, stride_rn: tl.constexpr,  # bitmatrix
                  n_rows, n_expts_tot,  # shape
                  S, BLOCK_S: tl.constexpr, s_blocks,  # thing to memset
                  APPLY_SOFTMAX: tl.constexpr,  # constant
                  BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):

    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid < s_blocks:
        tl.store(S + BLOCK_S * pid + tl.arange(0, BLOCK_S), tl.zeros([BLOCK_S], tl.int32))

    if pid * BLOCK_M >= n_rows:
        # early exit:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    if USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        y_values, y_indices = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m,  #
                                             N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)

    # write back
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
        tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :] * stride_rn
        tl.store(BitsPtrs, r, mask=mask_m)
