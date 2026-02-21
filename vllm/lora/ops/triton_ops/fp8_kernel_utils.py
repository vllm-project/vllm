# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for Punica kernel construction.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _accumulate_mm(
    tiled_a,
    tiled_b,
    accumulator,
    a_scale_ptr,
    b_scale_ptr,
    a_scale_k_stride,
    b_scale_k_stride,
    iter_k,
    group_k: tl.constexpr,
    group_n: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    Core matrix multiplication and accumulation logic with quantization support.

    Args:
        tiled_a: Loaded tile from A matrix
        tiled_b: Loaded tile from B matrix
        accumulator: Current accumulator value
        a_scale_ptr: Scale pointer for A matrix
        b_scale_ptr: Scale pointer for B matrix
        a_scale_k_stride: K dimension stride for A's block-wise scales
        b_scale_k_stride: K dimension stride for B's block-wise scales
        iter_k: Current iteration's global K offset
        group_k: Block size for K dimension in block-wise quantization
        group_n: Block size for N dimension in block-wise quantization
        use_fp8_w8a8: Whether using FP8 W8A8 quantization
        use_int8_w8a8: Whether using INT8 W8A8 quantization
        use_int8_w8a16: Whether using INT8 W8A16 quantization

    Returns:
        Updated accumulator
    """
    if use_int8_w8a16:
        accumulator = tl.dot(tiled_a, tiled_b, acc=accumulator)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # Block-wise quantization: scales are loaded per block
            offs_ks = iter_k // group_k
            # a_scale_ptr is (BLOCK_M,) tensor of base pointers per row
            # Load scale for current K-group, result shape: (BLOCK_M,)
            a_scale = tl.load(a_scale_ptr + offs_ks * a_scale_k_stride)
            # b_scale_ptr is (BLOCK_N,) tensor with N-offset pre-baked
            # Load scale for current K-group, result shape: (BLOCK_N,)
            b_scale = tl.load(b_scale_ptr + offs_ks * b_scale_k_stride)
            accumulator += (
                tl.dot(tiled_a, tiled_b) * a_scale[:, None] * b_scale[None, :]
            )
        elif use_fp8_w8a8:
            # Tensor-wise or per-channel: accumulate and scale at end
            accumulator = tl.dot(tiled_a, tiled_b, acc=accumulator)
    else:
        accumulator += tl.dot(tiled_a, tiled_b)
    return accumulator


@triton.jit
def fp8_mm_k(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    ak_stride,
    bk_stride,
    a_scale_k_stride,
    b_scale_k_stride,
    offset_k,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    group_k: tl.constexpr,
    group_n: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    base_k,
):
    """
    FP8-compatible matrix multiplication kernel with quantization support.
    Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
    B (k x n), iterate through the K dimension to compute the partial/complete
    matrix block product with proper dequantization.

    Args:
        a_ptr: Array of pointers, identifying rows of A (FP8 or other dtype)
        b_ptr: Array of pointers, identifying columns of B (FP8 dtype)
        a_scale_ptr: Scale pointer for A matrix (per-token or block-wise)
        b_scale_ptr: Scale pointer for B matrix (per-channel or block-wise)
        ak_stride: K dimension stride of the A matrix
        bk_stride: K dimension stride of the B matrix
        a_scale_k_stride: K dimension stride for A's block-wise scales
        b_scale_k_stride: K dimension stride for B's block-wise scales
        offset_k: Base offset along K dimension
        K: Length of the K dimension
        BLOCK_M: M dimension of the output block m x n
        BLOCK_N: N dimension of the output block m x n
        BLOCK_K: K dimension atom
        EVEN_K: True if the blocks of A and B can be loaded without masking
        SPLIT_K: Parameter signifying parallelism in the K dimension
        group_k: Block size for K dimension in block-wise quantization
        group_n: Block size for N dimension in block-wise quantization
        use_fp8_w8a8: Whether using FP8 W8A8 quantization
        use_int8_w8a8: Whether using INT8 W8A8 quantization
        use_int8_w8a16: Whether using INT8 W8A16 quantization
        per_channel_quant: Whether using per-channel quantization
        CAST_TYPE: if True, cast the values from the A matrix to the B
          matrix dtype.
        b_dtype: datatype of the B matrix
        USE_GDC: Whether to use PDL. True indicates use.
        base_k: Base offset along K dimension for current SPLIT_K group
    """
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Step size along K for each iteration
    STEP_K = BLOCK_K * SPLIT_K

    # Total number of iterations (compile-time constant)
    num_iters = tl.cdiv(K, STEP_K)

    for k in range(num_iters):
        # Current iteration's global K offset
        iter_k = k * STEP_K + base_k
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            # K is divisible by BLOCK_K, no masking ever needed
            tiled_b = tl.load(b_ptr)
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            tiled_a = tl.load(a_ptr)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)

            accumulator = _accumulate_mm(
                tiled_a,
                tiled_b,
                accumulator,
                a_scale_ptr,
                b_scale_ptr,
                a_scale_k_stride,
                b_scale_k_stride,
                iter_k,
                group_k,
                group_n,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
            )
        else:
            if iter_k >= K:
                pass
            elif block_end <= K:
                tiled_b = tl.load(b_ptr)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)

                accumulator = _accumulate_mm(
                    tiled_a,
                    tiled_b,
                    accumulator,
                    a_scale_ptr,
                    b_scale_ptr,
                    a_scale_k_stride,
                    b_scale_k_stride,
                    iter_k,
                    group_k,
                    group_n,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                )
            else:
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)

                accumulator = _accumulate_mm(
                    tiled_a,
                    tiled_b,
                    accumulator,
                    a_scale_ptr,
                    b_scale_ptr,
                    a_scale_k_stride,
                    b_scale_k_stride,
                    iter_k,
                    group_k,
                    group_n,
                    use_fp8_w8a8,
                    use_int8_w8a8,
                    use_int8_w8a16,
                )

        a_ptr += STEP_K * ak_stride
        b_ptr += STEP_K * bk_stride

    return accumulator


@triton.jit
def do_shrink_kernel_fp8(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    N,
    K,
    M_LEN,
    ram,
    # input strides
    input_d0_stride,
    input_d1_stride,
    # lora strides
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    # scale strides
    a_scale_m_stride,
    a_scale_k_stride,
    b_scale_l_stride,
    b_scale_n_stride,
    b_scale_k_stride,
    # output strides
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    # block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice, compute the
    matrix product and store in the appropriate output location.
    """

    # Identify the lora_ptr from slice_id.
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
        cur_b_scale_ptr = b_scale_ptr
    else:
        cur_lora_ptr = (
            tl.load(lora_ptr + slice_id).to(tl.pointer_type(tl.float8e4nv))
            if b_scale_ptr is not None
            else tl.load(lora_ptr + slice_id).to(
                tl.pointer_type(input_ptr.dtype.element_ty)
            )
        )
        cur_b_scale_ptr = (
            tl.load(b_scale_ptr + slice_id).to(tl.pointer_type(tl.float32))
            if b_scale_ptr is not None
            else b_scale_ptr
        )

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (
        input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    )
    b_ptr = (
        cur_lora_ptr
        + lora_d0_stride * lora_index
        + rbn[None, :] * lora_d1_stride
        + offset_k[:, None] * lora_d2_stride
    )

    # Load scales for tensor-wise or per-channel quantization (outside the loop)
    # Block-wise scales are loaded inside fp8_mm_k
    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # Block-wise: compute scale pointers for fp8_mm_k
            # a_scale: per-row base pointers, shape (BLOCK_M,)
            # Each pointer points to the start of that row's scale data
            mm_a_scale_ptr = a_scale_ptr + ram * a_scale_m_stride

            # b_scale: pre-compute N-dimension offset
            # We need to bake in the N-group offset since fp8_mm_k doesn't know pid_n
            n_offset = pid_n * BLOCK_N
            offs_ns = (n_offset + tl.arange(0, BLOCK_N)) // group_n
            # Base pointer with lora offset + N-group offset baked in, shape (BLOCK_N,)
            mm_b_scale_ptr = (
                cur_b_scale_ptr
                + lora_index * b_scale_l_stride
                + offs_ns * b_scale_n_stride
            )
        elif per_channel_quant:
            # Per-channel for weights, per-token for activations
            b_scale_ptrs = (
                cur_b_scale_ptr + lora_index * b_scale_l_stride + rbn * b_scale_n_stride
            )
            b_scale = tl.load(b_scale_ptrs)
            # Per-token activation scale
            a_scale = tl.load(a_scale_ptr + ram * a_scale_m_stride)[:, None]
            # For non-block-wise, pass original pointers (not used in mm loop)
            mm_a_scale_ptr = a_scale_ptr
            mm_b_scale_ptr = cur_b_scale_ptr
        else:
            # Tensor-wise quantization
            a_scale = tl.load(a_scale_ptr) if a_scale_ptr is not None else 1.0
            b_scale = tl.load(cur_b_scale_ptr + lora_index * b_scale_l_stride)
            # For non-block-wise, pass original pointers (not used in mm loop)
            mm_a_scale_ptr = a_scale_ptr
            mm_b_scale_ptr = cur_b_scale_ptr
    elif use_int8_w8a16:
        # INT8 weights with FP16 activations - only need weight scales
        b_scale_ptrs = (
            cur_b_scale_ptr + lora_index * b_scale_l_stride + rbn * b_scale_n_stride
        )
        b_scale = tl.load(b_scale_ptrs)
        mm_a_scale_ptr = a_scale_ptr
        mm_b_scale_ptr = cur_b_scale_ptr
    else:
        # Non-quantized path
        mm_a_scale_ptr = a_scale_ptr
        mm_b_scale_ptr = cur_b_scale_ptr

    # Compute partial/complete block matrix product.
    accumulator = fp8_mm_k(
        a_ptr,
        b_ptr,
        mm_a_scale_ptr,
        mm_b_scale_ptr,
        input_d1_stride,
        lora_d2_stride,
        a_scale_k_stride,
        b_scale_k_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        group_k,
        group_n,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        per_channel_quant,
        False,
        cur_lora_ptr.dtype.element_ty,
        USE_GDC,
        base_k=pid_sk * BLOCK_K,
    )
    # GDC launch dependents hints the runtime system to launch dependent kernels.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    # Apply dequantization scales for tensor-wise/per-channel quantization
    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # Block-wise: already applied in fp8_mm_k
            pass
        else:
            # Tensor-wise or per-channel: apply scales after accumulation
            accumulator = accumulator * a_scale * b_scale
    elif use_int8_w8a16:
        # INT8 weights with FP16 activations - only apply weight scale
        accumulator = accumulator * b_scale

    # Apply LoRA scaling factor
    accumulator *= scaling

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
    c_ptr = (
        cur_out_ptr
        + ram[:, None] * output_d1_stride
        + offset_cn[None, :] * output_d2_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    # Cast accumulator to output dtype
    accumulator = accumulator.to(out_ptr.dtype.element_ty)

    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def do_expand_kernel_fp8(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    N,
    K,
    M_LEN,
    ram,  # array identifying the rows of Input ptr to operate on
    slice_start_loc,
    # input ptr strides
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    # lora ptr strides
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    # scale strides
    a_scale_m_stride,
    a_scale_k_stride,
    b_scale_l_stride,
    b_scale_n_stride,
    b_scale_k_stride,
    # out ptr strides
    output_d0_stride,
    output_d1_stride,
    # block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_GDC: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
):
    """
    FP8-compatible expand kernel for LoRA.
    Given an array of integers that identifies the rows of A, ram,
    a lora index that identifies which LoRA to use from lora_ptr, lora_index,
    a slice_id that identifies the input/output slice,
    compute the matrix product with FP8 quantization support and store in
    the appropriate output location.

    For expand kernel, the input (shrink output) may be in FP32/FP16/BF16,
    while the LoRA B weights can be in FP8.

    Supports:
    - FP8 W8A8 quantization for LoRA B weights
    - Block-wise quantization with configurable group_k and group_n
    - Per-channel quantization
    - Tensor-wise quantization
    """

    # ls_d*_ptr can be either an integer or a pointer
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    # Identify the input_ptr and lora_ptr from slice_id.
    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        if use_fp8_w8a8:
            cur_lora_ptr = lora_ptr
            cur_b_scale_ptr = b_scale_ptr
        else:
            cur_lora_ptr = lora_ptr
            cur_b_scale_ptr = b_scale_ptr  # May be None for non-quantized
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        if use_fp8_w8a8:
            cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
                tl.pointer_type(tl.float8e4nv)
            )
            cur_b_scale_ptr = tl.load(b_scale_ptr + slice_id).to(
                tl.pointer_type(tl.float32)
            )
        else:
            cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
                tl.pointer_type(out_ptr.dtype.element_ty)
            )
            cur_b_scale_ptr = (
                tl.load(b_scale_ptr + slice_id).to(tl.pointer_type(tl.float32))
                if b_scale_ptr is not None
                else None
            )

    # Identify the column indices of B to process.
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    # Identify A and B block pointers
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (
        cur_input_ptr
        + ram[:, None] * input_d1_stride
        + offset_k[None, :] * input_d2_stride
    )
    b_ptr = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )

    # Setup scale pointers for FP8/INT8 quantization
    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            # Block-wise quantization - compute scale pointers for fp8_mm_k
            # a_scale: per-row base pointers, shape (BLOCK_M,)
            mm_a_scale_ptr = a_scale_ptr + ram * a_scale_m_stride

            # b_scale: pre-compute N-dimension offset since fp8_mm_k doesn't know pid_n
            n_offset = pid_n * BLOCK_N
            offs_ns = (n_offset + tl.arange(0, BLOCK_N)) // group_n
            # Base pointer with lora offset + N-group offset baked in, shape (BLOCK_N,)
            mm_b_scale_ptr = (
                cur_b_scale_ptr
                + lora_index * b_scale_l_stride
                + offs_ns * b_scale_n_stride
            )
        elif per_channel_quant:
            # Per-channel for weights, shape (BLOCK_N,)
            b_scale_ptrs = (
                cur_b_scale_ptr + lora_index * b_scale_l_stride + rbn * b_scale_n_stride
            )
            b_scale = tl.load(b_scale_ptrs)
            # Per-token activation scale, only if a_scale_ptr provided
            a_scale = tl.load(a_scale_ptr + ram * a_scale_m_stride)[:, None]
            # For non-block-wise, pass original pointers (not used in mm loop)
            mm_a_scale_ptr = a_scale_ptr
            mm_b_scale_ptr = cur_b_scale_ptr
        else:
            # Tensor-wise quantization
            a_scale = tl.load(a_scale_ptr) if a_scale_ptr is not None else 1.0
            b_scale = tl.load(cur_b_scale_ptr + lora_index * b_scale_l_stride)
            # For non-block-wise, pass original pointers (not used in mm loop)
            mm_a_scale_ptr = a_scale_ptr
            mm_b_scale_ptr = cur_b_scale_ptr
    elif use_int8_w8a16:
        # INT8 weights with FP16 activations - only need weight scales
        b_scale_ptrs = (
            cur_b_scale_ptr + lora_index * b_scale_l_stride + rbn * b_scale_n_stride
        )
        b_scale = tl.load(b_scale_ptrs)
        mm_a_scale_ptr = a_scale_ptr
        mm_b_scale_ptr = cur_b_scale_ptr
    else:
        # Non-quantized path
        mm_a_scale_ptr = a_scale_ptr
        mm_b_scale_ptr = cur_b_scale_ptr

    # Compute the block matrix product using fp8_mm_k
    # Note: For expand kernel, SPLIT_K=1, so we pass 1 for SPLIT_K
    accumulator = fp8_mm_k(
        a_ptr,
        b_ptr,
        mm_a_scale_ptr,
        mm_b_scale_ptr,
        input_d2_stride,  # ak_stride
        cur_lora_d2_stride,  # bk_stride
        a_scale_k_stride,
        b_scale_k_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,  # SPLIT_K = 1 for expand kernel
        group_k,
        group_n,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        per_channel_quant,
        CAST_TYPE,  # CAST_TYPE - cast FP8 B to A's dtype
        cur_lora_ptr.dtype.element_ty,
        USE_GDC,
        base_k=0,
    )

    # Apply dequantization scales for non-block-wise quantization
    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            pass  # Already applied per block in fp8_mm_k
        else:
            # Tensor-wise or per-channel: apply scales after accumulation
            accumulator = accumulator * a_scale * b_scale
    elif use_int8_w8a16:
        # INT8 weights with FP16 activations - only apply weight scale
        accumulator = accumulator * b_scale

    tiled_c = accumulator.to(out_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    # Identify the C output pointers to store the results of the accumulator.
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (
        out_ptr
        + ram[:, None] * output_d0_stride
        + offset_cn[None, :] * output_d1_stride
    )
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)
