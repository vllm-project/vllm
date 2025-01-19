import triton
import triton.language as tl

@triton.jit
def mm_k(a_ptr,
         b_ptr,
         ak_stride,
         bk_stride,
         offset_k,
         K: tl.constexpr,
         BLOCK_M: tl.constexpr,
         BLOCK_N: tl.constexpr,
         BLOCK_K: tl.constexpr,
         EVEN_K: tl.constexpr,
         SPLIT_K: tl.constexpr,
         CAST_TYPE: tl.constexpr,
         b_dtype: tl.constexpr):

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K),
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(b_dtype)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * SPLIT_K * ak_stride 
        b_ptr += BLOCK_K * SPLIT_K * bk_stride 
    return accumulator


@triton.jit
def do_expand_kernel(pid_m,
                     pid_n,
                     lora_index,
                     slice_id,
                     input_ptr,
                     lora_ptr,
                     out_ptr,
                     N,
                     K,
                     M_LEN,
                     ram, # array identifying the rows of Input ptr to operate on
                     slice_start_loc,
                     # input ptr strides
                     input_d0_stride,
                     input_d1_stride,
                     input_d2_stride,
                     # lora ptr strides
                     ls_d0_ptr,
                     ls_d1_ptr,
                     ls_d2_ptr,
                     # out ptr strides
                     output_d0_stride,
                     output_d1_stride,
                     # constants
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr,
                     SAME_STRIDE: tl.constexpr,
                     SLICE_NUM: tl.constexpr,
                     EVEN_K: tl.constexpr,
                     CAST_TYPE: tl.constexpr,
                     ADD_INPUTS: tl.constexpr,
                     ):

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)

    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N),
                            BLOCK_N)

    # ls_d*_ptr can be either an integer or a pointer
    if SAME_STRIDE:
        # integer
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        # pointer
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)
    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr

    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(out_ptr.dtype.element_ty))

    a_ptr = (cur_input_ptr +
             ram[:, None] * input_d1_stride +
             offset_k[None, :] * input_d2_stride, )
    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index +
             offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)

    SPLIT_K = 1
    accumulator = mm_k(a_ptr, b_ptr, input_d2_stride, cur_lora_d2_stride, offset_k, K,
                        BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
                        CAST_TYPE, cur_lora_ptr.dtype.element_ty)

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (out_ptr + ram[:, None] * output_d0_stride +
             offset_cn[None, :] * output_d1_stride)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)

@triton.jit
def do_shrink_kernel(pid_m,
                      pid_n,
                      pid_sk,
                      slice_id,
                      lora_index,
                      input_ptr,
                      lora_ptr,
                      out_ptr,
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
                      # output strides
                      output_d0_stride,
                      output_d1_stride,
                      output_d2_stride,
                      scaling,
                      BLOCK_M : tl.constexpr,
                      BLOCK_N : tl.constexpr,
                      BLOCK_K : tl.constexpr,
                      EVEN_K : tl.constexpr,
                      SPLIT_K : tl.constexpr,
                      SLICE_NUM: tl.constexpr,):

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)

    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    # input ptr
    a_ptr = (input_ptr +
             ram[:, None] * input_d0_stride +
             offset_k[None, :] * input_d1_stride)

    if SLICE_NUM == 1:
        # current lora ptr
        cur_lora_ptr = lora_ptr
    else:
        # current lora ptr
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(
            tl.pointer_type(input_ptr.dtype.element_ty))

    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index +
             rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    accumulator = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d2_stride, offset_k, K,
                        BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K,
                        False, cur_lora_ptr.dtype.element_ty)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr +
                   slice_id * output_d0_stride)
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[
        None, :] * output_d2_stride

    offset_cm = tl.arange(0, BLOCK_M)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    accumulator *= scaling
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)