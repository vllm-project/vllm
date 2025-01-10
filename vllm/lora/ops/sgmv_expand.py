"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import List

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op

from .utils import _get_lora_b_ptr


@triton.jit
def _sgmv_expand_kernel(
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        b_seq_start_loc,
        seq_lens,
        lora_indices,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,  # 1
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,  # 1
        output_d0_stride,
        output_d1_stride,  # 1
        output_hs_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ADD_INPUTS: tl.constexpr,
        CAST_TYPE: tl.constexpr,
        SLICE_NUM: tl.constexpr,
        SAME_STRIDE: tl.constexpr):
    """

    Similar to the 'sgmv_expand' operator, but with an added parameter
    'slice_offset'. The reason for not reusing the 'sgmv_expand' operator
    might be that in the future, we could implement a fusion operator to
    achieve the current functionality instead of having to call it multiple
    times.
    """
    pid = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    slice_id = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    if pid_n * BLOCK_N > curr_N:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % curr_N, BLOCK_N),
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

    a_ptr = (cur_input_ptr + cur_seq_start * input_d1_stride +
             ram[:, None] * input_d1_stride +
             offset_k[None, :] * input_d2_stride, )
    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index +
             offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < K - k * BLOCK_K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < K - k * BLOCK_K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(cur_lora_ptr.dtype.element_ty)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * input_d2_stride
        b_ptr += BLOCK_K * cur_lora_d2_stride

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    c_ptr = (out_ptr + offset_cm[:, None] * output_d0_stride +
             offset_cn[None, :] * output_d1_stride)
    M = tl.load(seq_lens + cur_batch)
    c_mask = (offset_cm[:, None] <
              (cur_seq_start + M)) & (offset_cn[None, :] <
                                      (cur_slice_start + curr_N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@torch.inference_mode()
def _sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (List[torch.Tensor]): lora'b weight
        output_tensor (torch.Tensor): output tensor
        b_seq_start_loc (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g., if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): (batch_size,). Record the sequence
            length of the sequences in the batch.
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences in the 
            batch.
        token_nums (int): The token numbers in the batch. Used to verify if the 
            token numbers in the inputs matches the one in the metadata.
        offset_start (int, optional): Offset start for output_tensor. 
            Defaults to 0.
        add_inputs (bool, optional): Whether to add the input tensor to the 
            output tensor. Defaults to False.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == token_nums
    assert inputs.size(0) == len(lora_b_weights)

    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert output_tensor.is_contiguous()
    (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor,
     lora_strides_d1_tensor, lora_strides_d2_tensor, hidden_sizes_tensor,
     same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start,
                                           b_seq_start_loc.device)

    # TODO tuning this config
    K = lora_b_weights[0].shape[-1]  # K= rank

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False

    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        batches,
        len(lora_b_weights),
    )
    _sgmv_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        MAX_N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        len(lora_b_weights),
        same_stride,
    )
    return


def _sgmv_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="sgmv_expand",
        op_func=_sgmv_expand,
        mutates_args=["output_tensor"],
        fake_impl=_sgmv_expand_fake,
    )
    sgmv_expand = torch.ops.vllm.sgmv_expand

except AttributeError:
    sgmv_expand = _sgmv_expand
