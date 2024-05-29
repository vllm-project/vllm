"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr, 
    lora_ptr, 
    out_ptr,  
    N,
    K,
    lora_indices,  
    xm_stride, 
    xk_stride, 
    l0_stride,  
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    cur_batch = tl.program_id(axis=2)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    a_ptr = input_ptr + cur_batch * xm_stride + offset_k[None, :] * xk_stride
    b_ptr = (
        lora_ptr
        + l0_stride * lora_index
        + rbn[None, :] * lora_k_stride
        + offset_k[:, None] * lora_n_stride
    )
    accumulator = tl.zeros((1, BLOCK_N), dtype=lora_ptr.dtype.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            k_remaining = K - k * BLOCK_K
            tiled_a = tl.load(a_ptr, mask=offset_k[None, :] < k_remaining, other=0.0)
            tiled_b = tl.load(b_ptr, mask=offset_k[:, None] < k_remaining, other=0.0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator += tl.sum(tiled_a[None, :] * tiled_b, 1)
        a_ptr += BLOCK_K * xk_stride
        b_ptr += BLOCK_K * lora_n_stride

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N+slice_offset
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn[None, :] * cn_stride
    c_mask = offset_cn[None, :] < (slice_offset+N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        accumulator += tiled_out
    tl.store(c_ptr, accumulator, mask=c_mask)


@torch.inference_mode()
def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batchs: int,
    max_seq_length: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    """_summary_

    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor
        b_seq_start_loc (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g.,if the sequence length is [4, 6], it is
            [0, 4, 10].
        seq_len_tensor (torch.Tensor): (batch_size,). record the sequence
            length of the sequences  in the batch
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch
        batchs (int): batch size
        max_seq_length (int):  The max sequence lengths of the sequences
            in the batch
        slice_offst (int): output_tensor's offst
        slice_size (int): current output_tensor's size
        add_inputs (bool, optional): _description_. Defaults to False.
    """

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert lora_indices_tensor.size(0) == batchs
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    # TODO tuning this config
    N, K = lora_b_weights.shape[-2:]  # K= rank,N=hidden_size

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = [
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        batchs,
    ]
    _bgmv_expand_slice_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N,
        K,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        slice_offset,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return
