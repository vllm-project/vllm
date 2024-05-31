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
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    """
    C=A@B, and B is col-major matrix
    """
    cur_batch = tl.program_id(axis=0)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    # tl.max_contiguous(offset_k, BLOCK_K)
    tiled_a = tl.load(
        input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
        mask=offset_k < K,
        other=0,
    )  # [BLOCK_K]
    b_ptr = lora_ptr + l0_stride * lora_index
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    # sliding  to  next row-block

    for n in range(0, N, BLOCK_N):
        current_n = n + offset_n
        # vector load
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < N) & (offset_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr
            + current_n_c[:, None] * lora_k_stride
            + offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )  # [BLOCK_N,BLOCK_K]

        accumulator = tl.sum(tiled_a * tiled_b, 1)

        c_ptr = (
            out_ptr
            + cur_batch * cm_stride
            + slice_offset  # slice size
            + current_n * cn_stride
        )
        c_mask = current_n < N
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

    N, K = lora_b_weights.shape[-2:]  # K= rank,N=hidden_size
    # TODO tuning this config
    BLOCK_N = 512
    BLOCK_K = triton.next_power_of_2(K)
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = [
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
        BLOCK_N,
        BLOCK_K,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return
