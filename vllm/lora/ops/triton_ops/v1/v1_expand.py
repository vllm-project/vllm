# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import List
from itertools import product

import torch
import triton
import triton.language as tl

from vllm.lora.ops.triton_ops.kernel_utils import do_expand_kernel
from vllm.lora.ops.triton_ops.utils import _get_lora_b_ptr
from vllm.utils import direct_register_custom_op

def block_m_ranges():
    return [16, 32, 64, 128, 256, 512]


def block_k_ranges():
    return [16]


def block_n_ranges():
    return [32, 64, 128, 256, 512, 1024]


def warp_ranges():
    return [4, 8]


def cta_ranges():
    return [1]


def num_stages():
    return [2, 4]


def autotune_configs():
    return [
        triton.Config(kwargs={
            'BLOCK_M': bm,
            'BLOCK_N': bn,
            'BLOCK_K': bk
        },
                      num_warps=nw,
                      num_ctas=nc,
                      num_stages=ns)
        for bm, bn, bk, nw, nc, ns in product(block_m_ranges(), block_n_ranges(
        ), block_k_ranges(), warp_ranges(), cta_ranges(), num_stages())
    ]


def prune_fn(*args, **kwargs):
    configs_list, kernel_kwargs = args

    # prune such that EVEN_K is true

    def is_m_good(config, kkwargs):
        return config.kwargs['BLOCK_M'] == 16 or config.kwargs[
            'BLOCK_M'] <= kkwargs['M']

    pruned = filter(lambda x: is_m_good(x, kernel_kwargs), configs_list)
    pruned = list(pruned)
    print(f"Trying #configs {len(pruned)}")
    return pruned


@triton.autotune(configs=autotune_configs(),
                 key=['M', 'N', 'K', 'SLICE_NUM', 'ADD_INPUTS', 'MAX_LORAS'],
                 prune_configs_by={'early_config_prune': prune_fn})
@triton.jit
def _v1_expand_kernel(
        input_ptr,
        lora_ptr,
        out_ptr,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
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
        EVEN_K: tl.constexpr,
        ADD_INPUTS: tl.constexpr,
        CAST_TYPE: tl.constexpr,
        SLICE_NUM: tl.constexpr,
        SAME_STRIDE: tl.constexpr,
        MAX_LORAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # Early exit for the no-lora case.
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # Early exit CTA.
        return

    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        # Early exit CTA.
        return

    # num rows this CTA should process.
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    # Identify all rows that this CTA should process.
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)

    # Load all relevant row indices.
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        curr_N,
        K,
        cta_m_len,
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
        # out ptr strides
        output_d0_stride,
        output_d1_stride,
        # constants
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS)

@torch.inference_mode()
def _v1_expand(
    inputs: torch.Tensor,  # shape [num_slices, num_tokens, lora_rank]
    lora_b_weights: List[
        torch.Tensor],  # shape [num_lora, hidden_size, lora_rank]
    output_tensor: torch.
    Tensor,  # shape [num_tokens, hidden_size * num_slices]
    token_lora_mapping: torch.Tensor,  # shape [num_tokens]
    token_indices_sorted_by_lora_ids: torch.Tensor,  # shape [num_tokens]
    num_tokens_per_lora: torch.Tensor,  # shape [max-loras + 1]
    lora_token_start_loc: torch.Tensor,  # shape [max-loras + 2]
    lora_ids: torch.Tensor,  # shape [max-loras + 1]
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (List[torch.Tensor]): lora'b weight
        output_tensor (torch.Tensor): output tensor
        token_lora_mapping (torch.Tensor): A tensor mapping each input token
            to the lora-id related to that token. A value of -1 indicates that
            LoRA doesn't apply to that token.
        token_indices_sorted_by_lora_ids (torch.Tensor): Row/Token indices from
            the A matrix grouped by LoRA IDs.
        num_tokens_per_lora (torch.Tensor): num_tokens_per_lora[i] is the number
            of tokens that are to be processed by LoRA ID lora_ids[i] 
        lora_token_start_loc (torch.Tensor): A cumulative sum of
            num_tokens_per_lora. lora_token_start_loc[0] is always 0 so that
            lora_token_start_loc[i], along with num_tokens_per_lora[i]
            identifies the the region in token_indices_sorted_by_lora_ids that
            LoRA lora_ids[i] should process.
        lora_ids (torch.Tensor): LoRA ids to process.
        offset_start (int, optional): Offset start for output_tensor. 
            Defaults to 0.
        add_inputs (bool, optional): Whether to add the input tensor to the 
            output tensor. Defaults to False.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()

    # metadata sanity check.
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(
        0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor,
     lora_strides_d1_tensor, lora_strides_d2_tensor, hidden_sizes_tensor,
     same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start,
                                           inputs.device)

    K = lora_b_weights[0].shape[-1]  # K= rank
    M = inputs.size(1)
    ADD_INPUTS = add_inputs
    MAX_LORAS = lora_ids.size(0)
    CAST_TYPE = False
    NUM_SLICES = len(lora_b_weights)

    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(
        MAX_N, meta['BLOCK_N']), NUM_SLICES, MAX_LORAS)

    _v1_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        MAX_N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
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
        True, # EVEN_K, i.e. multiples of 16
        ADD_INPUTS,
        CAST_TYPE,
        NUM_SLICES,
        same_stride,
        MAX_LORAS,
    )

    return


def _v1_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="v1_expand",
        op_func=_v1_expand,
        mutates_args=["output_tensor"],
        fake_impl=_v1_expand_fake,
    )
    v1_expand = torch.ops.vllm.v1_expand

except AttributeError:
    v1_expand = _v1_expand
