# SPDX-License-Identifier: Apache-2.0
import importlib.util
from math import prod
from typing import Optional, Tuple

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.utils import (_fp8_perm,
                                                        _fp8_quantize)
from vllm.utils import round_up

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


def _valid_deep_gemm(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor,
                     expert_map: Optional[torch.Tensor]) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm:
        return False

    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg

    # Expert maps not supported yet.
    if expert_map is not None:
        return False

    align = dg.get_m_alignment_for_contiguous_layout()
    M = hidden_states.shape[0]
    _, K, N = w2.shape

    # For now, disable DeepGemm for small N until better permute/unpermute
    # ops are available.
    if N <= 512:
        return False

    if align > M or N % align != 0 or K % align != 0:
        return False

    return (hidden_states.is_contiguous() and w1.is_contiguous()
            and w2.is_contiguous())


def _moe_permute(
    curr_hidden_states: torch.Tensor,
    a1q_scale: Optional[torch.Tensor],
    curr_topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    block_m: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Determine the sorted_token_ids, expert_ids for the given problem size.
    Permute the hidden states and scales according to `sorted_token_ids`.
    """
    top_k_num = curr_topk_ids.shape[1]

    tokens_in_chunk, _ = curr_hidden_states.shape

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(curr_topk_ids,
                             block_m,
                             global_num_experts,
                             expert_map,
                             pad_sorted_ids=True))

    inv_perm: Optional[torch.Tensor] = None

    num_tokens = top_k_num * tokens_in_chunk
    sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
    expert_ids = torch.repeat_interleave(expert_ids, block_m, dim=0)
    inv_perm = torch.argsort(sorted_token_ids)[:num_tokens]

    # Permute according to sorted token ids.
    curr_hidden_states = _fp8_perm(curr_hidden_states,
                                   sorted_token_ids // top_k_num)

    if a1q_scale is not None:
        a1q_scale = a1q_scale[sorted_token_ids // top_k_num]

    return (curr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
            inv_perm)


def _moe_unpermute_and_reduce(
    out: torch.Tensor,
    curr_hidden: torch.Tensor,
    inv_perm: Optional[torch.Tensor],
    topk_weight: torch.Tensor,
) -> None:
    """
    Unpermute the final result and apply topk_weights, then perform the final
    reduction on the hidden states.
    """
    M, topk = topk_weight.shape
    K = curr_hidden.shape[1]
    curr_hidden = curr_hidden[inv_perm, ...]
    curr_hidden = curr_hidden.view(-1, topk, K)
    curr_hidden.mul_(topk_weight.view(M, -1, 1))
    ops.moe_sum(curr_hidden, out)


def _resize_cache(x: torch.Tensor, v: Tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel()
    return x.flatten()[:prod(v)].view(*v)


def deep_gemm_moe_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with DeepGemm
    grouped gemm.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1 (torch.Tensor): The first set of fp8 quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2 (torch.Tensor): The second set of fp8 quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - topk_ids (torch.Tensor): The token->expert mapping for topk_weights.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - activation (str): The activation function to apply after the first
        MoE layer.
    - global_num_experts (int): The total number of experts in the global
        expert space.
    - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
        from the global expert space to the local expert space of the expert
        parallel shard.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]

    Returns:
    - torch.Tensor: The bfloat16 output tensor after applying the MoE layer.
    """
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg

    assert expert_map is None, "Expert maps not supported yet"

    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    assert w1.dtype == torch.float8_e4m3fn
    assert w2.dtype == torch.float8_e4m3fn
    assert w1.shape[0] == w2.shape[0], "Expert number mismatch"
    assert w1.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
    assert w1.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[
        0] == hidden_states.shape[0], "Input scale shape mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE

    assert _valid_deep_gemm(hidden_states, w1, w2, expert_map)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    block_m = dg.get_m_alignment_for_contiguous_layout()
    block_shape = [block_m, block_m]

    assert w1_scale is not None
    assert w2_scale is not None

    # We attempt to transpose and align offline in Fp8MoEMethod, in which
    # case these calls will be nops.  Otherwise, they'll be performed every
    # time the layer is executed.
    w1_scale = dg.get_col_major_tma_aligned_tensor(w1_scale).contiguous()
    w2_scale = dg.get_col_major_tma_aligned_tensor(w2_scale).contiguous()

    M_sum = topk_ids.numel() + global_num_experts * (block_m - 1)
    M_sum = round_up(M_sum, block_m)

    num_chunks = (num_tokens // CHUNK_SIZE) + 1

    # We can reuse the memory between cache1 and cache3 because by the time
    # we need cache3, we're done with cache1
    cache13 = torch.empty(M_sum * max(N, K),
                          device=hidden_states.device,
                          dtype=hidden_states.dtype)

    intermediate_cache1 = cache13[:M_sum * N].view(M_sum, N)
    intermediate_cache2 = torch.empty((M_sum, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = cache13[:M_sum * K].view(M_sum, K)

    for chunk in range(num_chunks):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        a1q_scale: Optional[torch.Tensor] = None

        qcurr_hidden_states, a1q_scale = _fp8_quantize(curr_hidden_states,
                                                       a1_scale, block_shape)

        (qcurr_hidden_states, a1q_scale, sorted_token_ids, expert_ids,
         inv_perm) = _moe_permute(qcurr_hidden_states, a1q_scale,
                                  curr_topk_ids, global_num_experts,
                                  expert_map, block_m)

        # Adjust the intermediate cache size and config for the last chunk.
        # Note that in most cases we only have one chunk so the cache size
        # and config are already set correctly and do not need to be adjusted.
        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            curr_M = sorted_token_ids.numel()
            intermediate_cache1 = _resize_cache(intermediate_cache1,
                                                (curr_M, N))
            intermediate_cache2 = _resize_cache(intermediate_cache2,
                                                (curr_M, N // 2))
            intermediate_cache3 = _resize_cache(intermediate_cache3,
                                                (curr_M, K))

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qcurr_hidden_states, a1q_scale), (w1, w1_scale),
            intermediate_cache1, expert_ids)

        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        qintermediate_cache2, a2q_scale = _fp8_quantize(
            intermediate_cache2, a2_scale, block_shape)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qintermediate_cache2, a2q_scale), (w2, w2_scale),
            intermediate_cache3, expert_ids)

        _moe_unpermute_and_reduce(
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
            intermediate_cache3.view(*intermediate_cache3.shape), inv_perm,
            curr_topk_weights)

    return out_hidden_states


#TODO make the grouped gemm kernel consistent with scaled gemm kernel
def cutlass_moe_fp8(
    a: torch.Tensor,
    w1_q: torch.Tensor,
    w2_q: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    ab_strides1: torch.Tensor,
    c_strides1: torch.Tensor,
    ab_strides2: torch.Tensor,
    c_strides2: torch.Tensor,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    """
    This function computes a a8w8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of fp8-quantized expert weights.
        Shape: [num_experts, K, 2N] (the weights are passed transposed)
    - w2_q (torch.Tensor): The second set of fp8-quantized expert weights.
        Shape: [num_experts, N, K] (the weights are passed transposed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts] or [num_experts, 2N]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts] or [num_experts, K]
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [M]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [M]
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp16 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.float8_e4m3fn
    assert w2_q.dtype == torch.float8_e4m3fn
    assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
    assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert a1_scale is None or a1_scale.dim(
    ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
        0], "Input scale shape mismatch"
    assert w1_scale.dim() == 1 or w1_scale.shape[1] == 1 or w1_scale.shape[
        1] == w1_q.shape[2], "W1 scale shape mismatch"
    assert w2_scale.dim() == 1 or w2_scale.shape[1] == 1 or w2_scale.shape[
        1] == w2_q.shape[2], "W2 scale shape mismatch"
    assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert ab_strides1.shape[0] == w1_q.shape[
        0], "AB Strides 1 expert number mismatch"
    assert c_strides1.shape[0] == w1_q.shape[
        0], "C Strides 1 expert number mismatch"
    assert ab_strides2.shape[0] == w2_q.shape[
        0], "AB Strides 2 expert number  mismatch"
    assert c_strides2.shape[0] == w2_q.shape[
        0], "C Strides 2 expert number mismatch"
    assert out_dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(1)
    n = w2_q.size(1)

    topk = topk_ids.size(1)
    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)

    a_q, a1_scale = ops.scaled_fp8_quant(
        a, a1_scale, use_per_token_if_dynamic=per_act_token)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
    c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

    ops.cutlass_moe_mm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                       expert_offsets[:-1], problem_sizes1, ab_strides1,
                       ab_strides1, c_strides1)

    intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale, use_per_token_if_dynamic=per_act_token)

    ops.cutlass_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                       expert_offsets[:-1], problem_sizes2, ab_strides2,
                       ab_strides2, c_strides2)

    return (c2[c_map].view(m, topk, k) *
            topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)
