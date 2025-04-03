# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import Optional, Tuple

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (_fp8_quantize,
                                                        _resize_cache)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    _moe_permute,
    _moe_unpermute_and_reduce
)
from vllm.model_executor.layers.fused_moe.dispatch_combine import (
    StandardDispatchCombine
)
from vllm.utils import round_up

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


def deep_gemm_block_shape() -> list[int]:
    # Lazy import to avoid CUDA initialization problems.
    import deep_gemm as dg
    block = dg.get_m_alignment_for_contiguous_layout()
    return [block, block]


def _valid_deep_gemm_shape(M: int, N: int, K: int):
    align = deep_gemm_block_shape()[0]
    return M >= align and N % align == 0 and K % align == 0


# TODO: check types?
def _valid_deep_gemm(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     expert_map: Optional[torch.Tensor] = None) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm:
        return False

    # Expert maps not supported yet.
    if expert_map is not None:
        return False

    M = hidden_states.shape[0]
    _, K, N = w2.shape
    if not _valid_deep_gemm_shape(M, N, K):
        return False

    return (hidden_states.is_contiguous() and w1.is_contiguous()
            and w2.is_contiguous())


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
    workspace13 = torch.empty(M_sum * max(N, K),
                              device=hidden_states.device,
                              dtype=hidden_states.dtype)

    workspace1 = workspace13[:M_sum * N].view(M_sum, N)
    workspace2 = torch.empty((M_sum, N // 2),
                             device=hidden_states.device,
                             dtype=hidden_states.dtype)
    workspace3 = workspace13[:M_sum * K].view(M_sum, K)

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
            workspace1 = _resize_cache(workspace1, (curr_M, N))
            workspace2 = _resize_cache(workspace2, (curr_M, N // 2))
            workspace3 = _resize_cache(workspace3, (curr_M, K))

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qcurr_hidden_states, a1q_scale), (w1, w1_scale), workspace1,
            expert_ids)

        if activation == "silu":
            torch.ops._C.silu_and_mul(workspace2, workspace1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(workspace2, workspace1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        qworkspace2, a2q_scale = _fp8_quantize(workspace2, a2_scale,
                                               block_shape)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qworkspace2, a2q_scale), (w2, w2_scale), workspace3, expert_ids)

        _moe_unpermute_and_reduce(
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
            workspace3.view(*workspace3.shape), inv_perm, curr_topk_weights)

    return out_hidden_states


class DeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self):
        super().__init__()
        self.block_shape = deep_gemm_block_shape()

    def workspace_shapes(
        self,
        a_dtype: torch.dtype,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int
    ) -> Tuple[int, int, torch.dtype]:
        block_m = self.block_shape[0]
        M_sum = (M * topk) + num_experts * (block_m - 1)
        M_sum = round_up(M_sum, block_m)
        workspace1 = M_sum * max(N * 2, K)
        workspace2 = M_sum * N
        return (workspace1, workspace2, a_dtype)

    def apply(
        self,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> torch.Tensor:
        import deep_gemm as dg

        # TODO: chunking in here or in FusedMoEModularKernel? ignore for now
        _, N, K = w1.shape

        assert global_num_experts != -1
        assert w2.shape[1] == K

        a1q, a1q_scale, _, expert_ids, inv_perm = _moe_permute(
            a1q,
            a1q_scale,
            topk_ids,
            global_num_experts,
            expert_map,
            self.block_shape[0],
        )

        # Note: M_sum is different than the pre-permuted shape of a1q.
        M_sum = a1q.shape[0]
        workspace1 = _resize_cache(workspace13, (M_sum, N))
        workspace2 = _resize_cache(workspace2, (M_sum, N // 2))
        workspace3 = _resize_cache(workspace13, (M_sum, K))

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (a1q, a1q_scale), (w1, w1_scale), workspace1, expert_ids)

        if activation == "silu":
            torch.ops._C.silu_and_mul(workspace2, workspace1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(workspace2, workspace1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        a2q_scale: Optional[torch.Tensor] = None

        a2q, a2q_scale = _fp8_quantize(workspace2, a2_scale, self.block_shape)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (a2q, a2q_scale), (w2, w2_scale), workspace3, expert_ids)

        workspace3 = workspace3[inv_perm, ...]

        return workspace3


def modular_deep_gemm_fused_moe_fp8() -> mk.FusedMoEModularKernel:
    return mk.FusedMoEModularKernel(
        StandardDispatchCombine(quant_dtype=torch.float8_e4m3fn,
                                block_shape=deep_gemm_block_shape()),
        DeepGemmExperts(),
    )
