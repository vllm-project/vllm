# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import log2
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deep_gemm_block_shape)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.utils.deep_gemm import (fp8_m_grouped_gemm_nt_masked,
                                  is_deep_gemm_e8m0_used)

logger = init_logger(__name__)


def silu_v1(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device,
                                             dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided((E, T, G),
                              (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32,
                              device=y.device)

    use_ue8m0 = is_deep_gemm_e8m0_used()

    if E <= 16:
        max_empirical_parallelism = 64
    elif E <= 32:
        max_empirical_parallelism = 16
    else:
        max_empirical_parallelism = 4

    # We never want to launch more than Tx number of threads
    # This computes the clip.
    num_parallel_tokens = max(
        1,
        min(max_empirical_parallelism, 2**int(log2(min(num_parallel_tokens,
                                                       T)))))

    torch.ops._C.silu_v1_cuda(y, tokens_per_expert, y_q, y_s, group_size,
                              use_ue8m0, num_parallel_tokens)

    return y_q, y_s


def silu_mul_fp8_quant_deep_gemm_cuda(
        y: torch.Tensor,  # (E, T, 2*H)
        tokens_per_expert: torch.
    Tensor,  # (E,) number of valid tokens per expert
        num_parallel_tokens=16,
        group_size: int = 128,
        expert_offsets: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device,
                                             dtype=torch.int32)

    fp8_dtype = torch.float8_e4m3fn
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    stride_ys_e = T * G
    stride_ys_t = 1
    stride_ys_g = T
    y_s = torch.empty_strided((E, T, G),
                              (stride_ys_e, stride_ys_t, stride_ys_g),
                              dtype=torch.float32,
                              device=y.device)

    use_ue8m0 = is_deep_gemm_e8m0_used()

    torch.ops._C.silu_mul_fp8_quant_deep_gemm_cuda(y, tokens_per_expert, y_q,
                                                   y_s, group_size, use_ue8m0,
                                                   num_parallel_tokens)

    return y_q, y_s


class BatchedDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
    ):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        num_dispatchers: The number of DP dispatchers.
        quant_config: Quantization configuration
        """
        super().__init__(quant_config)
        assert self.block_shape == deep_gemm_block_shape()
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.BatchedExperts,
                mk.FusedMoEActivationFormat.BatchedExperts)

    def supports_chunking(self) -> bool:
        return False

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_metadata: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = (num_experts, max_num_tokens * num_dispatchers,
                       max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, (N // 2))
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.block_shape is not None

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens
        fp8_m_grouped_gemm_nt_masked((a1q, a1q_scale), (w1, self.w1_scale),
                                     workspace1, expert_num_tokens, expected_m)

        a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_cuda(
            workspace1, expert_num_tokens)

        fp8_m_grouped_gemm_nt_masked((a2q, a2q_scale), (w2, self.w2_scale),
                                     output, expert_num_tokens, expected_m)
