# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, per_token_group_quant_fp8)

logger = init_logger(__name__)

has_deep_gemm = importlib.util.find_spec("deep_gemm") is not None


class BatchedDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE = 128

    def __init__(self, max_num_tokens: int, world_size: int, dp_size: int,
                 block_shape: list[int]):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        world_size: Number of EP ranks
        dp_size: Number of data-parallel ranks
        block_shape: Block quantization block shape
        """
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.block_shape = block_shape

        assert (len(self.block_shape) == 2 and all(
            [v == self.DEEPGEMM_BLOCK_SHAPE for v in self.block_shape]))

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        assert a.dim() == 2
        num_dp = self.world_size // self.dp_size
        max_num_tokens = a.size(
            0) if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = num_experts * max_num_tokens * num_dp * max(K, N)
        workspace2 = num_experts * max_num_tokens * num_dp * (N // 2)
        return (workspace13, workspace2, a.dtype)

    def apply(
        self,
        hidden_states: torch.Tensor,
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
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        import deep_gemm as dg
        assert hidden_states.ndim == 3

        a1q = hidden_states
        _, N, K = w1.size()

        if global_num_experts == -1:
            global_num_experts = w1.size(0)

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        workspace2 = _resize_cache(workspace2, (E, max_num_tokens, N // 2))
        workspace3 = _resize_cache(workspace13, (E, max_num_tokens, K))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a1q, a1q_scale),
                                                 (w1, w1_scale),
                                                 out=workspace1,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)

        # TODO (varun) [Optimization]: Use a batched version of activation.
        # Similarly for the quant below.
        self.activation(activation, workspace2, workspace1.view(-1, N))

        w2_hidden_size = workspace2.size(-1)
        workspace2 = workspace2.view(-1, w2_hidden_size)

        a2q_scale: Optional[torch.Tensor] = None
        a2q, a2q_scale = per_token_group_quant_fp8(workspace2,
                                                   self.block_shape[1],
                                                   column_major_scales=False)
        a2q = a2q.view(E, max_num_tokens, -1)
        a2q_scale = a2q_scale.view(E, max_num_tokens, -1)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a2q, a2q_scale),
                                                 (w2, w2_scale),
                                                 out=workspace3,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)

        return workspace3
