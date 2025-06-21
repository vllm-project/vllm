# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.masked_kernels import (
    masked_per_token_group_quant_fp8)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache

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

    def supports_chunking(self) -> bool:
        return False

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
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        assert a.dim() == 2
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        num_dispatchers = self.world_size
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
    ):
        import deep_gemm as dg
        assert hidden_states.ndim == 3

        a1q = hidden_states
        _, N, K = w1.size()

        assert w2.size(1) == K

        E, max_num_tokens, N, K, top_k_num = mk._moe_problem_size(
            hidden_states, w1, w2, topk_ids)

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))
        workspace2 = _resize_cache(workspace2, (E, max_num_tokens, N // 2))

        # (from deepgemm docs) : A value hint (which is a value on CPU)
        # for the M expectation of each batch, correctly setting this value
        # may lead to better performance.
        expected_m = max_num_tokens

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a1q, a1q_scale),
                                                 (w1, w1_scale),
                                                 out=workspace1,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)

        self.masked_activation(activation, workspace2, workspace1,
                               expert_num_tokens)

        # TODO (varun) : Pass in an output tensor derived from workspace
        # as a memory optimization.
        a2q, a2q_scale = masked_per_token_group_quant_fp8(
            x=workspace2,
            valid_tokens_array=expert_num_tokens,
            group_size=self.block_shape[1],
            column_major_scales=False)

        dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked((a2q, a2q_scale),
                                                 (w2, w2_scale),
                                                 out=output,
                                                 masked_m=expert_num_tokens,
                                                 expected_m=expected_m)
