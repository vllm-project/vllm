# SPDX-License-Identifier: Apache-2.0
"""Fused batched MoE kernel."""
from typing import List, Optional, Tuple

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import _resize_cache


class BatchedDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):
    def __init__(self,
                 world_size: int,
                 rank: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert topk_ids.dim() == 2
        assert topk_ids.shape[0] == a1.shape[0]

        if apply_router_weight_on_input:
            topk = topk_ids.shape[1]
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))

        num_tokens = a1.shape[0]
        topk = topk_ids.shape[1]

        tokens_per_expert = torch.bincount(topk_ids.view(-1), minlength=num_experts)
        max_num_tokens = tokens_per_expert.max()
        expert_counts = torch.zeros(num_experts, dtype=torch.int, device=a1.device)

        b_a1 = torch.zeros((num_experts, max_num_tokens, a1.shape[1]),
                           dtype=a1.dtype, device=a1.device)

        for token in range(num_tokens):
            for j in range(topk):
                expert_id = topk_ids[token, j]
                idx = expert_counts[expert_id]
                b_a1[expert_id, idx:idx+1, :] = a1[token, :]
                expert_counts[expert_id] = expert_counts[expert_id] + 1

        return b_a1, a1_scale, tokens_per_expert

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        num_tokens = topk_ids.shape[0]
        num_experts = fused_expert_output.shape[0]
        expert_counts = torch.zeros(num_experts, dtype=torch.int, device=fused_expert_output.device)
        for token in range(num_tokens):
            expert_ids = topk_ids[token]
            for i in range(topk_ids.shape[1]):
                expert_id = expert_ids[i]
                if expert_id < num_experts:
                    idx = expert_counts[expert_id]
                    if apply_router_weight_on_input:
                        output[token, :] = output[token, :] + fused_expert_output[expert_id, idx:idx+1, :]
                    else:
                        output[token, :] = output[token, :] + fused_expert_output[expert_id, idx:idx+1, :] * topk_weights[token, i]
                    expert_counts[expert_id] = expert_counts[expert_id] + 1


def rank_chunk(num, r, w):
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


class BatchedExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        max_num_tokens: Optional[int] = None,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape: Optional[List[int]] = None,
        block_m: Optional[int] = None,
    ):
        super().__init__()
        assert not use_fp8_w8a8
        assert not use_int4_w4a16
        assert not use_int8_w8a16
        assert block_shape is None
        assert block_m is None
        self.max_num_tokens = max_num_tokens
        self.rank = rank
        self.world_size = world_size
        assert not use_fp8_w8a8, "NYI"
        assert not use_int8_w8a8, "NYI"
        assert not use_int8_w8a16, "NYI"
        assert not use_int4_w4a16, "NYI"

    def workspace_shapes(
        self,
        a: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> Tuple[int, int, torch.dtype]:
        max_num_tokens = a.shape[1] if self.max_num_tokens is None else self.max_num_tokens
        workspace13 = num_experts * max_num_tokens * K * topk * 2 # TODO: *2 is a hack
        workspace2 = max_num_tokens * N
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
        assert hidden_states.dim() == 3
        assert expert_num_tokens is not None
        num_tokens = topk_ids.shape[0]
        _, tmp_max_num_tokens, K = hidden_states.shape
        max_num_tokens = tmp_max_num_tokens if self.max_num_tokens is None else self.max_num_tokens
        num_experts = global_num_experts
        out = _resize_cache(workspace13, (num_experts, max_num_tokens, w2.shape[1]))
        num_local_experts = expert_num_tokens.numel()

        # TODO: don't need world_size or rank if expert_base always == 0
        #assert w1.shape[0] == num_experts, f"{w1.shape} == {num_experts}"
        #expert_base = rank_chunk(w1.shape[0], self.rank, self.world_size) * self.rank
        expert_base = 0

        for expert in range(num_local_experts):
            num = expert_num_tokens[expert]
            assert num <= max_num_tokens, f"{num}, {max_num_tokens}"
            if num > 0:
                tmp = _resize_cache(workspace2, (num, w1.shape[1] // 2))
                self.activation(
                    activation,
                    tmp,
                    hidden_states[expert,:num,:] @ w1[expert_base + expert].transpose(0, 1)
                )
                out[expert, :num, :] = tmp @ w2[expert_base + expert].transpose(0, 1)

        return out
