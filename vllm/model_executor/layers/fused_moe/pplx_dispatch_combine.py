# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


logger = init_logger(__name__)

# Note use: layer.get_all_to_all() to get an AllToAll instance
# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
class PplxDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):

    def __init__(self,
                 a2a: pplx.AllToAll,
                 max_num_tokens: int,
                 world_size: int,
                 dp_size: int,
                 rank: int,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[List[int]] = None):
        super().__init__()
        self.a2a = a2a
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.rank = rank
        self.quant_dtype = quant_dtype

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Is this always going to be a1.device?
        device = a1.device
        num_tokens = a1.shape[0]   # M
        hidden_dim = a1.shape[-1]  # K

        assert expert_map is None, "NYI"

        if apply_router_weight_on_input:
            topk = rank_topk_ids.shape[1]
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        a1q, a1q_scale = moe_kernel_quantize_input(a1, a1_scale,
                                                   self.quant_dtype,
                                                   per_act_token,
                                                   self.block_shape)

        rem_experts = num_experts % self.world_size
        num_local_experts = ((num_experts // self.world_size) +
                             (1 if self.rank < rem_experts else 0))

        expert_num_tokens = torch.empty(
            num_local_experts,
            dtype=torch.int32,
            device=device,
        )
        #expert_num_tokens.fill_(-1)  # debugging, remove later

        num_dp = self.world_size // self.dp_size
        logger.debug(f"GOT HERE A {self.rank}: {self.max_num_tokens} {num_dp} {hidden_dim}")
        expert_x = torch.empty(
            (num_local_experts, self.max_num_tokens * num_dp, a1q.shape[-1]),
            dtype=a1q.dtype,
            device=device,
        )
        expert_x.fill_(0) #torch.nan   # debugging, remove later

        logger.debug(f"GOT HERE B {self.rank}")

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            float32_size = torch.float32.itemsize
            block_size = (self.block_shape[0] if self.block_shape is not None
                          else 1) * float32_size
            expert_x_scale = torch.empty(
                (
                    num_experts,
                    expert_x.size(1),
                    (expert_x.size(2) + block_size - 1) // block_size,
                ),
                dtype=torch.float32,
                device=device,
            )

        logger.debug(f"GOT HERE C {self.rank}")

        # This argument is optional, defaults to indices.shape[0]
        # This causes a deadlock????
        #bound_m = get_forward_context().dp_metadata.dp_rank_num_tokens
        #bound_m = torch.tensor([num_tokens], dtype=torch.uint32, device=device)
        bound_m = None

        # TODO: optimize this?
        indices = rank_topk_ids.to(dtype=torch.uint32)

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=indices,
            bound_m=bound_m,
        )
        return expert_x, expert_x_scale, expert_num_tokens

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        device = fused_expert_output.device
        #device = torch.device("cuda", self.rank)
        #device = get_dp_group().device
        #assert fused_expert_output.device == device

        logger.debug(f"COMBINE START {self.rank}")

        # This argument is optional
        #bound_m = get_forward_context().dp_metadata.dp_rank_num_tokens
        #num_tokens = fused_expert_output.shape[0]   # M
        #bound_m = torch.tensor([num_tokens], dtype=torch.uint32, device=device)
        bound_m = None

        assert output.shape[0] <= self.max_num_tokens
        assert output.shape[1] == fused_expert_output.shape[-1]

        # Set weights to 1 if we did them in dispatch.  This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        self.a2a.combine(out_tokens=output,
                         indices=topk_ids.to(torch.uint32),
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)

        logger.debug(f"COMBINE END {self.rank}")
