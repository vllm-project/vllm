# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)


# Note use: layer.get_all_to_all() to get an AllToAll instance
# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(self,
                 a2a: pplx.AllToAll,
                 max_num_tokens: int,
                 world_size: int,
                 rank: int,
                 dp_size: int,
                 quant_dtype: Optional[torch.dtype] = None,
                 block_shape: Optional[list[int]] = None):
        super().__init__()
        assert max_num_tokens > 0
        self.a2a = a2a
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size
        self.quant_dtype = quant_dtype

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert rank_topk_ids.size(0) == num_tokens
        # assert expert_map is None, "NYI"

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
            a2_scale.numel() != 1 if a2_scale is not None else False)

        a1q, a1q_scale = moe_kernel_quantize_input(a1, a1_scale,
                                                   self.quant_dtype,
                                                   per_act_token,
                                                   self.block_shape)

        # rem_experts need to be 0 for pplx to work properly.
        rem_experts = num_experts % self.world_size
        assert rem_experts == 0
        num_local_experts = ((num_experts // self.world_size) +
                             (1 if self.rank < rem_experts else 0))

        expert_num_tokens = torch.empty(
            num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        num_dp = self.world_size // self.dp_size
        expert_x = torch.empty(
            (num_local_experts, self.max_num_tokens * num_dp, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )

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

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=rank_topk_ids,
            bound_m=bound_m,
        )

        return expert_x, expert_x_scale, expert_num_tokens

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        num_tokens = output.size(0)  # M
        # This argument is optional
        # There's not much point setting this unless it is != topk_ids.size(0)
        bound_m: Optional[torch.Tensor] = None

        assert topk_ids.size(0) == num_tokens, (
            f"{topk_ids.size(0)} == {num_tokens}")
        assert output.size(0) <= self.max_num_tokens, (
            f"{output.size(0)} <= {self.max_num_tokens}")
        assert output.size(1) == fused_expert_output.size(-1)

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        self.a2a.combine(out_tokens=output,
                         indices=topk_ids,
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)
