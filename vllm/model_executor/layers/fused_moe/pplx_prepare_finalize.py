# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import pplx_kernels as pplx
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils import cdiv, round_up


def pplx_hidden_dim_scale_bytes(
    max_num_tokens: int,
    hidden_dim: int,
    in_dtype: torch.dtype,
    quant_dtype: Optional[torch.dtype],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]],
):
    # For blocked per token: set to
    #   ceil_div(hidden_dim, block_size) * sizeof(float32)
    # For per-token: set to 4 * sizeof(float32) (x4 for alignment)
    if quant_dtype is not None:
        assert quant_dtype.itemsize == 1
        hidden_dim_bytes = hidden_dim * quant_dtype.itemsize
        elem_size = torch.float32.itemsize
        align = 16

        if per_act_token_quant:
            # per-token
            assert block_shape is None
            hidden_scale_bytes = round_up(max_num_tokens * elem_size, align)
        elif block_shape is not None:
            # per-group
            block_size = block_shape[1]
            num_blocks = cdiv(hidden_dim, block_size)
            hidden_scale_bytes = round_up(num_blocks * elem_size, align)
        else:
            # per-tensor
            # ?
            hidden_scale_bytes = round_up(elem_size, align)
    else:
        hidden_dim_bytes = hidden_dim * in_dtype.itemsize
        hidden_scale_bytes = 0

    #print(f"pplx bytes {hidden_dim_bytes}, {hidden_scale_bytes}")

    return hidden_dim_bytes, hidden_scale_bytes


# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
class PplxPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        a2a: pplx.AllToAll,
        max_num_tokens: int,
        world_size: int,
        rank: int,
        dp_size: int,
    ):
        assert max_num_tokens > 0
        self.a2a = a2a
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.uint32

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert topk_ids.size(0) == num_tokens
        # assert expert_map is None, "NYI"

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1, (None if quant_config.per_act_token_quant else a1_scale),
            quant_config.quant_dtype, quant_config.per_act_token_quant,
            quant_config.block_shape)

        if a1q_scale is not None:
            scalar_scales = a1q_scale.numel() == 1

            # pplx requires 2-d scales even for scalar scales
            if a1q_scale.dim() <= 1:
                assert scalar_scales
                a1q_scale = a1q_scale.view(1, 1)

            orig_a_scale_block_shape = a1q_scale.shape[-1]

            # pad out scales if needed. TODO (bnell): do for non-scalar scales?
            if scalar_scales:
                #print(f"a1q_scale {a1q.shape}, {a1q_scale.shape}")
                a1q_scale = a1q_scale.repeat(a1q.shape[1],
                                             4 * torch.float32.itemsize)

            #assert a1_scale is None or a1_scale.shape[0] == a1q.shape[1], f"{a1_scale.shape}, {a1q_scale.shape}"

        assert a1q_scale is None or a1q_scale.ndim == 2, \
            f"{0 if a1q_scale is None else (a1q_scale.ndim, a1q_scale.shape)}"

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
        #print(f"EXPERT_X {(num_local_experts, self.max_num_tokens * num_dp, hidden_dim)}, {a1q.dtype}, {device}")
        expert_x = torch.zeros(
            (num_local_experts, self.max_num_tokens * num_dp, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            float32_size = torch.float32.itemsize
            block_size = (quant_config.block_shape[1] if quant_config.
                          block_shape is not None else 1) * float32_size

            expert_x_scale_shape = (
                num_local_experts,
                expert_x.size(1),
                (expert_x.size(2) + block_size - 1) //
                block_size if not scalar_scales else 1,
            )

            #print(f"EXPERT_X_SCALE {expert_x_scale_shape}")

            expert_x_scale = torch.zeros(
                expert_x_scale_shape,
                dtype=torch.float32,
                device=expert_x.device,
            )

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None

        #print(f"DISPATCH X={expert_x.shape}, X_SCALE={expert_x_scale.shape}, A={a1q.shape}, A_SCALE={a1q_scale.shape}, TOPK={topk_ids}")

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=topk_ids,
            bound_m=bound_m,
        )
        #print(f"DISPATCH DONE {device}")

        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, 0:1]

        if expert_x_scale is not None:
            expert_x_scale = expert_x_scale[:, :, :orig_a_scale_block_shape]
            assert expert_x_scale.ndim == 3

        return expert_x, expert_x_scale, expert_num_tokens, None, None

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

        #print(f"COMBINE {output.device}")
        self.a2a.combine(out_tokens=output,
                         indices=topk_ids,
                         weights=topk_weights,
                         expert_y=fused_expert_output,
                         bound_m=bound_m)
        #print(f"COMBINE DONE {output.device}")
