# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.experts.lora_context import (
    MoELoRAContext,
    MoELoRAExecutionState,
)


class LoRAExpertsMixin:
    """Mixin for FusedMoEExpertsModular subclasses that natively handle
    MoELoRAContext inside their apply() implementation.

    Mixing this class in:
    - Flips supports_lora() to True so _can_fused_experts_support lets
      LoRA through the gate check.
    - Stashes a MoELoRAContext on the experts instance via
      set_lora_context(), which apply() consumes from self._lora_context.
    - Provides apply_w13_lora / apply_w2_lora helpers that dispatch to
      the PunicaWrapper kernels.

    The helper methods are pure functions of their inputs; all required
    state is on lora_context or passed as arguments.
    """

    moe_config: FusedMoEConfig
    _lora_context: MoELoRAContext | None = None

    def set_lora_context(self, ctx: MoELoRAContext) -> None:
        self._lora_context = ctx

    @staticmethod
    def supports_lora() -> bool:
        return True

    def apply_w13_lora(
        self,
        lora_context: MoELoRAContext,
        *,
        y: torch.Tensor,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        num_tokens: int,
        top_k_num: int,
        add_inputs: bool = True,
        swap_w13_slices: bool = False,
    ) -> MoELoRAExecutionState:
        routes = lora_context.local_lora_routes
        if routes is None:
            routes = lora_context.punica_wrapper.lora_route_mapping
        if routes is not None and not lora_context.is_routed_projection:
            slots, coefficients = routes
            state = MoELoRAExecutionState(routes=[])
            if not add_inputs:
                y.zero_()
            if num_tokens == 0:
                return state
            for route in range(slots.shape[1]):
                delta = torch.zeros_like(y)
                route_context = replace(
                    lora_context,
                    local_token_lora_mapping=slots[:num_tokens, route]
                    .to(dtype=torch.int32)
                    .contiguous(),
                    is_routed_projection=True,
                )
                alignment = self.apply_w13_lora(
                    route_context,
                    y=delta,
                    x=x,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                    add_inputs=True,
                    swap_w13_slices=swap_w13_slices,
                )
                weights = coefficients[:num_tokens, route]
                delta.view(num_tokens, top_k_num, -1).mul_(weights[:, None, None])
                y.add_(delta)
                assert state.routes is not None
                state.routes.append((weights, alignment))
            return state
        w13_lora_a_stacked = lora_context.w13_lora_a_stacked
        w13_lora_b_stacked = lora_context.w13_lora_b_stacked
        if lora_context.enable_moe_shared_loras:
            # w13 lora_A is shared across experts (collapsed expert-dim 1);
            # broadcast to local_num_experts via a stride-0 view. The kernel
            # derives num_experts from shape[1] and indexes via stride(1)==0.
            w13_lora_a_stacked = tuple(
                a.expand(-1, lora_context.local_num_experts, -1, -1)
                for a in w13_lora_a_stacked
            )
        if swap_w13_slices:
            # The expand kernel writes slice j into the j-th half of y's last
            # dim. Reversing the (gate, up) slice tuples makes it emit
            # [up, gate] order directly -- used by the FlashInfer trtllm path,
            # whose SwiGLU expects the up half first, to avoid an out-of-place
            # concat swap afterwards.
            w13_lora_a_stacked = w13_lora_a_stacked[::-1]
            w13_lora_b_stacked = w13_lora_b_stacked[::-1]
        interleave_w13 = (
            lora_context.w13_num_slices == 2
            and self.moe_config.activation == MoEActivation.SWIGLUOAI
        )
        lora_output = torch.zeros_like(y) if interleave_w13 else y
        result = lora_context.punica_wrapper.add_lora_w13(
            lora_output,
            x,
            w13_lora_a_stacked,
            w13_lora_b_stacked,
            topk_ids,
            topk_weights,
            expert_map,
            w1,
            w2,
            num_tokens,
            top_k_num,
            lora_context.max_loras,
            lora_context.adapter_enabled,
            lora_context.local_num_experts,
            lora_context.top_k,
            lora_context.w13_num_slices,
            lora_context.fully_sharded,
            lora_context.use_tuned_config,
            add_inputs=add_inputs or interleave_w13,
            token_lora_mapping=lora_context.local_token_lora_mapping,
            routed=lora_context.is_routed_projection,
        )
        if interleave_w13:
            # Map gate/up slices onto even/odd output channels without a copy.
            gate_up_delta = lora_output.unflatten(-1, (2, -1))
            gate_up_output = y.unflatten(-1, (-1, 2)).transpose(-1, -2)
            if add_inputs:
                gate_up_output.add_(gate_up_delta)
            else:
                gate_up_output.copy_(gate_up_delta)
        return MoELoRAExecutionState(alignment=result)

    def apply_w2_lora(
        self,
        lora_context: MoELoRAContext,
        *,
        y: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        lora_metadata: MoELoRAExecutionState,
        num_tokens: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        top_k_num: int,
        add_inputs: bool = True,
    ) -> None:
        if lora_metadata.routes is not None:
            if not add_inputs:
                y.zero_()
            for weights, alignment in lora_metadata.routes:
                delta = torch.zeros_like(y)
                self.apply_w2_lora(
                    replace(lora_context, is_routed_projection=True),
                    y=delta,
                    x=x,
                    topk_weights=topk_weights,
                    lora_metadata=alignment,
                    num_tokens=num_tokens,
                    w1=w1,
                    w2=w2,
                    top_k_num=top_k_num,
                    add_inputs=True,
                )
                delta.view(num_tokens, top_k_num, -1).mul_(weights[:, None, None])
                y.add_(delta)
            return
        assert lora_metadata.alignment is not None
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = lora_metadata.alignment
        w2_lora_b_stacked = lora_context.w2_lora_b_stacked
        if lora_context.enable_moe_shared_loras:
            # w2 lora_B is shared across experts (collapsed expert-dim 1);
            # broadcast to local_num_experts via a stride-0 view.
            w2_lora_b_stacked = tuple(
                b.expand(-1, lora_context.local_num_experts, -1, -1)
                for b in w2_lora_b_stacked
            )
        lora_context.punica_wrapper.add_lora_w2(
            y,
            x,
            lora_context.w2_lora_a_stacked,
            w2_lora_b_stacked,
            topk_weights,
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
            num_tokens,
            w1,
            w2,
            top_k_num,
            lora_context.max_loras,
            lora_context.adapter_enabled,
            lora_context.top_k,
            lora_context.fully_sharded,
            lora_context.tp_rank,
            lora_context.use_tuned_config,
            add_inputs=add_inputs,
            routed=lora_context.is_routed_projection,
        )
