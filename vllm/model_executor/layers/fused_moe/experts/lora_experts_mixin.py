# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext


class LoRAExpertsMixin:
    """
    Mixin for FusedMoEExpertsModular subclasses that natively handle
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
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
        return lora_context.punica_wrapper.add_lora_w13(
            y,
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
            add_inputs=add_inputs,
            token_lora_mapping=lora_context.local_token_lora_mapping,
        )

    def apply_w2_lora(
        self,
        lora_context: MoELoRAContext,
        *,
        y: torch.Tensor,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        sorted_token_ids_lora: torch.Tensor | None,
        expert_ids_lora: torch.Tensor | None,
        num_tokens_post_padded_lora: torch.Tensor | None,
        token_lora_mapping: torch.Tensor | None,
        num_tokens: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        top_k_num: int,
        add_inputs: bool = True,
    ) -> None:
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
        )
