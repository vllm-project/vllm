# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext

# FP8 activation dtypes the Triton MoE-LoRA kernels cannot consume. The LoRA
# shrink computes ``tl.dot(x, lora_a)`` and Triton only permits fp8 operands
# when *both* are fp8. LoRA adapters are bf16/fp16, so an fp8 activation ``x``
# (produced by an FP8 base MoE) forces an unsupported mixed fp8 x bf16 dot,
# which fails with ``AssertionError: Unsupported lhs dtype fp8e4nv`` during the
# profiling forward. See https://github.com/vllm-project/vllm/issues/45101.
_FP8_ACTIVATION_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)


def _assert_lora_activation_supported(x: torch.Tensor) -> None:
    """Fail fast with a clear message if MoE-LoRA gets fp8 activations.

    Args:
        x: The activation tensor fed into the MoE-LoRA shrink kernel.

    Raises:
        NotImplementedError: If ``x`` is FP8-quantized, since the Triton
            MoE-LoRA kernels require bf16/fp16 activations.
    """
    if x.dtype in _FP8_ACTIVATION_DTYPES:
        raise NotImplementedError(
            "MoE LoRA is not supported with FP8-quantized activations "
            f"(the base MoE produced {x.dtype} activations). The Triton "
            "MoE-LoRA kernels require bf16/fp16 activations because the LoRA "
            "shrink does tl.dot(x, lora_a) and LoRA adapters are bf16/fp16, "
            "so an fp8 activation forces an unsupported fp8 x bf16 dot. Serve "
            "the base model in bf16/fp16 to use LoRA, or omit --enable-lora "
            "to serve the FP8 model without LoRA. See "
            "https://github.com/vllm-project/vllm/issues/45101."
        )


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
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        _assert_lora_activation_supported(x)
        return lora_context.punica_wrapper.add_lora_w13(
            y,
            x,
            lora_context.w13_lora_a_stacked,
            lora_context.w13_lora_b_stacked,
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
        _assert_lora_activation_supported(x)
        lora_context.punica_wrapper.add_lora_w2(
            y,
            x,
            lora_context.w2_lora_a_stacked,
            lora_context.w2_lora_b_stacked,
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
