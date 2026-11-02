# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton routed experts for ModelOpt IQ2_XXS / Q8_0 block weights."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.modelopt_block_quant_triton import (
    invoke_block_moe_gemv_kernel,
    invoke_block_moe_kernel,
    moe_config,
    moe_gemv_config,
)
from vllm.platforms import current_platform


class ModelOptBlockTritonExperts(TritonExperts):
    """Fused MoE on repacked block weights (`repack_blocks`); TP only, no LoRA."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        codec: str,
        lut: torch.Tensor | None,
    ):
        self.codec = codec
        self.lut = lut
        super().__init__(moe_config, quant_config)

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.has_device_capability(75)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    _supports_parallel_config = staticmethod(B12xExperts._supports_parallel_config)

    @staticmethod
    def supports_lora() -> bool:
        return False

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        E, num_tokens, N, K, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        gemv = moe_gemv_config(num_tokens, E, top_k, self.codec)
        config = gemv or moe_config(num_tokens, E, top_k)
        launch = invoke_block_moe_gemv_kernel if gemv else invoke_block_moe_kernel
        cache1 = _resize_cache(workspace2, (num_tokens, top_k, N))
        cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k, self.adjust_N_for_activation(N, activation)),
        )
        cache3 = _resize_cache(workspace2, (num_tokens, top_k, K))
        fused_relu2 = gemv is not None and activation == MoEActivation.RELU2_NO_MUL
        if gemv and expert_map is None:
            # Each GEMV CTA owns one routed row, so sorting cannot share work.
            routing = (None, topk_ids.contiguous(), None)
        else:
            routing = moe_align_block_size(
                topk_ids, 1 if gemv else config["BLOCK_SIZE_M"], E, expert_map
            )

        launch(
            hidden_states, w1, self.w1_scale, self.lut,
            cache2 if fused_relu2 else cache1, None, *routing,
            False, top_k, config, self.codec,
            **({"apply_relu2": fused_relu2} if gemv else {}),
        )  # fmt: skip
        if not fused_relu2:
            self.activation(activation, cache2, cache1.view(-1, N))
        launch(
            cache2, w2, self.w2_scale, self.lut, cache3, topk_weights, *routing,
            not apply_router_weight_on_input, 1, config, self.codec,
        )  # fmt: skip
        self.moe_sum(cache3, output)
