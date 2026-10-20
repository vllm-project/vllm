# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LUT-B weight x MXFP8 activation MoE experts.

``LutBSm107Experts`` runs the native Rubin (SM107) grouped GEMM, where the
tensor core decodes the 3-bit LUT-B weights. ``LutBEmulationTritonExperts``
reproduces the same arithmetic on any GPU for accuracy validation: weights are
dequantized to the model dtype once at load time and activations are MXFP8
quantize-dequantized before each GEMM.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    LUT_B_SM107_TILE_K,
    LUT_B_SM107_TILE_N,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kLutBStatic,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The SM107 kernel computes 128 x 128 output tiles of one expert.
LUT_B_SM107_BLOCK_M = 128


def mxfp8_qdq(x: torch.Tensor) -> torch.Tensor:
    """Round activations through MXFP8, as the native kernel consumes them."""
    x_q, x_scales = mxfp8_e4m3_quantize(x)
    return dequant_mxfp8_to_bf16(x_q, x_scales.view(*x.shape[:-1], -1)).to(x.dtype)


class _LutBExpertsBase(TritonExperts):
    """Shared support checks. Activations arrive unquantized; each subclass
    quantizes them to MXFP8 itself."""

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kLutBStatic, kMxfp8Dynamic)

    @staticmethod
    def supports_lora() -> bool:
        return False

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return None

    @property
    def block_shape(self) -> list[int] | None:
        return None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True


class LutBEmulationTritonExperts(_LutBExpertsBase):
    """Emulated LUT-B MoE: dequantized weights, MXFP8 QDQ activations."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using emulated LUT-B MoE: weights are stored dequantized and "
            "activations are MXFP8 quantize-dequantized. This validates "
            "accuracy, not memory use or speed."
        )

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        super().activation(activation, output, input, **kwargs)
        output.copy_(mxfp8_qdq(output))

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
    ) -> None:
        super().apply(
            output=output,
            hidden_states=mxfp8_qdq(hidden_states),
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class LutBSm107Experts(_LutBExpertsBase):
    """Native Rubin LUT-B x MXFP8 MoE.

    A simple path: MXFP8-quantize the activations, then run two grouped GEMMs
    (``ops.lut_b_moe_gemm``) over routing aligned to 128-row blocks.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability((10, 7))
            and ops.lut_b_moe_gemm_supported()
        )

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % LUT_B_SM107_TILE_K == 0

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # Weights are [E, N / 128, K / 128, tile bytes].
        num_experts = w1.size(0)
        n = w1.size(1) * LUT_B_SM107_TILE_N
        k = w1.size(2) * LUT_B_SM107_TILE_K
        assert a1.size(-1) == k and w2.size(1) * LUT_B_SM107_TILE_N == k
        return num_experts, a1.size(0), n, k, topk_ids.size(1)

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
    ) -> None:
        num_experts, num_tokens, n, k, top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = num_experts

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            LUT_B_SM107_BLOCK_M,
            global_num_experts,
            expert_map,
            ignore_invalid_experts=expert_map is not None,
        )

        cache1 = _resize_cache(workspace2, (num_tokens * top_k, n))
        a1, a1_scales = mxfp8_e4m3_quantize(hidden_states)
        ops.lut_b_moe_gemm(
            a1,
            a1_scales.view(num_tokens, -1),
            w1,
            cache1,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
        )

        cache2 = _resize_cache(
            workspace13,
            (num_tokens * top_k, self.adjust_N_for_activation(n, activation)),
        )
        self.activation(activation, cache2, cache1)

        cache3 = _resize_cache(workspace2, (num_tokens, top_k, k))
        a2, a2_scales = mxfp8_e4m3_quantize(cache2)
        ops.lut_b_moe_gemm(
            a2,
            a2_scales.view(cache2.size(0), -1),
            w2,
            cache3.view(-1, k),
            None if apply_router_weight_on_input else topk_weights.float().contiguous(),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            1,
        )
        self.moe_sum(cache3, output, topk_ids, expert_map)


def make_lut_b_moe_kernel(
    experts_cls: type[_LutBExpertsBase],
    moe_config: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    supported, reason = experts_cls.is_supported_config(
        experts_cls,
        moe_config,
        kLutBStatic,
        kMxfp8Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )
    if not supported:
        raise ValueError(f"{experts_cls.__name__} is unsupported: {reason}")

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
    )
    assert prepare_finalize is not None
    return mk.FusedMoEKernel(prepare_finalize, experts_cls(moe_config, quant_config))
