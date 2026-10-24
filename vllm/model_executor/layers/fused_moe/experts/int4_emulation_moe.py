# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int4 weight-only quantization emulation for MoE.

Weights are dequantized from packed int4 to BF16 once at load time;
the forward pass then runs plain TritonExperts in BF16.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Int4EmulationTritonExperts(TritonExperts):
    """Int4 W-only MoE that dequantizes weights to BF16 at load time.

    Weights arrive already dequantized (convert_to_wna16_moe_kernel_format
    does the unpacking); apply() simply forwards to TritonExperts.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        logger.warning_once(
            "Using Int4EmulationTritonExperts MoE backend. Int4 weights are "
            "dequantized to BF16 at load time "
        )
        # Weights are dequantized to BF16 before apply() is called, so
        # TritonExperts must see them as plain float — clear the int4 dtype
        # and scales so the hidden-size assertion and kernel dispatch treat
        # them as unquantized.
        self.quant_config._w1.dtype = None
        self.quant_config._w2.dtype = None
        self.quant_config._w1.scale = None
        self.quant_config._w2.scale = None

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (
            weight_key
            in (
                kInt4Static,
                kInt4Static32,
                kInt4StaticAsym,
                kInt4Static32Asym,
            )
            and activation_key is None
        )

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return None

    @property
    def block_shape(self) -> list[int] | None:
        return None

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

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
        if w1.element_size() < 2:
            raise RuntimeError(
                "Int4EmulationTritonExperts.apply() received packed int4 weights "
                "(element_size < 2). Weights must be dequantized to BF16 before "
                "the forward pass via convert_to_wna16_moe_kernel_format."
            )
        return super().apply(
            output=output,
            hidden_states=hidden_states,
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
