# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.router.fused_gate import (
    FusedGateBackend,
    FusedGateConfig,
    FusedGateMetadata,
    FusedGatePlan,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm, bf16_mega_gate


@functools.cache
def _is_sm100_mega_gate_supported() -> bool:
    if not current_platform.is_device_capability_family(100):
        return False
    module = _import_deep_gemm()
    return module is not None and callable(getattr(module, "bf16_mega_gate", None))


@dataclass
class DeepGemmGatePlan(FusedGatePlan):
    hidden_states: torch.Tensor
    weight: torch.Tensor
    config: FusedGateConfig
    metadata: FusedGateMetadata

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return bf16_mega_gate(
            self.hidden_states,
            self.weight,
            self.config.top_k,
            scoring_func=self.config.scoring_func,
            routed_scaling_factor=self.config.routed_scaling_factor,
            ep_rank=self.config.ep_rank,
            bias=self.config.bias,
            image_bias=self.config.image_bias,
            image_token_mask=self.metadata.image_token_mask,
            fix_routing_mask=self.metadata.fixed_routing_mask,
            unmapped_topk_idx=self.metadata.fixed_expert_ids,
        )


class DeepGemmGateBackend(FusedGateBackend):
    """Adapter for the existing SM100 BF16 Mega Gate kernel."""

    def is_supported(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        config: FusedGateConfig,
        *,
        has_padding: bool,
    ) -> bool:
        if (
            has_padding
            or config.scoring_func != "sqrtsoftplus"
            or not config.renormalize
        ):
            return False
        # Preserve the native path's GateLinear crossover for small batches.
        gate_threshold = 1 if weight.shape[0] == 128 else 16
        if hidden_states.shape[0] <= gate_threshold:
            return False
        for bias in (config.bias, config.image_bias):
            if bias is not None and (
                bias.dtype != torch.float32
                or bias.device != weight.device
                or bias.shape != (weight.shape[0],)
                or not bias.is_contiguous()
            ):
                return False
        return (
            hidden_states.is_cuda
            and hidden_states.dtype == weight.dtype == torch.bfloat16
            and hidden_states.device == weight.device
            and hidden_states.dim() == weight.dim() == 2
            and hidden_states.shape[1] == weight.shape[1]
            and hidden_states.shape[1] > 0
            and hidden_states.shape[1] % 256 == 0
            and 0 < weight.shape[0] <= 512
            and weight.shape[0] % 4 == 0
            and 0 < config.top_k <= min(32, weight.shape[0])
            and hidden_states.shape[0] <= 1 << 20
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and hidden_states.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0
            and _is_sm100_mega_gate_supported()
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        config: FusedGateConfig,
        metadata: FusedGateMetadata,
    ) -> FusedGatePlan:
        assert metadata.valid_token_mask is None, (
            "The DeepGEMM gate adapter requires routing without a padding mask."
        )
        return DeepGemmGatePlan(hidden_states, weight, config, metadata)
