# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash latent experts with HYV4's routing, shared experts and weight format."""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


def _defers_allreduce(config: HYV4FlashConfig) -> bool:
    """Whether the MLP leaves its output TP-partial for the sandwich norm to
    reduce in a fused all-reduce + RMSNorm."""
    parallel_config = get_current_vllm_config().parallel_config
    return bool(
        config.sandwich_norm
        and not parallel_config.enable_expert_parallel
        and not parallel_config.use_sequence_parallel_moe
    )


def _post_norm(norm: RMSNorm | None, x: torch.Tensor, reduce: bool) -> torch.Tensor:
    if norm is None:
        return x
    return fused_allreduce_rms_norm(x, None, norm)[0] if reduce else norm(x)


class LatentUpProjection(ReplicatedLinear):
    """Routed latent-to-hidden projection.

    When bias-free and unquantized it commutes with the TP sum, so the routed
    and shared outputs share one all-reduce, and it takes the shared output as
    the GEMM's accumulate input instead of a separate add.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        plain = self.bias is None and isinstance(
            self.quant_method, UnquantizedLinearMethod
        )
        self.reduce_commutative = self.accumulates_residual = plain

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        if residual is None:
            return super().forward(x)
        assert self.accumulates_residual
        return residual.addmm_(x, self.weight.t()), None


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        self.defer_allreduce = _defers_allreduce(config)
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            reduce_results=not self.defer_allreduce,
            prefix=prefix,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _post_norm(self.post_fc2_norm, super().forward(x), self.defer_allreduce)


class HYV4FlashMoE(HYV4MoEFused):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        latent = config.routed_expert_hidden_size
        down: nn.Module | None = None
        up: nn.Module | None = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = LatentUpProjection(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        defer_allreduce = _defers_allreduce(config)
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            routed_hidden_size=latent,
            routed_input_transform=down,
            routed_output_transform=up,
            reduce_results=not defer_allreduce,
        )
        self.defer_allreduce = defer_allreduce
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _post_norm(
            self.combine_output_norm,
            super().forward(hidden_states),
            self.defer_allreduce,
        )
