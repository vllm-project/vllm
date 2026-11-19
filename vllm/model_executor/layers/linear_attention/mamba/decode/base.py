# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mamba selective state update (decode) kernel interface, shared by Mamba1
and Mamba2 layers."""

from typing import TYPE_CHECKING, Protocol

import torch

from vllm.config.mamba import MambaConfig
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class SSULayer(Protocol):
    """What an SSU kernel reads from its layer."""

    mamba_config: MambaConfig


class MambaSSUKernel(LinearAttnKernel):
    """The Mamba selective state update: one decode step (or a few, with
    speculative decoding) that reads and writes the cached SSM state."""

    @classmethod
    def supports_mamba_type(cls, mamba_type: MambaAttentionBackendEnum) -> bool:
        return False

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return False

    @classmethod
    def supports_ssu_algorithm(cls) -> bool:
        return False

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = super().validate_configuration(cfg)
        spec = cfg.kv_cache_spec
        if isinstance(spec, MambaSpec) and not cls.supports_mamba_type(spec.mamba_type):
            reasons.append(f"{spec.mamba_type.name.title()} layers not supported")
        if cfg.use_stochastic_rounding and not cls.supports_stochastic_rounding():
            reasons.append("stochastic rounding not supported")
        if cfg.use_ssu_algorithm and not cls.supports_ssu_algorithm():
            reasons.append("--mamba-ssu-algorithm not supported")
        return reasons

    @classmethod
    def forward(
        cls,
        layer: SSULayer,
        metadata: "BaseMambaAttentionMetadata",
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> None:
        """Update each decode request's cached state (read from and written to
        its slots in `metadata.state_indices_tensor_d`) and write the output
        into `out`."""
        raise NotImplementedError
