# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mamba2 selective state update (decode) kernel interface."""

from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaConfig
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class MambaSSUKernel(LinearAttnKernel):
    """The Mamba selective state update: one decode step (or a few, with
    speculative decoding) that reads and writes the cached SSM state."""

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return False

    @classmethod
    def supports_ssu_algorithm(cls) -> bool:
        return False

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = super().validate_configuration(cfg)
        if cfg.use_stochastic_rounding and not cls.supports_stochastic_rounding():
            reasons.append("stochastic rounding not supported")
        if cfg.use_ssu_algorithm and not cls.supports_ssu_algorithm():
            reasons.append("--mamba-ssu-algorithm not supported")
        return reasons

    @classmethod
    def forward(
        cls,
        mamba_config: MambaConfig,
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
