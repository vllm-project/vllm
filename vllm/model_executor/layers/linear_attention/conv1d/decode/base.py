# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The causal conv1d decode kernel interface."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata


class Conv1dDecodeKernel(LinearAttnKernel):
    """The causal conv1d over each decode's new tokens (one, or the verify
    window with speculative decoding), updating its cached conv state."""

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return False

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = super().validate_configuration(cfg)
        if cfg.num_speculative_tokens > 0 and not cls.supports_spec_decode():
            reasons.append("speculative decoding not supported")
        return reasons

    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        metadata: "BaseMambaAttentionMetadata",
        *,
        max_query_len: int,
    ) -> torch.Tensor:
        """Convolve the decode tokens `x` (`(num_tokens, dim)`) and return the
        output in the same layout. Each row reads and writes its slot in
        `metadata.state_indices_tensor_d` (its first column) of `conv_state`
        (`(num_slots, dim, state_len)`); with speculative decoding, the row's
        `max_query_len` tokens start after its accepted tokens' state."""
        raise NotImplementedError
