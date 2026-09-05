# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intel XPU Kimi-K3 MTP model interfaces."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors


class KimiK3MTP(nn.Module):
    """Placeholder for the native Intel XPU Kimi-K3 MTP draft model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del vllm_config, prefix
        raise NotImplementedError("Native XPU Kimi-K3 MTP is not implemented.")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError


__all__ = ["KimiK3MTP"]
