# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sublayer steps for Ampere SM80 Specialized Decode."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..runtime.decode import DecodeStep


class AmpereAttentionStep:
    """Specialized Attention Step on SM80."""

    def __init__(self, attn_layer):
        self.attn = attn_layer

    def run(
        self,
        layer_input: torch.Tensor,
        positions: torch.Tensor,
        step: "DecodeStep",
    ) -> torch.Tensor:
        """Execute attention forward for decode batch."""
        return self.attn(positions, layer_input)


class AmpereMoEStep:
    """Specialized MoE Step on SM80 with low-batch fast path."""

    def __init__(self, ffn_layer, max_rows: int = 192):
        self.ffn = ffn_layer
        self.max_rows = max_rows

    def run(
        self,
        ffn_input: torch.Tensor,
        input_ids: torch.Tensor | None,
        step: "DecodeStep",
    ) -> torch.Tensor:
        """Execute MoE forward passing input_ids for multimodal vision routing."""
        return self.ffn(ffn_input, input_ids=input_ids)
