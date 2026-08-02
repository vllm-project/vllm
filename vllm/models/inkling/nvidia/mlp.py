# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling dense SwiGLU MLP (also used as the MoE shared expert).

The checkpoint stores the gate/up projection as a single fused, *interleaved*
weight (``[gate0, up0, gate1, up1, ...]``), so we use a plain
``ColumnParallelLinear`` whose contiguous row-sharding keeps each gate/up pair
together, and an interleaved SwiGLU activation.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig


class InklingDenseMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        use_global_scale: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.down_proj",
        )
        if use_global_scale:
            self.global_scale = nn.Parameter(torch.empty(1), requires_grad=False)
        else:
            self.global_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .ops import silu_and_mul_triton

        gate_up, _ = self.gate_up_proj(x)
        x = silu_and_mul_triton(gate_up)
        x, _ = self.down_proj(x)
        if self.global_scale is not None:
            x = x * self.global_scale
        # TP-partial output: the layer's reduce-scatter fallback consumes it.
        return x
