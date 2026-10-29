# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scalable-softmax (SSMax) query scaling."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.mistral.nvidia.ops import fusion_enabled
from vllm.models.mistral.nvidia.ops.ssmax import (
    fused_ssmax_scaling,
)


class SSMaxScaling(nn.Module):
    """Per-head scalable-softmax query scaling: ``q *= s * log(pos + 1) + b``."""

    def __init__(self, num_heads: int, tp_sharded: bool) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.empty(num_heads), requires_grad=False)
        self.b = nn.Parameter(torch.empty(num_heads), requires_grad=False)
        if tp_sharded:
            for param in (self.s, self.b):
                set_weight_attrs(param, {"weight_loader": sharded_weight_loader(0)})
        self.fused = fusion_enabled("ssmax")

    def forward(
        self,
        q: torch.Tensor,
        log_positions: torch.Tensor,
        out_head_dim: int | None = None,
    ) -> torch.Tensor:
        # q: [T, H, D], log_positions: [T] (fp32). Computed in q's dtype.
        # ``out_head_dim`` first zero-pads the head dim (NoPE layers).
        if self.fused:
            return fused_ssmax_scaling(q, log_positions, self.s, self.b, out_head_dim)
        if out_head_dim is not None:
            q = F.pad(q, (0, out_head_dim - q.shape[-1]))
        scale = (
            log_positions.to(q.dtype)[:, None, None] * self.s.to(q.dtype)[:, None]
            + self.b.to(q.dtype)[:, None]
        )
        return q * scale
