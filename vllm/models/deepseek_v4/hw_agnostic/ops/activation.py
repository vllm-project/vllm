# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU activation functions — hw-agnostic native-only copies.

Vendored from ``vllm/model_executor/layers/activation.py``. Only the
two activations used by DSv4 are kept (``SiluAndMul``,
``SiluAndMulWithClamp``); the ``CustomOp`` dispatch and the
vendor-specific ``forward_cuda``/``forward_xpu`` paths are dropped.
OOT plugins that want a CUDA fast path subclass and override
``forward``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU: ``x -> silu(x[:d]) * x[d:]`` where ``d = x.shape[-1] // 2``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class SiluAndMulWithClamp(nn.Module):
    """SwiGLU activation with input clamping (used by DSv4 MLPs).

    Computes::

        gate = clamp(x[..., :d], max=swiglu_limit)
        up = clamp(x[..., d:], min=-swiglu_limit, max=swiglu_limit)
        out = silu(gate) * up

    where ``d = x.shape[-1] // 2``.
    """

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up
