# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Activation functions for I64 token-routed experts.

SwiGLU is the default activation for I64 experts, matching
the complexity-framework training setup.
"""

import torch
import torch.nn.functional as F


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(gate) * up."""
    return F.silu(gate) * up
