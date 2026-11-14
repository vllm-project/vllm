# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference implementation for DSpark confidence head.

Matches exactly the semantics of the current vLLM DSparkConfidenceHead:
``sigmoid(cat([hidden, markov_embed]) @ W / temp)`` computed in fp32 with
fp32 weights. Used as a ground truth in unit tests and as the unfused
fallback when the TileLang kernel is unavailable.
"""

import torch


def dspark_confidence_torch(
    hidden: torch.Tensor,  # (n, H) bf16
    markov_embed: torch.Tensor,  # (n, r) bf16 (may be 0-dim'd dummy when r=0)
    weight: torch.Tensor,  # (H + r,) fp32
    temp: float,
) -> torch.Tensor:
    """FP32-quality confidence. Returns shape (n,) fp32."""
    r = weight.shape[0] - hidden.shape[-1]
    if r > 0:
        x = torch.cat([hidden, markov_embed[:, :r]], dim=-1).float()
    else:
        x = hidden.float()
    return torch.sigmoid((x @ weight) / temp)
