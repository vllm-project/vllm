# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loaders shared by DeepSeek-V4 model implementations."""

import torch


def attn_sink_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load local attention sinks into a padded runtime parameter."""
    num_heads = loaded_weight.shape[0]
    if num_heads > param.shape[0]:
        raise ValueError(
            f"Attention sink has {num_heads} rows, "
            f"but the destination has only {param.shape[0]}"
        )
    param[:num_heads].copy_(loaded_weight)
