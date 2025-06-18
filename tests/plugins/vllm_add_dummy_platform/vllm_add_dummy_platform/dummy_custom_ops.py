# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


# Register CustomRotaryEmbedding to CustomOP.
# Using either way as below works
# Example 1:
# - @CustomOp.register_oot("RotaryEmbedding")
# Example 2:
# - @RotaryEmbedding.register_oot()
@RotaryEmbedding.register_oot()
class DummyRotaryEmbedding(RotaryEmbedding):
    """Original rotary positional embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addition_config = True

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward_oot(positions, query, key, offsets)
