# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from .base import RotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding


class TeleChat3RoPEScaledRotaryEmbedding(YaRNScalingRotaryEmbedding):
    """TeleChat3 uses a variant of YaRN method.

    To achieve code reuse as much as possible, we have rewritten the
    `get_mscale` method in the initialization function
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        truncate: bool = True,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate

        def get_mscale(scale, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.07 * mscale * math.log(scale) + 1.0

        self.mscale = float(get_mscale(self.scaling_factor) * attn_factor)
        # Initialization must be performed after mscale, otherwise mscale is useless
        RotaryEmbedding.__init__(
            self,
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )
