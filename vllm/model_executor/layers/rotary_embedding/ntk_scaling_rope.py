# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from .base import RotaryEmbedding


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with fixed and mixed NTK scaling.
    https://kexue.fm/archives/9706"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        mixed_b: float | None = None,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.mixed_b = mixed_b
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        base = self.base * (self.scaling_factor if self.mixed_b is None else 1)
        inv_freq = super()._compute_inv_freq(base)

        if self.mixed_b is None:
            inv_freq = inv_freq / self.scaling_factor ** (2 / self.rotary_dim)
        else:
            a = (
                torch.tensor(self.scaling_factor).log()
                / (self.rotary_dim / 2) ** self.mixed_b
            )
            lambda_1_m = (
                a * torch.arange(1, self.rotary_dim // 2 + 1).float() ** self.mixed_b
            ).exp()
            inv_freq = inv_freq / lambda_1_m

        return inv_freq
