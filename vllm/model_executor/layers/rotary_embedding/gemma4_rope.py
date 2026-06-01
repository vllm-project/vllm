# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-specific Rotary Positional Embeddings (proportional scaling).

Gemma4 uses "proportional" RoPE which computes inv_freq frequencies scaled
by head_dim (not rotary_dim), and zero-pads for non-rotated dimensions when
partial_rotary_factor < 1. The actual rotation uses standard neox-style
rotate_half, matching HF transformers' apply_rotary_pos_emb.
"""

import torch

from .base import RotaryEmbedding


class Gemma4RotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Extends RotaryEmbedding (which provides standard neox-style rotation
    via ops.rotary_embedding CUDA kernel) but overrides the inv_freq
    computation to match HF's _compute_proportional_rope_parameters:
    - Frequency exponents use head_dim (not rotary_dim) as denominator
    - Non-rotated dims are zero-padded (cos=1, sin=0 = identity rotation)

    When partial_rotary_factor=1.0 (the default for some variants), ALL dims are
    rotated and this is equivalent to standard RotaryEmbedding with
    head_dim-scaled frequencies.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        # Number of rotation angle pairs (from partial_rotary_factor)
        self.rope_angles = rotary_dim // 2
        # Non-rotated angle pairs per half
        self.nope_angles = (head_size // 2) - self.rope_angles

        # Important: set rotary_dim = head_size so the base class's
        # forward_static applies rotation to ALL dims of the cos/sin cache.
        # The non-rotated dims will have cos=1, sin=0 (identity) thanks
        # to our _compute_inv_freq zero-padding.
        super().__init__(
            head_size,
            head_size,  # rotary_dim = head_size (full application)
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute frequencies matching HF proportional RoPE.

        Key difference from base: exponent denominator is head_size (not
        rotary_dim), and non-rotated dims are zero-padded.
        """
        # HF formula: base ** (arange(0, 2*rope_angles, 2) / head_dim)
        freq_exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float) / self.head_size
        )
        inv_freq = 1.0 / (base**freq_exponents)

        # Zero-pad for non-rotated dims (identity rotation: cos=1, sin=0)
        if self.nope_angles > 0:
            inv_freq = torch.cat(
                [
                    inv_freq,
                    torch.zeros(self.nope_angles, dtype=torch.float),
                ]
            )
        return inv_freq

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", rope_angles={self.rope_angles}, nope_angles={self.nope_angles}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
