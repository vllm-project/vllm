# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR configuration."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

# Named OSCAR presets, selected via --kv-cache-dtype. Each maps to a frozen
# (key_bits, value_bits) pair. The remaining, deployment-specific knobs
# (rotation matrix paths, clip ratios, mixed-precision window sizes) are read
# from environment variables, mirroring the SGLang reference UX so that a
# checkpoint's RotationZoo artifacts can be pointed at without re-exporting a
# model. See ``OscarConfig.from_cache_dtype``.
OSCAR_PRESETS: dict[str, dict] = {
    # OSCAR's headline 2-bit KV configuration (BPE ~2.28 in the paper).
    "oscar_int2": {"key_quant_bits": 2, "value_quant_bits": 2},
}

# Environment knobs (defaults match the OSCAR README serving recipe).
_ENV_K_ROTATION = "VLLM_OSCAR_K_ROTATION_PATH"
_ENV_V_ROTATION = "VLLM_OSCAR_V_ROTATION_PATH"
_ENV_K_CLIP = "VLLM_OSCAR_K_CLIP_RATIO"
_ENV_V_CLIP = "VLLM_OSCAR_V_CLIP_RATIO"
_ENV_GROUP_SIZE = "VLLM_OSCAR_GROUP_SIZE"


@dataclass
class OscarConfig:
    """Configuration for OSCAR INT2 KV-cache quantization.

    Applies a *calibrated* per-layer orthogonal rotation (loaded from disk)
    to keys and values, optional percentile clipping, then per-group
    asymmetric INT2 scalar quantization.

    Because the rotation is orthogonal, the attention scores are invariant
    under it: keys are stored as ``K @ R_k`` and the query is rotated by the
    same ``R_k`` at score time, so ``(Q R_k)(K R_k)^T = Q K^T``. Values are
    stored as ``V @ R_v`` and the attention output is mapped back with
    ``R_v^T`` after the weighted sum, which is mathematically equivalent to
    absorbing ``R_v`` into the output projection (the ``ABSORB_V_ROTATION``
    path in the reference) without touching model weights.

    Args:
        head_dim: Attention head dimension (e.g. 128).
        key_quant_bits: Bits per key element (2 for the headline preset).
        value_quant_bits: Bits per value element (2 for the headline preset).
        group_size: Quantization group size along head_dim. One scale/zero
            pair is stored per group. With ``head_dim <= group_size`` this is
            a single group per vector (the Qwen3 ``head_dim=128`` case).
        k_clip_ratio: Fraction of the per-vector dynamic range retained for
            keys (0 disables clipping). 0.96 in the README recipe.
        v_clip_ratio: As ``k_clip_ratio`` for values. 0.92 in the recipe.
        k_rotation_path: Path to the ``[num_layers, head_dim, head_dim]`` (or
            per-layer) key rotation tensor. Empty means identity (no
            rotation), which degrades to naive clipped INT2.
        v_rotation_path: As ``k_rotation_path`` for values.
    """

    head_dim: int = 128
    key_quant_bits: int = 2
    value_quant_bits: int = 2
    group_size: int = 128
    k_clip_ratio: float = 0.0
    v_clip_ratio: float = 0.0
    k_rotation_path: str = ""
    v_rotation_path: str = ""

    # ----- derived geometry ------------------------------------------------
    @property
    def num_groups(self) -> int:
        return math.ceil(self.head_dim / self.group_size)

    @property
    def key_levels(self) -> int:
        return 2**self.key_quant_bits

    @property
    def value_levels(self) -> int:
        return 2**self.value_quant_bits

    @property
    def key_data_bytes(self) -> int:
        """Packed index bytes for one key vector (4 INT2 values per byte)."""
        return math.ceil(self.head_dim * self.key_quant_bits / 8)

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_quant_bits / 8)

    @property
    def meta_bytes(self) -> int:
        """Per-vector scale+zero metadata: fp16 scale + fp16 zero per group."""
        return self.num_groups * 4

    @property
    def key_packed_size(self) -> int:
        return self.key_data_bytes + self.meta_bytes

    @property
    def value_packed_size(self) -> int:
        return self.value_data_bytes + self.meta_bytes

    @property
    def slot_size(self) -> int:
        """Combined per-head per-position bytes: [key_packed | value_packed]."""
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size rounded up to an even number so ``slot // 2`` is integral
        (vLLM derives ``effective_head_size = slot_size_aligned // 2``)."""
        s = self.slot_size
        return s + (s % 2)

    # ----- constructors ----------------------------------------------------
    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> OscarConfig:
        """Create a config from a named preset plus environment knobs."""
        if cache_dtype not in OSCAR_PRESETS:
            valid = ", ".join(OSCAR_PRESETS.keys())
            raise ValueError(
                f"Unknown OSCAR cache dtype: {cache_dtype!r}. Valid presets: {valid}"
            )
        preset = OSCAR_PRESETS[cache_dtype]
        group_size = int(os.environ.get(_ENV_GROUP_SIZE, "128"))
        return OscarConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            group_size=group_size,
            k_clip_ratio=float(os.environ.get(_ENV_K_CLIP, "0.0")),
            v_clip_ratio=float(os.environ.get(_ENV_V_CLIP, "0.0")),
            k_rotation_path=os.environ.get(_ENV_K_ROTATION, ""),
            v_rotation_path=os.environ.get(_ENV_V_ROTATION, ""),
        )

    @staticmethod
    def get_boundary_skip_layers(
        model_config: ModelConfig,
        n: int = 2,
    ) -> list[str]:
        """Layer indices kept in native dtype (boundary protection).

        OSCAR's reference keeps a BF16 sink + recent *token* window; until
        that allocator-level machinery lands in vLLM, this port protects the
        first and last ``n`` attention layers in full precision, matching the
        TurboQuant integration. Hybrid models disable boundary protection
        (too few full-attention layers to spare two on each side).
        """
        if model_config.is_hybrid:
            return []
        num_layers = model_config.hf_text_config.num_hidden_layers
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)
        indices = sorted(set(list(range(n)) + list(range(num_layers - n, num_layers))))
        return [str(i) for i in indices]
