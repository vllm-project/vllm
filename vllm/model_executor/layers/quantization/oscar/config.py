# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR configuration."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = logging.getLogger(__name__)


@dataclass
class OscarConfig(QuantizationConfig):
    """Configuration for OSCAR INT2 KV-cache quantization.

    Applies per-layer orthogonal rotation followed by clipped INT2
    scalar quantization for both keys and values.

    Expected environment variables:
      - VLLM_OSCAR_K_ROTATION_PATH: Path to the K rotation tensors
      - VLLM_OSCAR_V_ROTATION_PATH: Path to the V rotation tensors
      - VLLM_OSCAR_K_CLIP_RATIO: Clip ratio for K (default: 0.96)
      - VLLM_OSCAR_V_CLIP_RATIO: Clip ratio for V (default: 0.92)
    """

    head_dim: int = 128

    def __post_init__(self) -> None:
        super().__init__()

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single KEY vector.

        INT2 quant: ceil(head_dim * 2 / 8) bytes.
        + 4 bytes for scale (fp16) and zero (fp16).
        Aligned to 16 bytes to prevent CUDA illegal memory access
        during vectorized Triton loads/stores.
        """
        data_bytes = math.ceil(self.head_dim * 2 / 8)
        raw_size = data_bytes + 4
        return (raw_size + 15) // 16 * 16

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.
        Aligned to 16 bytes to prevent CUDA illegal memory access
        during vectorized Triton loads/stores.
        """
        data_bytes = math.ceil(self.head_dim * 2 / 8)
        raw_size = data_bytes + 4
        return (raw_size + 15) // 16 * 16

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined)."""
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size is already 16-byte aligned."""
        return self.slot_size

    @staticmethod
    def get_k_clip_ratio() -> float:
        return float(os.environ.get("VLLM_OSCAR_K_CLIP_RATIO", "0.96"))

    @staticmethod
    def get_v_clip_ratio() -> float:
        return float(os.environ.get("VLLM_OSCAR_V_CLIP_RATIO", "0.92"))

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> OscarConfig:
        if cache_dtype != "oscar_int2":
            raise ValueError(
                f"Unknown OSCAR cache dtype: {cache_dtype!r}. "
                "Only 'oscar_int2' is currently supported."
            )
        return OscarConfig(head_dim=head_dim)

    def get_name(self) -> QuantizationMethods:
        return "oscar_int2"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return []

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OscarConfig:
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None
