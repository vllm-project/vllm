# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR INT2 KV-cache quantization config."""

from __future__ import annotations

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


@dataclass
class OscarConfig(QuantizationConfig):
    """Config for OSCAR INT2 KV-cache quantization.

    Per-layer orthogonal rotation + clipped INT2 scalar quantization
    for both keys and values.
    """

    head_dim: int = 128

    def __post_init__(self) -> None:
        super().__init__()

    @property
    def key_packed_size(self) -> int:
        raw = math.ceil(self.head_dim * 2 / 8) + 4
        return (raw + 15) // 16 * 16

    @property
    def value_packed_size(self) -> int:
        raw = math.ceil(self.head_dim * 2 / 8) + 4
        return (raw + 15) // 16 * 16

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
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
            raise ValueError(f"Unknown OSCAR cache dtype: {cache_dtype!r}")
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
