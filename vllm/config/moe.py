# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for MoE backends."""

from enum import Enum
from typing import Any

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config


class Mxfp4Backend(Enum):
    NONE = 0

    # FlashInfer Backends (NVIDIA)
    SM100_FI_MXFP4_MXFP8_TRTLLM = 1
    SM100_FI_MXFP4_MXFP8_CUTLASS = 2
    SM100_FI_MXFP4_BF16 = 3
    SM90_FI_MXFP4_BF16 = 4

    # Marlin Backend
    MARLIN = 5

    # Triton Backend
    TRITON = 6

    # ROCm CK (Composable Kernel) Backend
    CK = 7


@config
@dataclass
class MoeConfig:
    """
    If not specified via --moe_config.backend, the backend will be
    auto-selected based on platform and libraries

    usage:
        vllm serve model_name --moe_config.backend=TRITON
        vllm serve model_name --moe_config.backend=CK
        vllm serve model_name --moe_config.backend=MARLIN
    """

    backend: Mxfp4Backend | None = None

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: list[str] = []
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return Mxfp4Backend[value.upper()]
        return value
