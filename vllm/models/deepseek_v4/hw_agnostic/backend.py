# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse-MLA backend stub for the hw_agnostic path."""

from vllm.models.deepseek_v4.hw_agnostic.attention.sparse_mla import (
    DeepseekV4FlashMLABackend,
)
from vllm.platforms.interface import DeviceCapability

__all__ = [
    "DeepseekV4HWAgnosticBackend",
]


class DeepseekV4HWAgnosticBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_HW_AGNOSTIC"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # The agnostic stream does not call FlashMLA kernels, so the
        # parent backend's Hopper / Blackwell gate does not apply. Any
        # Triton-capable accelerator is fine.
        return True
