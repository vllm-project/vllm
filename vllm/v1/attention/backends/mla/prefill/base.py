# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for MLA prefill backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )


class MLAPrefillBackend(ABC):
    """Abstract base class for MLA prefill backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    requires_r1_mla_dimensions: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @classmethod
    def supports_compute_capability(cls, device_capability: "DeviceCapability") -> bool:
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        selector_config: "MLAPrefillSelectorConfig",
    ) -> list[str]:
        invalid_reasons: list[str] = []

        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append(
                f"compute capability {device_capability.major}."
                f"{device_capability.minor} not supported"
            )

        if not cls.supports_dtype(selector_config.dtype):
            invalid_reasons.append(f"dtype {selector_config.dtype} not supported")

        if not cls.is_available():
            invalid_reasons.append("required dependencies not available")

        if cls.requires_r1_mla_dimensions and not selector_config.is_r1_compatible:
            invalid_reasons.append(
                "model does not have DeepSeek R1 MLA dimensions "
                "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)"
            )

        return invalid_reasons

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
        device: torch.device,
        layer_names: list[str] | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config
        self.device = device
        self.layer_names = layer_names

    def prepare_metadata(  # noqa: B027
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
    ) -> None:
        """Prepare backend-specific metadata before the forward pass.

        Called by the metadata builder after constructing the prefill metadata.
        """
        self._prefill_metadata = prefill_metadata

    @abstractmethod
    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
