# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base class for MLA prefill backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.prefill.selector import (
        MLAPrefillSelectorConfig,
    )


@dataclass(frozen=True, kw_only=True)
class MLADimensions:
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def __str__(self) -> str:
        return (
            f"(qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"qk_rope_head_dim={self.qk_rope_head_dim}, "
            f"v_head_dim={self.v_head_dim})"
        )


class MLAPrefillBackend(ABC):
    """Abstract base class for MLA prefill backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_mla_dimensions: ClassVar[list[MLADimensions]] = []

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

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        """Whether `run_prefill_new_tokens` can write quantized output
        directly (fused) for the given quant key, skipping the post-quant
        pass. Overridden by backends that support it."""
        return False

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

        if (
            cls.supported_mla_dimensions
            and selector_config.mla_dimensions not in cls.supported_mla_dimensions
        ):
            supported = ", ".join(str(dims) for dims in cls.supported_mla_dimensions)
            invalid_reasons.append(
                "Model does not have supported MLA dimensions "
                f"(got {selector_config.mla_dimensions}; supported: {supported})"
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
    ) -> None:
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config

    def clone(self) -> "MLAPrefillBackend":
        return self.__class__(
            num_heads=self.num_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=self.vllm_config,
        )

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
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
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
