# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract base classes for MLA prefill backends.

This module defines the interface for MLA prefill backends, enabling
priority-based selection similar to how MLA decode backends work.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class MLAPrefillBuilderState:
    """State created by a prefill backend for use during metadata building.

    This class holds backend-specific resources (workspaces, wrappers, etc.)
    that persist across metadata build calls. Backends can subclass this
    to add their own state.
    """

    # Common state that may be used by multiple backends
    workspace_buffer: torch.Tensor | None = None

    # Generic storage for backend-specific state
    backend_state: dict[str, Any] = field(default_factory=dict)


class MLAPrefillBackend(ABC):
    """Abstract base class for MLA prefill backends.

    Each prefill backend declares its capabilities (supported dtypes,
    compute capabilities, etc.) and provides a factory method for
    creating the implementation class.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    requires_r1_mla_dimensions: ClassVar[bool] = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the name of this prefill backend."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_prefill_impl_cls() -> type["MLAPrefillImpl"]:
        """Return the implementation class for this prefill backend."""
        raise NotImplementedError

    @staticmethod
    def get_prefill_metadata_cls() -> type["MLACommonPrefillMetadata"]:
        """Return the metadata class for this prefill backend.

        Override this method if the backend requires a specialized
        metadata class (e.g., FlashInferPrefillMetadata).
        """
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonPrefillMetadata,
        )

        return MLACommonPrefillMetadata

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        """Check if this backend supports the given compute capability.

        Override this method if the backend has specific hardware requirements.
        """
        return True

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """Check if this backend supports the given dtype."""
        return dtype in cls.supported_dtypes

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend's dependencies are available.

        Override this method to check for required libraries/imports.
        """
        return True

    @classmethod
    def validate_configuration(
        cls,
        device_capability: "DeviceCapability",
        dtype: torch.dtype,
        vllm_config: "VllmConfig",
    ) -> list[str]:
        """Validate if this backend can be used with the given configuration.

        Returns:
            A list of invalid reasons. Empty list if configuration is valid.
        """
        from vllm.v1.attention.backends.mla.prefill.selector import (
            is_deepseek_r1_mla_compatible,
        )

        invalid_reasons: list[str] = []

        if not cls.supports_compute_capability(device_capability):
            major, minor = device_capability.major, device_capability.minor
            invalid_reasons.append(f"compute capability {major}.{minor} not supported")

        if not cls.supports_dtype(dtype):
            invalid_reasons.append(f"dtype {dtype} not supported")

        if not cls.is_available():
            invalid_reasons.append("required dependencies not available")

        if cls.requires_r1_mla_dimensions and not is_deepseek_r1_mla_compatible(
            vllm_config
        ):
            invalid_reasons.append(
                "model does not have DeepSeek R1 MLA dimensions "
                "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)"
            )

        return invalid_reasons

    @classmethod
    def create_builder_state(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        device: torch.device,
    ) -> MLAPrefillBuilderState:
        """Create backend-specific state for the metadata builder.

        This is called once when the metadata builder is initialized.
        Override to allocate workspaces, create wrappers, etc.

        Args:
            vllm_config: The vLLM configuration.
            kv_cache_spec: The attention specification.
            layer_names: Names of attention layers.
            device: The device to allocate tensors on.

        Returns:
            A state object containing backend-specific resources.
        """
        return MLAPrefillBuilderState()

    @staticmethod
    def get_chunked_context_metadata_cls() -> type:
        """Return the ChunkedContextMetadata class for this backend.

        Override if the backend needs a specialized ChunkedContextMetadata.
        """
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonPrefillMetadata,
        )

        return MLACommonPrefillMetadata.ChunkedContextMetadata

    @classmethod  # noqa: B027
    def post_process_prefill_metadata(
        cls,
        prefill_metadata: "MLACommonPrefillMetadata",
        builder_state: MLAPrefillBuilderState,
        prefill_query_start_loc: torch.Tensor,
    ) -> None:
        """Post-process the prefill metadata after creation.

        This is called after the prefill metadata is created but before
        it's attached to the attention metadata. Use this to set
        backend-specific fields on the metadata.
        """
        pass

    @classmethod  # noqa: B027
    def finalize_attention_metadata(
        cls,
        attn_metadata: Any,
        builder_state: MLAPrefillBuilderState,
        num_prefills: int,
        num_heads: int,
        kv_cache_spec: "AttentionSpec",
        mla_dims: Any,
        model_config: Any,
    ) -> None:
        """Finalize the attention metadata after all components are built.

        This is called after the full attention metadata is constructed.
        Use this for any final processing (e.g., building FlashInfer wrappers).
        """
        pass


class MLAPrefillImpl(ABC):
    """Abstract base class for MLA prefill implementations.

    Each implementation provides the actual prefill attention computation
    for new tokens (causal) and context chunks (non-causal).
    """

    # Whether this backend needs to pad V to match Q/K head dim
    requires_v_padding: bool = True

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
    ) -> None:
        """Initialize the prefill implementation.

        Args:
            num_heads: Number of attention heads.
            scale: Softmax scale factor.
            kv_lora_rank: Latent dimension for KV.
            qk_nope_head_dim: QK head dimension without RoPE.
            qk_rope_head_dim: QK head dimension with RoPE.
            v_head_dim: Value head dimension.
            vllm_config: vLLM configuration.
            device: Device to use for computation.
        """
        self.num_heads = num_heads
        self.scale = scale
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.vllm_config = vllm_config
        self.device = device

    @abstractmethod
    def run_prefill_new_tokens(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run prefill attention for new tokens (causal).

        Args:
            prefill_metadata: Metadata for the prefill operation.
            q: Query tensor of shape [num_tokens, num_heads, qk_head_dim].
            k: Key tensor of shape [num_tokens, num_heads, qk_head_dim].
            v: Value tensor of shape [num_tokens, num_heads, v_head_dim].
            return_softmax_lse: Whether to return log-sum-exp values.

        Returns:
            If return_softmax_lse is False:
                Output tensor of shape [num_tokens, num_heads, v_head_dim].
            If return_softmax_lse is True:
                Tuple of (output, lse) where lse has shape [num_heads, num_tokens].
        """
        raise NotImplementedError

    @abstractmethod
    def run_prefill_context_chunk(
        self,
        prefill_metadata: "MLACommonPrefillMetadata",
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run prefill attention for context chunks (non-causal).

        This is used for chunked prefill where we process cached context
        in chunks to manage memory usage.

        Args:
            prefill_metadata: Metadata for the prefill operation.
            chunk_idx: Index of the current context chunk.
            q: Query tensor of shape [num_tokens, num_heads, qk_head_dim].
            k: Key tensor of shape [chunk_tokens, num_heads, qk_head_dim].
            v: Value tensor of shape [chunk_tokens, num_heads, v_head_dim].

        Returns:
            Tuple of (output, lse) where:
                output has shape [num_tokens, num_heads, v_head_dim]
                lse has shape [num_heads, num_tokens]
        """
        raise NotImplementedError
