# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from vllm.v1.kv_cache_interface import KVCacheLayerConfig


class AttentionLayerBase(ABC):
    """Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True
    # Whether quant configs give this layer a KV-cache quant method. Set by the
    # layers that call `_init_kv_cache_quant`: `Attention`, `MLAAttention`,
    # Kimi's `MultiHeadLatentAttention` and `MiniMaxM3SparseAttention`. Mamba
    # states, indexer caches and other layers keep the default.
    supports_kv_cache_quant: bool = False
    # Set by the speculator on its draft model's layers.
    is_draft_layer: bool = False
    # Name of the layer whose KV cache this layer reads instead of owning one.
    kv_sharing_target_layer_name: str | None = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig | None:
        """Get the KV cache config for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
