# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from vllm.v1.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True

    def get_kv_cache_bundle(self) -> tuple["AttentionLayerBase", ...] | None:
        """Components whose KV updates are replicated across TP ranks.

        Returning None leaves this layer ineligible for layer-sharded storage.
        Implementations list actual cache modules, including auxiliary caches.
        """
        return None

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
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
