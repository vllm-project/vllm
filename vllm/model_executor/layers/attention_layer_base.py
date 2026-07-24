# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from vllm.v1.kv_cache_interface import KVCacheSpec


class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True
    # Specialized layers may consume complete history on every query rank or
    # perform their own partial-output reduction. Such layers do not require
    # their backend to expose LSE to the generic DCP reducer.
    handles_dcp_decode_internally: bool = False

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    def bind_pcp_peer_kv_cache(
        self,
        peer_kv_cache: torch.Tensor,
        fence: Callable[[], None],
    ) -> None:
        """Bind the rank-major peer view for owner-history cache writes.

        The ordinary ``kv_cache`` remains the rank-local consumer view. The
        peer view adds a leading PCP-rank dimension and is only present when
        the experimental direct-store path is enabled.
        """
        from vllm.model_executor.layers.attention.pcp_peer_cache import (
            make_rank_major_block_tensor_view,
        )

        self.pcp_peer_kv_cache = peer_kv_cache
        (
            self.pcp_peer_block_kv_cache,
            self.pcp_peer_block_stride,
        ) = make_rank_major_block_tensor_view(peer_kv_cache)
        self.pcp_peer_cache_fence = fence

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
