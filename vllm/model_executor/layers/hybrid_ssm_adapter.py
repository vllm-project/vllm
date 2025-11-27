#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec


class HybridSSMAdapter(nn.Module, AttentionLayerBase):
    """
    History branch based on Mamba-style SSM state.

    This module exposes a minimal interface expected by the v1 KV cache /
    attention stack:

    - It behaves like an ``AttentionLayerBase`` so it can obtain its own
      ``MambaSpec`` KV pool (managed by ``MambaManager``).
    - It provides helper methods that the hybrid attention backend can call to
      obtain an SSM contribution over the same flattened token set as the
      sliding-window attention output.

    The current implementation focuses on wiring and KV-spec integration.
    The actual SSM compute path intentionally reuses the metadata layout of
    Mamba-1 (``Mamba1AttentionMetadata``) but returns a zero contribution for
    now. This keeps the feature opt‑in and avoids touching any CUDA kernels,
    while providing a scaffold to plug in the full Mamba pipeline later.
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        *,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.intermediate_size = intermediate_size

        # These are cached so that get_state_shape/get_state_dtype can be
        # computed consistently with existing Mamba layers.
        self.model_config = model_config
        self.cache_config = cache_config

        # Layer name used by vLLM's compilation / forward context.
        self.layer_name = prefix

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Placeholder KV cache – this will be replaced by MambaManager via
        # bind_kv_cache() once the v1 engine initializes the cache tensors.
        self.kv_cache: tuple[torch.Tensor, ...] = (
            torch.tensor([]),
            torch.tensor([]),
        )

    # ------------------------------------------------------------------
    # KV cache spec / Mamba state description
    # ------------------------------------------------------------------
    def get_state_shape(self) -> Iterable[tuple[int, ...]]:
        """Return the logical shapes of the Mamba SSM state tensors.

        This mirrors ``MambaMixer.get_state_shape`` by delegating to
        ``MambaStateShapeCalculator`` so that the adapter can share the same
        ``MambaSpec`` / ``MambaManager`` infrastructure.
        """
        tp_world_size = get_tensor_model_parallel_world_size()
        return MambaStateShapeCalculator.mamba1_state_shape(
            tp_world_size=tp_world_size,
            intermediate_size=self.intermediate_size,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        """Return the dtypes of the Mamba SSM state tensors.

        The adapter mirrors the dtype choices of the Mamba-1 implementation,
        driven by the model and cache configuration.
        """
        # Defer to the runtime vLLM config if explicit configs were not
        # provided at construction time. This keeps the adapter usable from
        # simple unit tests where a full ``ModelConfig`` is not wired yet.
        model_config: ModelConfig
        cache_config: CacheConfig
        if self.model_config is None or self.cache_config is None:
            vllm_config = get_current_vllm_config()
            model_config = vllm_config.model_config
            cache_config = vllm_config.cache_config
        else:
            model_config = self.model_config
            cache_config = self.cache_config

        return MambaStateDtypeCalculator.mamba1_state_dtype(
            model_config.dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )

    # ------------------------------------------------------------------
    # AttentionLayerBase integration
    # ------------------------------------------------------------------
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Use the existing Mamba-1 backend for KV grouping / metadata."""
        return Mamba1AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Expose a ``MambaSpec`` so the adapter obtains its own KV pool.

        This allows the v1 KV cache manager to allocate a dedicated Mamba
        state pool (managed by ``MambaManager``) alongside standard
        sliding-window KV pages for attention.
        """
        # Follow the same speculative decoding constraints as MambaBase.
        if (
            vllm_config.speculative_config is not None
            and vllm_config.model_config.hf_config.model_type not in ["qwen3_next"]
        ):
            raise NotImplementedError(
                "Hybrid SSM adapter with speculative decoding is not supported yet."
            )

        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded

        return MambaSpec(
            shapes=tuple(self.get_state_shape()),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type="mamba1",
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )

    # ------------------------------------------------------------------
    # History-branch API used by HybridAttentionImpl
    # ------------------------------------------------------------------
    def _get_mamba_attn_metadata(self) -> Any | None:
        """Fetch the Mamba1AttentionMetadata for this adapter, if present."""
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            return attn_metadata.get(self.layer_name, None)
        return None

    def forward_history_branch_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        """History branch for prefill tokens.

        For now this method returns a zero contribution while ensuring that
        the tensor is correctly shaped and indexed over the same flattened
        token set as the sliding-window attention output.
        """
        if attn_metadata is None:
            attn_metadata = self._get_mamba_attn_metadata()

        if attn_metadata is None:
            # Profiling / shape-only runs: match the input shape.
            return torch.zeros_like(hidden_states)

        num_actual_tokens: int = getattr(attn_metadata, "num_prefill_tokens", 0)
        if num_actual_tokens <= 0:
            return torch.zeros_like(hidden_states)

        ssm_out = torch.zeros_like(hidden_states)
        return ssm_out

    def forward_history_branch_decode(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Any | None = None,
    ) -> torch.Tensor:
        """History branch for decode tokens.

        The adapter is expected to produce an SSM contribution aligned with the
        flattened decode token set. The current implementation produces a zero
        tensor but wires in the same metadata shape as Mamba-1 so that a
        future implementation can swap in the full Mamba pipeline without
        changing call sites.
        """
        if attn_metadata is None:
            attn_metadata = self._get_mamba_attn_metadata()

        if attn_metadata is None:
            # Profiling / shape-only runs: match the input shape.
            return torch.zeros_like(hidden_states)

        num_actual_tokens: int = getattr(attn_metadata, "num_decode_tokens", 0)
        if num_actual_tokens <= 0:
            return torch.zeros_like(hidden_states)

        ssm_out = torch.zeros_like(hidden_states)
        return ssm_out


