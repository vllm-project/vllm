# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side capturer and scheduler-side manager for indexer topk indices.

Mirrors the architecture of :mod:`routed_experts_capturer`: the worker
captures per-layer sparse-attention topk indices into a device buffer
during the forward pass, D2H-copies them into a pinned CPU buffer, and
hands them to the scheduler via :class:`IndexerTopkLists`. The scheduler
persists them into a slot-indexed CPU buffer keyed by the physical KV-cache
block layout, and returns the per-request slice when the request finishes.

Key differences from routed_experts:
  - Capture source is :class:`SparseAttnIndexer` (not MoE router).
  - Only indexer layers capture (compact layer dimension, not
    ``num_hidden_layers``).
  - ``topk_size`` is ``index_topk`` (typically 512+), much larger than
    ``num_experts_per_tok`` (4-8).
  - dtype is ``int32`` (KV slot indices can exceed 65535).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer


def _get_index_topk(hf_config) -> int:
    """Resolve ``index_topk`` from the HF config.

    DeepSeek-V32 and V4 store the sparse-attention topk under
    ``index_topk``. Returns 0 when the model has no indexer.
    """
    index_topk = getattr(hf_config, "index_topk", 0)
    return index_topk if isinstance(index_topk, int) else 0


def _get_num_indexer_layers(hf_config) -> int:
    """Count the number of backbone layers that build an indexer.

    Mirrors the skip logic in
    :func:`vllm.model_executor.models.deepseek_v2.DeepseekV32Attention.__init__`:
    when ``index_topk_pattern`` is ``None``, layer ``i`` builds an indexer
    iff ``max(i - index_skip_topk_offset + 1, 0) % index_topk_freq == 0``;
    otherwise the per-layer pattern string ``"S"`` means skip.
    """
    if not hasattr(hf_config, "index_topk"):
        return 0
    num_hidden_layers = hf_config.num_hidden_layers
    if not isinstance(num_hidden_layers, int):
        return 0
    freq = getattr(hf_config, "index_topk_freq", 1)
    pattern = getattr(hf_config, "index_topk_pattern", None)
    skip_offset = getattr(hf_config, "index_skip_topk_offset", 2)

    if not isinstance(freq, int) or freq <= 0:
        return 0
    if pattern is not None and not isinstance(pattern, str):
        return 0
    if pattern is not None:
        # The model only applies the pattern where it has an entry. Layers
        # beyond a short pattern still build an indexer.
        return sum(
            1
            for i in range(num_hidden_layers)
            if i >= len(pattern) or pattern[i] != "S"
        )
    count = 0
    for layer_id in range(num_hidden_layers):
        if max(layer_id - skip_offset + 1, 0) % freq == 0:
            count += 1
    return count


def get_indexer_shape(hf_config) -> tuple[int, int]:
    """Return the canonical ``(num_indexer_layers, index_topk)`` shape."""
    num_indexer_layers = _get_num_indexer_layers(hf_config)
    index_topk = _get_index_topk(hf_config)
    if num_indexer_layers <= 0 or index_topk <= 0:
        raise ValueError(
            "Indexer top-k capture requires positive indexer layer and top-k "
            f"counts, got {num_indexer_layers=}, {index_topk=}."
        )
    return num_indexer_layers, index_topk


def get_sparse_attn_indexers(
    model: torch.nn.Module,
) -> list[SparseAttnIndexer]:
    """Return indexers owned by the target model only."""
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    return [
        module for module in model.modules() if isinstance(module, SparseAttnIndexer)
    ]


def get_indexer_attn_group_id(kv_cache_config: KVCacheConfig) -> int:
    """Return the full-attention KV-cache group used by the indexer."""
    full_attention_kinds = {
        KVCacheSpecKind.FULL_ATTENTION,
        KVCacheSpecKind.MLA_ATTENTION,
        KVCacheSpecKind.SINK_FULL_ATTENTION,
    }
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if get_kv_cache_spec_kind(group.kv_cache_spec) in full_attention_kinds:
            return gid
    raise ValueError(
        "--enable-return-indexer-topk requires a full-attention KV cache group."
    )


class IndexerTopkCapturer:
    """Worker-side capturer for indexer topk indices, lives on GPU.

    :class:`SparseAttnIndexer` calls :meth:`capture` from inside its
    forward pass with the per-layer topk-indices tensor. The tensor is
    written into a preallocated device buffer indexed by a compact
    layer id. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`IndexerTopkLists`.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - Every configured indexer layer overwrites the current step's token
          rows before the buffer is consumed. Rows beyond the current step
          are never read.
        - ``device_buffer.dtype`` is ``torch.int32``.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        num_indexer_layers: int,
        index_topk: int,
        device: str,
    ) -> None:
        self.num_indexer_layers = num_indexer_layers
        self.index_topk = index_topk
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                num_indexer_layers,
                index_topk,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._captured_layers = [False] * num_indexer_layers
        self._enabled = True

    def begin_step(self, enabled: bool = True) -> None:
        """Reset the small per-step callback completeness tracker."""
        self._enabled = enabled
        self._captured_layers[:] = [False] * self.num_indexer_layers

    def capture(self, compact_layer_id: int, topk_indices: torch.Tensor) -> None:
        """Capture topk indices for a specific indexer layer.

        Args:
            compact_layer_id: The compact index (0..num_indexer_layers-1)
                assigned at bind time, NOT the model layer_id.
            topk_indices: Tensor of shape (batch_size, index_topk).
        """
        if not self._enabled:
            return
        if compact_layer_id < 0 or compact_layer_id >= self.device_buffer.shape[1]:
            raise IndexError(
                f"indexer capture layer {compact_layer_id} exceeds buffer "
                f"layer dim {self.device_buffer.shape[1]}"
            )
        if topk_indices.ndim != 2 or topk_indices.shape[1] != self.index_topk:
            raise ValueError(
                "Indexer capture tensor must have shape "
                f"(batch, {self.index_topk}), got {tuple(topk_indices.shape)}."
            )
        batch_size = topk_indices.shape[0]
        if batch_size > self.device_buffer.shape[0]:
            raise ValueError(
                f"Indexer capture batch {batch_size} exceeds buffer capacity "
                f"{self.device_buffer.shape[0]}."
            )
        self.device_buffer[:batch_size, compact_layer_id, :] = topk_indices
        self._captured_layers[compact_layer_id] = True

    def validate_step(self) -> None:
        """Fail closed when a model path skipped an indexer callback."""
        if not self._enabled:
            return
        missing = [
            layer_id
            for layer_id, captured in enumerate(self._captured_layers)
            if not captured
        ]
        if missing:
            raise RuntimeError(
                "Indexer top-k capture missed indexer layers "
                f"{missing}; refusing to return stale device-buffer data."
            )

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer for D2H copy."""
        return self.device_buffer


class IndexerTopkManager:
    """Scheduler-side slot-indexed buffer for indexer topk indices.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block``, tying topk data to
    physical KV-cache blocks (same layout as :class:`RoutedExpertsManager`).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`IndexerTopkLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(topk_data, slot_mapping)``.
      3. On request completion, the scheduler calls :meth:`get` with the
         request's block IDs to recover the full per-token topk.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self.attn_gid = get_indexer_attn_group_id(kv_cache_config)
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        hf_config = vllm_config.model_config.hf_text_config
        self.num_indexer_layers, self.index_topk = get_indexer_shape(hf_config)
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        self.indexer_topk_by_slot = np.zeros(
            (
                max_num_slots,
                self.num_indexer_layers,
                self.index_topk,
            ),
            dtype=np.int32,
        )
        logger.info(
            "IndexerTopkManager CPU buffer: %.2f GB "
            "(slots=%d, indexer_layers=%d, index_topk=%d)",
            self.indexer_topk_by_slot.nbytes / 1e9,
            max_num_slots,
            self.num_indexer_layers,
            self.index_topk,
        )

    def store_batch(self, data: np.ndarray, slot_mapping: np.ndarray) -> None:
        """Persist one step's indexer topk into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``.
        """
        if data.ndim != 3 or data.shape[1:] != (
            self.num_indexer_layers,
            self.index_topk,
        ):
            raise ValueError(
                "Indexer top-k data shape does not match the configured shape: "
                f"got {data.shape}, expected (*, {self.num_indexer_layers}, "
                f"{self.index_topk})."
            )
        if slot_mapping.ndim != 1 or data.shape[0] != slot_mapping.shape[0]:
            raise ValueError(
                "Indexer top-k data and slot mapping must have matching leading "
                f"dimensions, got {data.shape[0]} and {slot_mapping.shape}."
            )
        if slot_mapping.size and (
            np.any(slot_mapping < 0)
            or np.any(slot_mapping >= self.indexer_topk_by_slot.shape[0])
        ):
            raise ValueError("Indexer top-k slot mapping contains an invalid slot.")
        self.indexer_topk_by_slot[slot_mapping] = data

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read indexer topk for a completed / preempted request.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass (typically ``request.num_tokens - 1``).
            token_start: Skip the first ``token_start`` tokens.

        Returns:
            Array of shape (num_tokens - token_start, num_indexer_layers,
            index_topk).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        slot_mapping = slot_mapping[token_start:]
        return self.indexer_topk_by_slot[slot_mapping]
