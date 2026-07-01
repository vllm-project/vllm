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

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig

logger = logging.getLogger(__name__)


def _get_index_topk(hf_config) -> int:
    """Resolve ``index_topk`` from the HF config.

    DeepSeek-V32 and V4 store the sparse-attention topk under
    ``index_topk``. Returns 0 when the model has no indexer.
    """
    return getattr(hf_config, "index_topk", 0)


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
    freq = getattr(hf_config, "index_topk_freq", 1)
    pattern = getattr(hf_config, "index_topk_pattern", None)
    skip_offset = getattr(hf_config, "index_skip_topk_offset", 2)

    if pattern is not None:
        return sum(
            1 for i in range(min(len(pattern), num_hidden_layers)) if pattern[i] != "S"
        )
    count = 0
    for layer_id in range(num_hidden_layers):
        if max(layer_id - skip_offset + 1, 0) % freq == 0:
            count += 1
    return count


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
        - :meth:`clear_buffer` is called at the start of every step, so
          unused slots stay zero.
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

    def capture(self, compact_layer_id: int, topk_indices: torch.Tensor) -> None:
        """Capture topk indices for a specific indexer layer.

        Args:
            compact_layer_id: The compact index (0..num_indexer_layers-1)
                assigned at bind time, NOT the model layer_id.
            topk_indices: Tensor of shape (batch_size, index_topk).
        """
        batch_size = topk_indices.shape[0]
        if compact_layer_id >= self.device_buffer.shape[1]:
            return
        self.device_buffer[:batch_size, compact_layer_id, :] = topk_indices

    def clear_buffer(self) -> None:
        """Zero the device buffer. Called at the start of every step."""
        self.device_buffer.zero_()

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
        self.attn_gid = next(
            gid
            for gid, g in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, FullAttentionSpec)
        )
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        hf_config = vllm_config.model_config.hf_text_config
        self.num_indexer_layers = _get_num_indexer_layers(hf_config)
        self.index_topk = _get_index_topk(hf_config)
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
