# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts slot buffer."""

import contextlib
import logging
import numpy as np

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.routed_experts_capture.common import (
    get_num_experts_per_token,
    get_routing_slot_shape_and_dtype,
    require_full_attn_group_id,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    SharedRoutingRegion,
    shared_routing_mmap_path,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)


class RoutedExpertsManager:
    """Scheduler-side slot buffer for routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self.full_attn_group_id = require_full_attn_group_id(kv_cache_config)
        full_attn_group = kv_cache_config.kv_cache_groups[self.full_attn_group_id]
        self.block_size = full_attn_group.kv_cache_spec.block_size

        hf_config = vllm_config.model_config.hf_text_config
        moe_top_k = get_num_experts_per_token(hf_config)
        self.num_layers = hf_config.num_hidden_layers
        self.moe_top_k = moe_top_k
        # The worker writer derives its mmap from the same helper, so scheduler
        # and worker always agree on the shared /dev/shm buffer layout.
        slot_shape, slot_dtype = get_routing_slot_shape_and_dtype(
            vllm_config, kv_cache_config
        )
        self.expert_id_dtype = np.dtype(slot_dtype)
        self._slot_region = SharedRoutingRegion(
            path=shared_routing_mmap_path(
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
            ),
            shape=slot_shape,
            dtype=slot_dtype,
        )
        self.routed_experts_by_slot = self._slot_region.array
        self._blocks_view = self.routed_experts_by_slot.reshape(
            kv_cache_config.num_blocks,
            self.block_size,
            self.num_layers,
            moe_top_k,
        )
        logger.info(
            "RoutedExpertsManager slot buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            slot_shape[0],
            self.num_layers,
            moe_top_k,
            self.routed_experts_by_slot.dtype.name,
        )

    def shutdown(self) -> None:
        """Release the shared slot mmap."""
        region = getattr(self, "_slot_region", None)
        if region is not None:
            # Drop the ndarray view before closing the mmap it is backed by.
            self.routed_experts_by_slot = None  # type: ignore[assignment]
            self._blocks_view = None  # type: ignore[assignment]
            region.close()
            self._slot_region = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()

    def get(
        self,
        block_ids: list[int],
        token_end: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed-experts rows for a request token range.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            token_end: Exclusive end offset of the request token range.
            token_start: Inclusive start offset of the request token range.

        Returns:
            Array of shape (token_end - token_start, num_layers,
            moe_top_k).
        """
        block_size = self.block_size
        block_ids_array = np.asarray(block_ids, dtype=np.int64)
        token_positions = np.arange(token_start, token_end)
        slot_mapping = (
            block_ids_array[token_positions // block_size] * block_size
            + token_positions % block_size
        )
        return self.routed_experts_by_slot[slot_mapping]

    def get_by_slots(self, slots: np.ndarray) -> np.ndarray:
        """Read routing for explicit slot indices (decode path)."""
        return self.routed_experts_by_slot[slots]
