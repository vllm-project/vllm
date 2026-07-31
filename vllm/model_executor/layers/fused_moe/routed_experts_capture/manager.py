# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts buffers and block maps."""

import contextlib
import logging
import mmap

import numpy as np
import numpy.typing as npt

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.sidecar import (
    KVConnectorSidecarBlockMap,
    KVConnectorSidecarConfig,
    KVConnectorSidecarTransferPlan,
)
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


def _allocate_zeroed_mmap(shape: tuple[int, ...], dtype: npt.DTypeLike) -> np.ndarray:
    """Allocate a demand-paged zero ndarray backed by anonymous mmap."""
    num_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if num_bytes == 0:
        return np.zeros(shape, dtype=dtype)
    mmap_buffer = mmap.mmap(-1, num_bytes)
    return np.frombuffer(mmap_buffer, dtype=dtype).reshape(shape)


class RoutedExpertsManager:
    """Scheduler-side slot and offload buffers for routed experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sidecar_config: KVConnectorSidecarConfig | None = None,
    ) -> None:
        if (
            sidecar_config is not None
            and min(
                sidecar_config.num_connector_blocks,
                sidecar_config.blocks_per_connector_block,
            )
            <= 0
        ):
            raise ValueError(
                "routed-experts sidecar block counts must be positive, got "
                f"{sidecar_config}"
            )

        self.full_attn_group_id = require_full_attn_group_id(kv_cache_config)
        full_attn_group = kv_cache_config.kv_cache_groups[self.full_attn_group_id]
        self.block_size = full_attn_group.kv_cache_spec.block_size
        self.block_size_factor = (
            sidecar_config.blocks_per_connector_block
            if sidecar_config is not None
            else 1
        )

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
        slot_region = SharedRoutingRegion(
            path=shared_routing_mmap_path(vllm_config.instance_id),
            shape=slot_shape,
            dtype=slot_dtype,
        )
        self._slot_region: SharedRoutingRegion | None = slot_region
        self.routed_experts_by_slot = slot_region.array
        self._blocks_view = self.routed_experts_by_slot.reshape(
            kv_cache_config.num_blocks,
            self.block_size,
            self.num_layers,
            moe_top_k,
        )
        # Indexed by offloaded block id, then sub-block within that block.
        self.routed_experts_by_offload_block: np.ndarray | None = None
        if sidecar_config is not None:
            self.routed_experts_by_offload_block = _allocate_zeroed_mmap(
                (
                    sidecar_config.num_connector_blocks,
                    self.block_size_factor,
                    self.block_size,
                    self.num_layers,
                    moe_top_k,
                ),
                dtype=self.expert_id_dtype,
            )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s), "
            "offloaded routed experts: %.2f GB "
            "(cpu_blocks=%s, block_size_factor=%d)",
            self.routed_experts_by_slot.nbytes / 1e9,
            slot_shape[0],
            self.num_layers,
            moe_top_k,
            self.routed_experts_by_slot.dtype.name,
            self.routed_experts_by_offload_block.nbytes / 1e9
            if self.routed_experts_by_offload_block is not None
            else 0.0,
            sidecar_config.num_connector_blocks if sidecar_config is not None else None,
            self.block_size_factor,
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

    def _get_offload_blocks(self) -> np.ndarray:
        """Return the offloaded-block buffer, or raise if absent."""
        if self.routed_experts_by_offload_block is None:
            raise RuntimeError(
                "routed-experts offload buffer is not initialized "
                "but a KV offload transfer was observed"
            )
        return self.routed_experts_by_offload_block

    def store_to_offload_blocks(self, block_map: KVConnectorSidecarBlockMap) -> None:
        """Copy GPU block rows to offloaded sub-block rows."""
        offload_blocks = self._get_offload_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        offload_blocks[
            block_map.connector_block_ids,
            block_map.connector_block_offsets,
        ] = self._blocks_view[block_map.gpu_block_ids]

    def load_from_offload_blocks(self, block_map: KVConnectorSidecarBlockMap) -> None:
        """Copy offloaded sub-block rows to GPU block rows."""
        offload_blocks = self._get_offload_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        self._blocks_view[block_map.gpu_block_ids] = offload_blocks[
            block_map.connector_block_ids,
            block_map.connector_block_offsets,
        ]

    def apply_offload_transfers(
        self, transfers: KVConnectorSidecarTransferPlan
    ) -> None:
        """Move routing rows according to a connector's public block mapping.

        Runs after the worker writes this step's slots and before request
        outputs read routing back. Stores are written as soon as prepare_store
        assigns block ids; loads stay gated by KV complete_store.
        """
        if transfers.load is not None:
            self.load_from_offload_blocks(transfers.load)
        if transfers.store is not None:
            self.store_to_offload_blocks(transfers.store)

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
