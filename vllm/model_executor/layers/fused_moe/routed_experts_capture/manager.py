# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side routed-experts buffers and block maps."""

import contextlib
import logging
import mmap
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

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

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )

logger = logging.getLogger(__name__)


class FullAttnBlockMap(NamedTuple):
    """Maps anchor-group GPU blocks to their offloaded sub-block slots."""

    gpu_block_ids: np.ndarray  # GPU block id per moved block
    cpu_block_ids: np.ndarray  # offloaded block holding that block
    sub_offsets: np.ndarray  # sub-block index within the offloaded block

    @classmethod
    def concatenate(cls, maps: list["FullAttnBlockMap"]) -> "FullAttnBlockMap":
        """Merge per-job maps so all blocks move in one vectorized copy."""
        return cls(
            gpu_block_ids=np.concatenate(
                [block_map.gpu_block_ids for block_map in maps]
            ),
            cpu_block_ids=np.concatenate(
                [block_map.cpu_block_ids for block_map in maps]
            ),
            sub_offsets=np.concatenate([block_map.sub_offsets for block_map in maps]),
        )


def _divide_round_up(dividend: int, divisor: int) -> int:
    """Ceiling division of non-negative integers."""
    return -(-dividend // divisor)


def compute_full_attn_block_map(
    gpu_block_ids: np.ndarray,
    cpu_block_ids: np.ndarray,
    group_sizes: Sequence[int],
    block_indices: Sequence[int],
    full_attn_group_id: int,
    block_size_factor: int,
    expected_num_groups: int | None = None,
) -> FullAttnBlockMap:
    """Map a KV transfer job's anchor-group blocks to offload rows.

    Args:
        gpu_block_ids: Group-major GPU block ids for the whole job.
        cpu_block_ids: Group-major offloaded block ids for the whole job.
        group_sizes: GPU block count per KV cache group.
        block_indices: Logical block index in GPU blocks of each group's
            first block.
        full_attn_group_id: Full-attention anchor group index.
        block_size_factor: GPU blocks per offloaded block.
        expected_num_groups: If set, the KV-group count the job must span;
            mismatch signals a contract break.

    Returns:
        FullAttnBlockMap covering only the anchor group.

    Raises:
        RuntimeError: If the group-major flat-order contract is violated.
    """
    offloaded_blocks_per_group = [
        _divide_round_up(
            int(group_size) + int(block_indices[group_id]) % block_size_factor,
            block_size_factor,
        )
        for group_id, group_size in enumerate(group_sizes)
    ]
    if (
        (expected_num_groups is not None and len(group_sizes) != expected_num_groups)
        or sum(group_sizes) != len(gpu_block_ids)
        or sum(offloaded_blocks_per_group) != len(cpu_block_ids)
    ):
        raise RuntimeError(
            "routed-experts offload transfer violates the group-major "
            f"flat-order contract: group_sizes={list(group_sizes)}, "
            f"block_indices={list(block_indices)}, "
            f"full_attn_group_id={full_attn_group_id}, "
            f"block_size_factor={block_size_factor}, "
            f"len(gpu)={len(gpu_block_ids)}, "
            f"len(cpu)={len(cpu_block_ids)}, "
            f"expected_cpu={sum(offloaded_blocks_per_group)}, "
            f"expected_num_groups={expected_num_groups}"
        )

    gpu_block_offset = int(sum(group_sizes[:full_attn_group_id]))
    num_gpu_blocks = int(group_sizes[full_attn_group_id])
    anchor_gpu_block_ids = np.asarray(
        gpu_block_ids[gpu_block_offset : gpu_block_offset + num_gpu_blocks]
    )

    if num_gpu_blocks == 0:
        empty_block_ids = np.empty(0, dtype=np.int64)
        return FullAttnBlockMap(
            empty_block_ids, empty_block_ids, empty_block_ids.copy()
        )

    cpu_block_offset = sum(offloaded_blocks_per_group[:full_attn_group_id])
    first_sub_block_offset = int(block_indices[full_attn_group_id]) % block_size_factor
    block_positions = np.arange(num_gpu_blocks, dtype=np.int64)
    sub_offsets = (first_sub_block_offset + block_positions) % block_size_factor
    local_cpu_block_indices = (
        first_sub_block_offset + block_positions
    ) // block_size_factor
    anchor_cpu_block_ids = np.asarray(cpu_block_ids)[
        cpu_block_offset + local_cpu_block_indices
    ]
    return FullAttnBlockMap(anchor_gpu_block_ids, anchor_cpu_block_ids, sub_offsets)


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
        num_offload_blocks: int | None = None,
        block_size_factor: int = 1,
    ) -> None:
        self.full_attn_group_id = require_full_attn_group_id(kv_cache_config)
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        full_attn_group = kv_cache_config.kv_cache_groups[self.full_attn_group_id]
        self.block_size = full_attn_group.kv_cache_spec.block_size
        if block_size_factor < 1:
            raise ValueError(f"block_size_factor must be >= 1, got {block_size_factor}")
        self.block_size_factor = block_size_factor

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
            path=shared_routing_mmap_path(
                vllm_config.instance_id,
                vllm_config.parallel_config.data_parallel_rank,
            ),
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
        if num_offload_blocks is not None:
            self.routed_experts_by_offload_block = _allocate_zeroed_mmap(
                (
                    num_offload_blocks,
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
            num_offload_blocks,
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

    def store_to_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Copy GPU block rows to offloaded sub-block rows."""
        offload_blocks = self._get_offload_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        offload_blocks[block_map.cpu_block_ids, block_map.sub_offsets] = (
            self._blocks_view[block_map.gpu_block_ids]
        )

    def load_from_offload_blocks(self, block_map: FullAttnBlockMap) -> None:
        """Copy offloaded sub-block rows to GPU block rows."""
        offload_blocks = self._get_offload_blocks()
        if len(block_map.gpu_block_ids) == 0:
            return
        self._blocks_view[block_map.gpu_block_ids] = offload_blocks[
            block_map.cpu_block_ids, block_map.sub_offsets
        ]

    def _compute_full_attn_block_map(
        self, gpu_spec: object, cpu_spec: object
    ) -> FullAttnBlockMap:
        """Map one KV offload transfer job to the routed-experts block map."""
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        if not isinstance(gpu_spec, GPULoadStoreSpec):
            raise RuntimeError(
                f"expected GPULoadStoreSpec, got {type(gpu_spec).__name__}"
            )
        if not isinstance(cpu_spec, CPULoadStoreSpec):
            raise RuntimeError(
                f"expected CPULoadStoreSpec, got {type(cpu_spec).__name__}"
            )
        return compute_full_attn_block_map(
            gpu_block_ids=gpu_spec.block_ids,
            cpu_block_ids=cpu_spec.block_ids,
            group_sizes=gpu_spec.group_sizes,
            block_indices=gpu_spec.block_indices,
            full_attn_group_id=self.full_attn_group_id,
            block_size_factor=self.block_size_factor,
            expected_num_groups=self.num_kv_cache_groups,
        )

    def apply_offload_transfers(self, metadata: "OffloadingConnectorMetadata") -> None:
        """Store/load offloaded routing alongside this step's KV offload jobs.

        Runs after the worker writes this step's slots and before request
        outputs read routing back. Stores are written as soon as prepare_store
        assigns block ids; loads stay gated by KV complete_store.
        """
        # Batch every job into one fancy-index per direction. Under heavy
        # offload the per-job numpy call overhead dominates the scheduler
        # thread, while the data volume moved is the same. Empty maps are
        # dropped so concatenate() never sees a zero-length job.
        load_block_maps: list[FullAttnBlockMap] = []
        for job in metadata.load_jobs.values():
            source_spec, destination_spec = job.src_spec, job.dst_spec
            block_map = self._compute_full_attn_block_map(destination_spec, source_spec)
            if len(block_map.gpu_block_ids):
                load_block_maps.append(block_map)
        if load_block_maps:
            self.load_from_offload_blocks(FullAttnBlockMap.concatenate(load_block_maps))
        store_block_maps: list[FullAttnBlockMap] = []
        for job in metadata.store_jobs.values():
            source_spec, destination_spec = job.src_spec, job.dst_spec
            block_map = self._compute_full_attn_block_map(source_spec, destination_spec)
            if len(block_map.gpu_block_ids):
                store_block_maps.append(block_map)
        if store_block_maps:
            self.store_to_offload_blocks(FullAttnBlockMap.concatenate(store_block_maps))

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
