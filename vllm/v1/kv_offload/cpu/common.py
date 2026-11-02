# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, ConfigInfo
from vllm.v1.kv_offload.config import OffloadingConfig


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing KV chunks to/from CPU memory.

    The inherited block_ids field holds chunk indices into the
    CPU cache (not GPU block IDs). The chunk_ids alias exposes
    the same array under the name used by the tiering layer.
    """

    @property
    def chunk_ids(self) -> np.ndarray:
        return self.block_ids


@dataclass(frozen=True)
class CPUOffloadingInfo(ConfigInfo):
    """Static facts about the CPU offload tier."""

    # Chunk slots in the tier. Chunks, not GPU blocks.
    cpu_num_chunks: int
    # GPU blocks for each chunk.
    cpu_blocks_per_chunk: int
    # Size of the tier in bytes, including chunk padding.
    cpu_kv_cache_size_bytes: int
    # KV tokens the tier holds for requests of max_model_len tokens. None
    # when max_model_len is not known.
    cpu_kv_cache_size_tokens: int | None

    @classmethod
    def from_config(
        cls, config: OffloadingConfig, num_chunks: int, kv_bytes_per_chunk: int
    ) -> "CPUOffloadingInfo":
        return cls(
            cpu_num_chunks=num_chunks,
            cpu_blocks_per_chunk=config.cache.blocks_per_chunk,
            cpu_kv_cache_size_bytes=num_chunks * kv_bytes_per_chunk,
            cpu_kv_cache_size_tokens=_kv_cache_size_tokens(config, num_chunks),
        )


def _kv_cache_size_tokens(config: OffloadingConfig, num_chunks: int) -> int | None:
    """KV tokens the tier holds at max_model_len, like GPU kv_cache_size_tokens."""
    max_model_len = config.model.max_model_len
    if max_model_len <= 0 or not config.groups:
        return None
    chunks_per_request = 0
    for group in config.groups:
        group_chunks = cdiv(
            max_model_len, group.tokens_per_block * config.cache.blocks_per_chunk
        )
        if group.sliding_window_size_in_chunks is not None:
            group_chunks = min(group_chunks, group.sliding_window_size_in_chunks)
        chunks_per_request += group_chunks
    return num_chunks * max_model_len // chunks_per_request
