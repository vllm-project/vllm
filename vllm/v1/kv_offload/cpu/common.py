# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, ConfigInfo


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
    """Static, per-engine facts about the CPU offload tier.

    CPUOffloadingManager fills the values through
    OffloadingManager.config_info(), and CPUOffloadingSpec declares the names
    through OffloadingSpec.config_info_keys(). Document every field in
    docs/features/kv_offloading_usage.md.

    """

    key_prefix: ClassVar[str] = "cpu_"

    # Chunk slots in the tier. Chunks, not GPU blocks. See blocks_per_chunk.
    num_chunks: int
    # GPU blocks for each chunk: the chunk-to-block conversion factor.
    blocks_per_chunk: int
    # Page-aligned bytes of one chunk, or None from a caller that does not
    # report it. With num_chunks it gives the exact size of the tier in bytes,
    # the only capacity that holds for every model shape.
    kv_bytes_per_chunk: int | None
    # Upper bound on the KV tokens the tier holds, over the request lengths up
    # to max_model_len. None when max_model_len is not known. See
    # CPUOffloadingManager._capacity_tokens_at_max_len.
    capacity_tokens_at_max_len: int | None
