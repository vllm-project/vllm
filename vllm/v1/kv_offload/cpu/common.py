# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


@dataclass(frozen=True)
class CPUCacheOffloadingInfo:
    """Static, per-engine facts about the CPU offload tier.

    CPUOffloadingManager publishes these facts as info metric labels. See
    OffloadingManager.config_info().
    """

    # Chunk count, not GPU blocks; see blocks_per_chunk.
    num_chunks: int
    # GPU blocks per chunk; the CPU-slot to GPU-block conversion factor.
    blocks_per_chunk: int
    # Page-aligned bytes per chunk. With num_chunks this is the tier's exact
    # size in bytes, the only capacity valid for every model shape.
    kv_bytes_per_chunk: int
    # Upper bound on the KV tokens the tier holds, over the request lengths up
    # to max_model_len. None when max_model_len is not known.
    # See _capacity_tokens_at_max_len.
    capacity_tokens_at_max_len: int | None

    def as_config_info(self) -> Mapping[str, str | int]:
        """Render the facts as info metric labels, one label for each field.

        Every key carries the "cpu_" prefix, so a fact keeps one name whether
        the CPU cache runs alone or as the primary tier of a tiering manager.
        An unknown token capacity reads "None", because a label value holds a
        string, and a dropped label would change the label set.
        """
        return {
            f"cpu_{name}": "None" if value is None else value
            for name, value in asdict(self).items()
        }


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing KV chunks to/from CPU memory.

    The inherited block_ids field holds chunk indices into the
    CPU cache (not GPU block IDs). The chunk_ids alias exposes
    the same array under the name used by the tiering layer.
    """

    @property
    def chunk_ids(self) -> np.ndarray:
        return self.block_ids
