# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


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
class CPUOffloadingInfo:
    """Static, per-engine facts about the CPU offload tier.

    CPUOffloadingManager fills the values through
    OffloadingManager.config_info(), and CPUOffloadingSpec declares the names
    through OffloadingSpec.config_info_keys(). Document every field in
    docs/features/kv_offloading_usage.md.

    """

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

    @classmethod
    def config_info_keys(cls) -> tuple[str, ...]:
        """Return one label name for each field, in field order."""
        return tuple(f"cpu_{field.name}" for field in fields(cls))

    def as_config_info(self) -> Mapping[str, str | int]:
        """Return the label values, under the names of config_info_keys().

        Returns "None" for an unknown fact. Done so the label will not be
        dropped, as an empty label would be.
        """
        values = (getattr(self, field.name) for field in fields(self))
        return {
            key: "None" if value is None else value
            for key, value in zip(self.config_info_keys(), values)
        }
