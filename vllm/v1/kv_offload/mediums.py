# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC
from collections.abc import Sequence

import numpy as np

from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.

    If there are multiple KV groups, the blocks are expected to be
    ordered by the group index.
    In that case, group_sizes[i] determines the number of blocks
    per the i-th KV group, and thus sum(group_sizes) == len(block_ids).
    group_sizes=None indicates a single KV group.

    If block_indices is given, each group (determined by group_sizes) of block IDs
    will correspond to logically contiguous blocks, e.g. blocks 5-10 of a some request.
    block_indices[i] will represent the block index of the first block in group #i.
    Thus, len(block_indices) == len(group_sizes) = number of KV cache groups.
    This information is required in order to support loading from offloaded blocks
    which are larger than GPU blocks.
    In such cases, the first GPU block per each group may be unaligned to the offloaded
    block size, and so knowing block_indices[i] allows the worker to correctly
    skip part of the first matching offloaded block.
    Offloading from GPU is always aligned to offloaded block size, and so
    block_indices will only be set by the offloading connector when loading into GPU.
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int] | None = None,
        block_offsets: Sequence[int] | None = None,
        block_counts: Sequence[int] | None = None,
    ):
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids)
        assert block_indices is None or len(block_indices) == len(group_sizes)
        assert (block_offsets is None) == (block_counts is None)
        if block_offsets is not None and block_counts is not None:
            assert len(block_offsets) == len(block_ids)
            assert len(block_counts) == len(block_ids)
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] | None = block_indices
        self.block_offsets: np.ndarray | None = (
            np.array(block_offsets, dtype=np.int64)
            if block_offsets is not None
            else None
        )
        self.block_counts: np.ndarray | None = (
            np.array(block_counts, dtype=np.int64)
            if block_counts is not None
            else None
        )

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"
