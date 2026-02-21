# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

import numpy as np

from vllm.v1.kv_offload.abstract import LoadStoreSpec


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing KV blocks from given block numbers.

    If group_sizes is given, then it is assumed that block_ids list
    can be braked to respective sub-lists given by group_sizes, where
    each sub-list represent a series of consecutive logical blocks.
    group_sizes = None is equivalent to group_sizes = [len(block_ids)].
    """

    def __init__(self, block_ids: list[int], group_sizes: list[int] | None = None):
        self.block_ids = np.array(block_ids, dtype=np.int64)
        self.group_sizes = (
            np.array(group_sizes, dtype=np.int64) if group_sizes else None
        )
        assert group_sizes is None or sum(group_sizes) == len(block_ids)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

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
