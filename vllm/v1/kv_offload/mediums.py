# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

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


class DRAMLoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "DRAM"


class CXLLoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "CXL"


class CXLExtentLoadStoreSpec(CXLLoadStoreSpec):
    def __init__(self, base_block_id: int, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.base_block_id = int(base_block_id)
        self.num_blocks = int(num_blocks)
        self.block_ids = np.arange(
            self.base_block_id,
            self.base_block_id + self.num_blocks,
            dtype=np.int64,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_block_id={self.base_block_id}, "
            f"num_blocks={self.num_blocks})"
        )


class CXLLayerGroupExtentLoadStoreSpec(CXLExtentLoadStoreSpec):
    def __init__(self, base_block_id: int, num_blocks: int, layer_group_id: int):
        super().__init__(base_block_id=base_block_id, num_blocks=num_blocks)
        self.layer_group_id = int(layer_group_id)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_block_id={self.base_block_id}, "
            f"num_blocks={self.num_blocks}, layer_group_id={self.layer_group_id})"
        )
