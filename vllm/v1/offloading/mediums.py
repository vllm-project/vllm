# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC

from vllm.v1.offloading.abstract import LoadStoreSpec


class BlockIDLoadStoreSpec(LoadStoreSpec, ABC):
    """
    Spec for loading/storing a KV block from a given block number.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id

    def __repr__(self) -> str:
        return str(self.block_id)


class GPULoadStoreSpec(BlockIDLoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(BlockIDLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    def medium() -> str:
        return "CPU"
