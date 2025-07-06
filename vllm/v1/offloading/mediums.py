# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.offloading.abstract import LoadStoreSpec


class GPULoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing a KV block to GPU memory.
    """

    def __init__(self, block_id: list[int]):
        self.block_id = block_id

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id

    @staticmethod
    def medium() -> str:
        return "CPU"
