# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.offloading.abstract import LoadStoreSpec


class GPULoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing KV data to GPU memory.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = block_ids

    @staticmethod
    def medium() -> str:
        return "GPU"


class CPULoadStoreSpec(LoadStoreSpec):
    """
    Spec for loading/storing KV data to CPU memory.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = block_ids

    @staticmethod
    def medium() -> str:
        return "CPU"
