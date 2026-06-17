# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing_extensions import override

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec

METRIC_STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
METRIC_CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    @override
    def medium() -> str:
        return "CPU"
