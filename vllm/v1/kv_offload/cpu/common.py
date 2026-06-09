# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing_extensions import override

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    @override
    def medium() -> str:
        return "CPU"
