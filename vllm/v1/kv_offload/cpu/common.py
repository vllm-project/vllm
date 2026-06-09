# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    OffloadingMetricMetadata,
)

METRIC_STORES_SKIPPED = "vllm:kv_offload_stores_skipped"


@dataclass
class CPUOffloadingConfig:
    num_blocks: int
    eviction_policy: str
    enable_events: bool
    store_threshold: int
    max_tracker_size: int
    metric_definitions: dict[str, OffloadingMetricMetadata]


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.
    """

    @staticmethod
    @override
    def medium() -> str:
        return "CPU"
