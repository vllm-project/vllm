# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadingCounterMetadata:
    name: str
    documentation: str


_OFFLOADING_MANAGER_COUNTERS = {
    "stores_skipped": OffloadingCounterMetadata(
        name="vllm:kv_offload_stores_skipped",
        documentation=(
            "Number of KV offload stores skipped because the reuse threshold "
            "was not reached."
        ),
    ),
}


def get_offloading_counter_metadata(counter_name: str) -> OffloadingCounterMetadata:
    return _OFFLOADING_MANAGER_COUNTERS[counter_name]
