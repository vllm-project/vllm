# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class DiskPrefetchSpec:
    """Disk→CPU prefetch request transported from scheduler to worker."""
    cpu_block_ids: list[int]
    disk_block_ids: list[int]


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]
    reqs_to_flush: set[str] | None = None
    # Disk prefetch requests (scheduler → worker)
    disk_prefetches: list[DiskPrefetchSpec] = field(default_factory=list)
