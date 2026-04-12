# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class StoreJobEntry:
    """A store job entry bundling request context with transfer spec.

    Keyed by scheduler-assigned job ID in the metadata dict.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler calls complete_store immediately.
    """

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class LoadJobEntry:
    """A load job entry bundling request context with transfer spec."""

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    reqs_to_load: dict[int, LoadJobEntry]
    reqs_to_store: dict[int, StoreJobEntry]
    reqs_to_flush: set[str] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store jobs.

    Each worker reports {job_id: 1} for newly completed stores.
    aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a store completion only when count reaches num_workers.
    """

    completed_store_jobs: dict[int, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)
        merged = dict(self.completed_store_jobs)
        for k, v in other.completed_store_jobs.items():
            merged[k] = merged.get(k, 0) + v
        return OffloadingWorkerMetadata(completed_store_jobs=merged)
