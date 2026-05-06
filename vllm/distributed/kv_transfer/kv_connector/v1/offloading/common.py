# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    transfer_spec: TransferSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        return OffloadingWorkerMetadata(completed_jobs=merged)
