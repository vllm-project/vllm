# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str


@dataclass(slots=True)
class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)

    def aggregate(
        self, other: "DirectionalTransferStats"
    ) -> "DirectionalTransferStats":
        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )

    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)

    def is_empty(self) -> bool:
        return self.bytes == 0 and self.time == 0.0 and not self.sizes


@dataclass(slots=True)
class TransferStats:
    load: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)
    store: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)

    def aggregate(self, other: "TransferStats") -> "TransferStats":
        return TransferStats(
            load=self.load.aggregate(other.load),
            store=self.store.aggregate(other.store),
        )

    def is_empty(self) -> bool:
        return self.load.is_empty() and self.store.is_empty()


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    worker_transfer_jobs: dict[int, TransferJob] = field(default_factory=dict)
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
    failed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    def mark_completed(self, job_id: int, success: bool = True) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1
        if not success:
            self.failed_jobs[job_id] = 1

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v
        failed = dict(self.failed_jobs)
        for job_id, v in other.failed_jobs.items():
            failed[job_id] = failed.get(job_id, 0) + v

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            failed_jobs=failed,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
