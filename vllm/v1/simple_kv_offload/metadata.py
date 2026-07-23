# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False

    # Disk write-back: CPU blocks being evicted → write to NVMe
    disk_write_cpu_blocks: list[int] = field(default_factory=list)
    disk_write_disk_blocks: list[int] = field(default_factory=list)

    # Disk read (prefetch): NVMe blocks → read into CPU before CPU→GPU load
    disk_read_cpu_blocks: list[int] = field(default_factory=list)
    disk_read_disk_blocks: list[int] = field(default_factory=list)


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(completed_store_events=merged)
