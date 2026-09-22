# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral runtime contract for UMBP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from ..data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
)


@dataclass(frozen=True)
class UMBPRuntimeCapabilities:
    """Capabilities exposed to the shared connector."""

    lookup: bool = True
    load: bool = True
    store: bool = True
    publish: bool = True
    ranged_io: bool = False
    layerwise_load: bool = False
    layerwise_store: bool = False
    partial_hash_hits: bool = False
    cancellation: bool = False
    async_transfer: bool = False
    eviction_events: bool = False


class UMBPSchedulerHandle(Protocol):
    def lookup(self, keys: Sequence[str]) -> Sequence[bool]:
        """Return authoritative existence for each key."""

    def clear(self) -> bool:
        """Clear all objects visible to this scheduler handle."""

    def close(self) -> None:
        """Release scheduler-side resources."""


class UMBPWorkerHandle(Protocol):
    def register_buffers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register each unique KV allocation with the runtime."""

    def load(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        """Start loading the requested plans."""

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        """Start storing the requested plans."""

    def wait(self, job: TransferJobState) -> TransferJobState:
        """Wait for a job and return its final per-key state."""

    def poll(self, job: TransferJobState) -> TransferJobState | None:
        """Return a finished job without blocking, or None if still pending."""

    def publish(self, job: TransferJobState) -> None:
        """Make a completed store visible to scheduler lookups."""

    def cancel(self, job: TransferJobState) -> TransferJobState:
        """Cancel if possible, otherwise wait until buffers are safe to reuse."""

    def take_evicted_keys(self) -> Sequence[str]:
        """Return locally published keys evicted since the last call."""

    def close(self) -> None:
        """Release worker-side resources."""


class IUMBPRuntime(Protocol):
    capabilities: UMBPRuntimeCapabilities

    def create_scheduler_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPSchedulerHandle:
        """Create a scheduler-side handle."""

    def create_worker_handle(
        self,
        namespace: str,
        topology: RankTopology,
        layout: KVLayoutDescriptor,
    ) -> UMBPWorkerHandle:
        """Create a worker-side handle."""
