# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-neutral runtime contract for UMBP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from ..data import (
    BlockTransferPlan,
    KVLayoutDescriptor,
    RankTopology,
    TransferJobState,
)


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

    def load_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        """Optionally load logical blocks; None requests ordinary materialization."""
        return None

    def store(self, plans: Sequence[BlockTransferPlan]) -> TransferJobState:
        """Start storing the requested plans."""

    def store_blocks(
        self, plans: Sequence[BlockTransferPlan]
    ) -> TransferJobState | None:
        """Optionally store logical blocks; None requests ordinary materialization."""
        return None

    def wait(self, job: TransferJobState) -> TransferJobState:
        """Return a final state only when buffers are safe to reuse; otherwise raise."""

    def poll(self, job: TransferJobState) -> TransferJobState | None:
        """Return a finished job without blocking, or None if still pending."""

    def publish(self, job: TransferJobState) -> None:
        """Accept a completed store; a runtime may defer visibility until then."""

    def cancel(self, job: TransferJobState) -> TransferJobState:
        """Cancel if possible, otherwise wait until buffers are safe to reuse."""

    def take_evicted_keys(self) -> Sequence[str]:
        """Return locally published keys evicted since the last call."""

    def close(self) -> None:
        """Release worker-side resources."""


class IUMBPRuntime(Protocol):
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
