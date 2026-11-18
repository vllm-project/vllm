# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serializable peer memory descriptions and scheduler/worker messages."""

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

from .session import Grant


@dataclass(frozen=True)
class PeerMemory:
    rank: int
    endpoint: str
    epoch: int
    host: str
    device: int
    compatibility: str
    num_blocks: int
    segments: tuple[tuple[tuple[int, int, int], ...], ...]
    # Exported once per receiver. Descriptors never enter the scheduler metadata.
    handles: dict[int, tuple[Any, ...]] = field(repr=False)


@dataclass
class PeerHandshake(KVConnectorHandshakeMetadata):
    members: list[PeerMemory]


@dataclass(frozen=True)
class CopyJob:
    peer: int
    grant: Grant
    # (cache group, source row, destination row).
    blocks: tuple[tuple[int, int, int], ...]
    request_id: str = ""


@dataclass
class PeerMetadata(KVConnectorMetadata):
    step: int
    jobs: tuple[CopyJob, ...] = ()


@dataclass
class PeerWorkerMetadata(KVConnectorWorkerMetadata):
    completed_step: int
    quarantined: dict[str, tuple[int, ...]] = field(default_factory=dict)
    drained: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "PeerWorkerMetadata":
        if not isinstance(other, PeerWorkerMetadata):
            raise TypeError("Incompatible peer L1 worker metadata")
        return PeerWorkerMetadata(
            min(self.completed_step, other.completed_step),
            {**self.quarantined, **other.quarantined},
            self.drained | other.drained,
        )
