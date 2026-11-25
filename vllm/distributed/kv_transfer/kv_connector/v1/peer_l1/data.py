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
    # IPC handles are exported once per receiver, then removed before scheduling.
    handles: dict[int, tuple[Any, ...]] = field(repr=False)
    instance_id: str = ""
    kv_compatibility_id: str = ""
    hash_settings: str = ""
    restart_group: str = ""
    session_kinds: tuple[str, ...] = ("ipc",)
    rdma: bytes = field(default=b"", repr=False)
    rdma_transport: str = ""
    lease_s: float = 1.0
    role: str = "full"


@dataclass
class PeerHandshake(KVConnectorHandshakeMetadata):
    members: list[PeerMemory]
    credential: bytes = field(default=b"", repr=False)


@dataclass(frozen=True)
class CopyJob:
    peer: int | str
    grant: Grant
    # (cache group, source row, destination row).
    blocks: tuple[tuple[int, int, int], ...]
    request_id: str = ""
    memory: PeerMemory | None = None


@dataclass
class PeerMetadata(KVConnectorMetadata):
    step: int
    jobs: tuple[CopyJob, ...] = ()


@dataclass
class PeerWorkerMetadata(KVConnectorWorkerMetadata):
    completed_step: int
    quarantined: dict[str, tuple[int, ...]] = field(default_factory=dict)
    drained: set[str] = field(default_factory=set)
    failed_peers: set[str] = field(default_factory=set)
    successful_peers: dict[str, float] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "PeerWorkerMetadata":
        if not isinstance(other, PeerWorkerMetadata):
            raise TypeError("Incompatible peer L1 worker metadata")
        return PeerWorkerMetadata(
            min(self.completed_step, other.completed_step),
            {**self.quarantined, **other.quarantined},
            self.drained | other.drained,
            self.failed_peers | other.failed_peers,
            {
                key: max(
                    self.successful_peers.get(key, float("-inf")),
                    other.successful_peers.get(key, float("-inf")),
                )
                for key in self.successful_peers.keys() | other.successful_peers.keys()
            },
        )
