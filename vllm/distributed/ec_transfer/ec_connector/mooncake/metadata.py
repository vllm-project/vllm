# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata exchanged by the Mooncake encoder-cache connector."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorWorkerMetadata,
)


@dataclass
class ECMooncakeLoadSpec:
    """Describe a remote reservation or resident allocation to load."""

    mm_hash: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    transfer_id: str
    # The consumer pool still holds this item, so the load is a local handoff:
    # no transfer, no producer.
    local: bool = False


@dataclass
class ECMooncakePushSpec:
    """Describe a destination reservation prepared before a tensor is ready."""

    mm_hash: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    consumer_zmq: str
    transfer_id: str
    request_id: str = ""


@dataclass
class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Worker operations emitted for one Scheduler step."""

    loads: list[ECMooncakeLoadSpec] = field(default_factory=list)
    pushes: list[ECMooncakePushSpec] = field(default_factory=list)
    freed: list[str] | None = None


@dataclass
class ECMooncakeWorkerMetadata(ECConnectorWorkerMetadata):
    """Completion state reported from Workers to the Scheduler."""

    loaded: set[str] = field(default_factory=set)
    failed_loads: set[str] = field(default_factory=set)
    # Items the receive pool dropped under pressure. The scheduler assumes an
    # evicted item stays resident until told otherwise.
    reclaimed: set[str] = field(default_factory=set)
    pending_saves: bool = False
    failed_saves: set[str] = field(default_factory=set)

    def aggregate(self, other: ECConnectorWorkerMetadata) -> ECMooncakeWorkerMetadata:
        assert isinstance(other, ECMooncakeWorkerMetadata)
        return ECMooncakeWorkerMetadata(
            # Every tensor-parallel rank gathers the embedding from its own
            # cache, so an item counts as loaded only where all of them have
            # it; one rank falling short must fail the load rather than leave
            # the scheduler believing it is ready.
            loaded=self.loaded & other.loaded,
            failed_loads=self.failed_loads | other.failed_loads,
            reclaimed=self.reclaimed | other.reclaimed,
            pending_saves=self.pending_saves or other.pending_saves,
            failed_saves=self.failed_saves | other.failed_saves,
        )
