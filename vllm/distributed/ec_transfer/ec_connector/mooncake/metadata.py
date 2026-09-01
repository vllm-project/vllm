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
    """Describe one Consumer-side cache load requested by the Scheduler.

    Attributes:
        mm_hash: Stable identifier of the multimodal encoder item.
        num_token: Number of encoder tokens expected by the request.
        nbytes: Tensor payload size in bytes.
        shape: Tensor shape reconstructed by the Consumer.
        dtype: Unqualified ``torch.dtype`` name used for reconstruction.
        pushed: Whether the tensor came from a remote Producer reservation.
        transfer_id: Identity shared by Scheduler, Producer, and Consumer.
        reservation_id: Consumer-issued capability for completing the write.
        local: Whether to reuse a tensor already resident on the Consumer.
    """

    mm_hash: str
    num_token: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    pushed: bool = False
    transfer_id: str = ""
    reservation_id: str = ""
    # The consumer pool still holds this item, so the load is a local handoff:
    # no transfer, no producer.
    local: bool = False


@dataclass
class ECMooncakePushSpec:
    """Describe a destination reservation prepared before a tensor is ready.

    Attributes:
        mm_hash: Stable identifier of the multimodal encoder item.
        nbytes: Number of bytes the Consumer must reserve.
        shape: Shape of the tensor that will be written.
        dtype: Unqualified ``torch.dtype`` name of the tensor.
        consumer_zmq: Base control address of the destination Consumer.
        transfer_id: Identity shared by Scheduler, Producer, and Consumer.
        request_id: Request that owns the push and may cancel it.
    """

    mm_hash: str
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    consumer_zmq: str
    transfer_id: str
    request_id: str = ""


@dataclass
class ECMooncakeConnectorMetadata(ECConnectorMetadata):
    """Worker operations emitted for one Scheduler step.

    Attributes:
        loads: Consumer loads that should be attached to ``encoder_cache``.
        pushes: Producer reservations that should begin before sources arrive.
    """

    loads: list[ECMooncakeLoadSpec] = field(default_factory=list)
    pushes: list[ECMooncakePushSpec] = field(default_factory=list)

    def add_load(self, spec: ECMooncakeLoadSpec) -> None:
        self.loads.append(spec)

    def add_push(self, spec: ECMooncakePushSpec) -> None:
        self.pushes.append(spec)


@dataclass
class ECMooncakeWorkerMetadata(ECConnectorWorkerMetadata):
    """Completion state reported from Workers to the Scheduler.

    Attributes:
        loaded: Cache identifiers loaded successfully on this Worker.
        failed_loads: Cache identifiers that could not be loaded.
        reclaimed: Resident items evicted because the receive pool was full.
        pending_loads: Whether this Worker still owns asynchronous load work.
        pending_saves: Whether this Worker still owns asynchronous push work.
    """

    loaded: set[str] = field(default_factory=set)
    failed_loads: set[str] = field(default_factory=set)
    # Items the receive pool dropped under pressure. The scheduler assumes an
    # evicted item stays resident until told otherwise.
    reclaimed: set[str] = field(default_factory=set)
    pending_loads: bool = False
    pending_saves: bool = False

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
            pending_loads=self.pending_loads or other.pending_loads,
            pending_saves=self.pending_saves or other.pending_saves,
        )
