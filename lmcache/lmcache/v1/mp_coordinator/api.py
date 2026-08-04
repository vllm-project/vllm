# SPDX-License-Identifier: Apache-2.0
"""Cross-module contract vocabulary for the MP coordinator.

The cache-event types both sides of the event stream speak: emitted by
MP servers, consumed by the coordinator's key directory.
Encoding-level checks (key convertibility,
hex validity) belong to the HTTP envelopes in :mod:`schemas`.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import Enum

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, Tier


class CacheEventType(str, Enum):
    """The kind of cache-state change a :class:`CacheEventBatch` reports.

    ``STORE`` commits placements; ``DELETE`` removes them (owners report
    evictions as deletes); ``ACCESS`` refreshes recency without changing
    placement state.
    """

    STORE = "store"
    DELETE = "delete"
    ACCESS = "access"


@dataclass(frozen=True)
class CacheEventEntry:
    """One key's worth of change inside a :class:`CacheEventBatch`.

    Attributes:
        key: The object key the change applies to.
        size_bytes: Bytes committed for the key (``store`` only; ``0``
            otherwise).
        content_hash_hex: Hex of the chunk's position-independent content
            hash; empty when the emitter does not compute it.
    """

    key: EncodedObjectKey
    size_bytes: int = 0
    content_hash_hex: str = ""

    def __post_init__(self) -> None:
        """Enforce intrinsic invariants.

        Raises:
            ValueError: If ``size_bytes`` is negative.
        """
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be >= 0 (got {self.size_bytes})")


@dataclass(frozen=True)
class CacheEventBatch:
    """A batch of same-typed cache events from one MP server.

    Attributes:
        instance_id: The emitting MP server (non-empty).
        incarnation: The emitter's restart counter (non-negative). A
            higher value fences off all placements reported by lower
            values of the same ``instance_id``.
        seq: Per-``(instance_id, incarnation)`` monotonic batch counter,
            starting at 1.
        event_type: What happened to every entry in the batch.
        tier: The cache tier the events apply to (``l1`` or ``l2``;
            never ``all``).
        backend: The storage backend within the tier (``"dram"``,
            ``"cxl"``, ``"fs"``, ``"valkey"``, ...). Required non-empty
            for ``store``/``delete`` (it is part of the placement
            identity); empty for ``access``, which only refreshes
            key-level recency and carries no placement identity.
        entries: The affected keys.
        ts: Emitter wall-clock seconds for the batch (``0.0`` if unknown).
    """

    instance_id: str
    incarnation: int
    seq: int
    event_type: CacheEventType
    tier: Tier
    backend: str
    entries: list[CacheEventEntry] = field(default_factory=list)
    ts: float = 0.0

    def __post_init__(self) -> None:
        """Enforce intrinsic invariants.

        Raises:
            ValueError: If ``instance_id`` is empty, ``backend`` is empty
                on a placement-bearing batch (``store``/``delete``),
                ``incarnation`` or ``ts`` is negative, ``seq`` < 1, or
                ``tier`` is not a concrete tier (``l1``/``l2``).
        """
        if not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        if not self.backend and self.event_type != CacheEventType.ACCESS:
            raise ValueError(
                f"backend must be non-empty for {self.event_type.value} batches"
            )
        if self.incarnation < 0:
            raise ValueError(f"incarnation must be >= 0 (got {self.incarnation})")
        if self.seq < 1:
            raise ValueError(f"seq must be >= 1 (got {self.seq})")
        if self.tier not in (Tier.L1, Tier.L2):
            raise ValueError(
                f"cache events must target a concrete tier (got {self.tier.value!r})"
            )
        if self.ts < 0.0:
            raise ValueError(f"ts must be >= 0 (got {self.ts})")
