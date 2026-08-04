# SPDX-License-Identifier: Apache-2.0
"""MP-server-side cache-event emission for the coordinator key directory.

A :class:`CacheEventSubscriber` on the observability event bus turns the
storage layer's L1/L2 key events into ordered :class:`CacheEventBatch`
lists and delivers them through a :class:`CacheEventSink` — the
transport seam (HTTP today, a message queue later). Mapping, batching,
and delivery all run on the bus's drain thread; there is no dedicated
emission thread or task. See
``docs/design/v1/mp_coordinator/cache_events.md``.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, ObjectKey, Tier
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.schemas import DirectoryEventsRequest
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber

logger = init_logger(__name__)

_DEFAULT_FLUSH_INTERVAL = 1.0


class CacheEventPublishError(Exception):
    """A sink failed to deliver a list of cache-event batches."""


class CacheEventSink(ABC):
    """Transport seam for delivering cache-event batches to the directory.

    Implementations provide at-least-once delivery and preserve batch
    order within and across :meth:`publish` calls; the directory's seq
    dedup, gap detection, and incarnation fencing absorb everything else.
    """

    @abstractmethod
    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` to the directory, in list order.

        Args:
            batches: The batches to deliver; never empty.

        Raises:
            CacheEventPublishError: If delivery failed. Retrying and
                dropping are both safe (dedup / gap-flagged resync).
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """Release transport resources. Called once at shutdown."""
        pass


class HttpCacheEventSink(CacheEventSink):
    """Sink that POSTs batches to the coordinator's ``/directory/events``.

    Owns a synchronous HTTP client: publishing happens on the event
    bus's drain thread, so the request timeout bounds how long a flush
    can stall event dispatch.

    Args:
        coordinator_url: Coordinator base URL.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, coordinator_url: str, timeout: float = 2.0) -> None:
        self._base_url = coordinator_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` via one ``POST /directory/events`` request.

        Args:
            batches: The batches to deliver; never empty.

        Raises:
            CacheEventPublishError: If the request failed or returned
                a non-2xx status.
        """
        body = DirectoryEventsRequest(batches=batches)
        try:
            resp = self._client.post(
                f"{self._base_url}/directory/events",
                json=body.model_dump(mode="json"),
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise CacheEventPublishError(
                f"failed to publish {len(batches)} cache-event batches to "
                f"{self._base_url}: {e}"
            ) from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


@dataclass
class _PendingBatch:
    """A buffered batch-to-be: entries sharing one ``(event_type, tier,
    backend)`` identity. Becomes exactly one :class:`CacheEventBatch` at
    flush, which stamps ``seq`` and ``ts``."""

    event_type: CacheEventType
    tier: Tier
    backend: str
    entries: list[CacheEventEntry]


class CacheEventSubscriber(EventSubscriber):
    """Event-bus subscriber that emits the fleet cache-event stream.

    Not thread-safe by design: every method runs on the bus's single
    drain thread (``EventBus.stop()`` invokes :meth:`shutdown` only
    after that thread has been joined), so no locking is needed.

    Args:
        sink: Transport that delivers flushed batches.
        instance_id: This MP server's id (sent with every batch).
        incarnation: This server process's incarnation (its start time);
            fences out placements reported before a restart.
        flush_interval: Minimum seconds between event-driven flushes
            (must be >= 0).

    Raises:
        ValueError: If ``flush_interval`` is negative.
    """

    def __init__(
        self,
        sink: CacheEventSink,
        instance_id: str,
        incarnation: int,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        if flush_interval < 0:
            raise ValueError(f"flush_interval must be >= 0 (got {flush_interval})")
        self._sink = sink
        self._instance_id = instance_id
        self._incarnation = incarnation
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()
        self._seq = 0
        # Consecutive same-identity entries append to the last pending
        # batch; an identity change starts a new one (order-preserving).
        self._pending_batches: list[_PendingBatch] = []

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the storage key events this subscriber consumes."""
        return {
            EventType.L1_WRITE_FINISHED: self._on_l1_store,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_l1_store,
            EventType.L1_KEYS_EVICTED: self._on_l1_delete,
            EventType.L1_KEYS_ACCESSED: self._on_l1_access,
            EventType.L2_KEYS_STORED: self._on_l2_store,
            EventType.L2_KEYS_DELETED: self._on_l2_delete,
            EventType.L2_KEYS_ACCESSED: self._on_l2_access,
            # TODO: decouple the flush tick from the eviction loop (e.g. a
            # bus-owned periodic hook) so cache-event freshness does not
            # silently depend on the eviction loop's cadence.
            EventType.L1_EVICTION_LOOP_TICK: self._on_tick,
        }

    def flush(self) -> None:
        """Drain the buffer and publish one batch per pending batch.

        Publish failures are logged and the drained list is dropped.
        """
        if not self._pending_batches:
            return
        pending_batches = self._pending_batches
        self._pending_batches = []
        ts = time.time()
        batches = [
            CacheEventBatch(
                instance_id=self._instance_id,
                incarnation=self._incarnation,
                seq=self._seq + offset + 1,
                event_type=pending.event_type,
                tier=pending.tier,
                backend=pending.backend,
                entries=pending.entries,
                ts=ts,
            )
            for offset, pending in enumerate(pending_batches)
        ]
        self._seq += len(pending_batches)
        try:
            self._sink.publish(batches)
        except CacheEventPublishError as e:
            logger.warning(
                "Dropping %d cache-event batches (instance %s): %s",
                len(batches),
                self._instance_id,
                e,
            )

    def shutdown(self) -> None:
        """Flush buffered events and close the sink. Called by
        ``EventBus.stop()`` after the final drain."""
        self.flush()
        self._sink.close()

    # -- Event handlers (bus drain thread) ------------------------------------

    def _on_tick(self, event: Event) -> None:
        self._flush_if_due()

    def _on_l1_store(self, event: Event) -> None:
        self._record_l1_placements(CacheEventType.STORE, event)

    def _on_l1_delete(self, event: Event) -> None:
        self._record_l1_placements(CacheEventType.DELETE, event)

    def _on_l1_access(self, event: Event) -> None:
        keys: list[ObjectKey] = event.metadata["keys"]
        # ACCESS refreshes key-level recency only; it carries no
        # placement identity, so the backend is empty by contract.
        self._record(
            CacheEventType.ACCESS,
            Tier.L1,
            "",
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )

    def _on_l2_store(self, event: Event) -> None:
        keys: list[ObjectKey] = event.metadata["keys"]
        sizes: list[int] = event.metadata["sizes"]
        self._record(
            CacheEventType.STORE,
            Tier.L2,
            event.metadata["backend"],
            [
                CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size)
                for key, size in zip(keys, sizes, strict=True)
            ],
        )

    def _on_l2_delete(self, event: Event) -> None:
        self._record_l2_keys(CacheEventType.DELETE, event)

    def _on_l2_access(self, event: Event) -> None:
        self._record_l2_keys(CacheEventType.ACCESS, event)

    # -- Internals -------------------------------------------------------------

    def _record_l1_placements(self, event_type: CacheEventType, event: Event) -> None:
        """Record one ``event_type`` batch per medium found in the
        event's ``meta`` list (parallel to ``keys``)."""
        keys: list[ObjectKey] = event.metadata["keys"]
        metadata: list[L1ObjectMeta] = event.metadata["meta"]
        by_backend: dict[L1BackendType, list[CacheEventEntry]] = {}
        for key, meta in zip(keys, metadata, strict=True):
            by_backend.setdefault(meta.backend, []).append(
                CacheEventEntry(
                    key=key.to_encoded_object_key(),
                    size_bytes=meta.size_bytes
                    if event_type is CacheEventType.STORE
                    else 0,
                )
            )
        for backend, entries in by_backend.items():
            self._record(event_type, Tier.L1, backend.value, entries)

    def _record_l2_keys(self, event_type: CacheEventType, event: Event) -> None:
        """Record a size-less L2 batch (deletes and accesses)."""
        keys: list[ObjectKey] = event.metadata["keys"]
        self._record(
            event_type,
            Tier.L2,
            event.metadata["backend"],
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )

    def _record(
        self,
        event_type: CacheEventType,
        tier: Tier,
        backend: str,
        entries: list[CacheEventEntry],
    ) -> None:
        """Buffer ``entries``, then flush if the flush interval elapsed."""
        if not entries:
            return
        last = self._pending_batches[-1] if self._pending_batches else None
        if (
            last is not None
            and last.event_type == event_type
            and last.tier == tier
            and last.backend == backend
        ):
            last.entries.extend(entries)
        else:
            self._pending_batches.append(
                _PendingBatch(
                    event_type=event_type,
                    tier=tier,
                    backend=backend,
                    entries=list(entries),
                )
            )
        self._flush_if_due()

    def _flush_if_due(self) -> None:
        """Flush once ``flush_interval`` has elapsed since the last flush."""
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._last_flush = now
            self.flush()
