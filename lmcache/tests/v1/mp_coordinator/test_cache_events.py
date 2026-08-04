# SPDX-License-Identifier: Apache-2.0
"""Tests for MP-server-side cache-event emission: the event-bus
subscriber's event mapping, batching/ordering, seq/gap semantics on
publish failure, and the HTTP sink end-to-end against a coordinator
app."""

# Standard
from dataclasses import asdict
import asyncio
import threading

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType, ObjectKey, Tier
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import (
    CacheEventPublishError,
    CacheEventSink,
    CacheEventSubscriber,
    HttpCacheEventSink,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


def _entry(hash_byte: int, size_bytes: int = 0) -> CacheEventEntry:
    return CacheEventEntry(
        key=_key(hash_byte).to_encoded_object_key(), size_bytes=size_bytes
    )


def _meta(
    size_bytes: int = 0, backend: L1BackendType = L1BackendType.DRAM
) -> L1ObjectMeta:
    return L1ObjectMeta(size_bytes=size_bytes, backend=backend)


class _RecordingSink(CacheEventSink):
    """Sink that records every published list; optionally fails."""

    def __init__(self) -> None:
        self.published: list[list[CacheEventBatch]] = []
        self.fail_next = False
        self.closed = False

    def publish(self, batches: list[CacheEventBatch]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise CacheEventPublishError("injected failure")
        self.published.append(batches)

    def close(self) -> None:
        self.closed = True


def _subscriber(
    sink: CacheEventSink,
    incarnation: int = 7,
    flush_interval: float = 3600.0,
) -> CacheEventSubscriber:
    """Build a subscriber whose default interval never auto-flushes in a
    test, so batching assertions drive ``flush()`` explicitly."""
    return CacheEventSubscriber(
        sink=sink,
        instance_id="node-a",
        incarnation=incarnation,
        flush_interval=flush_interval,
    )


def _dispatch(subscriber: CacheEventSubscriber, *events: Event) -> None:
    """Deliver events to the subscriber the way the bus drain thread does."""
    subscriptions = subscriber.get_subscriptions()
    for event in events:
        subscriptions[event.event_type](event)


# -- Subscriber event mapping -------------------------------------------------


def test_l1_store_access_delete_events_map_to_batches():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={"keys": [_key(1), _key(2)], "meta": [_meta(100), _meta(200)]},
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
            metadata={"keys": [_key(3)], "meta": [_meta(300)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(1)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(2)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_EVICTED,
            metadata={"keys": [_key(3)], "meta": [_meta(300)]},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    # Consecutive same-identity records coalesce across events: the two
    # store events form one batch, the two access events form one batch.
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200, 300]
    assert len(access.entries) == 2
    assert all(e.size_bytes == 0 for e in access.entries)  # ACCESS never sizes
    assert access.backend == ""  # ACCESS carries no placement identity
    assert delete.entries[0].key == _key(3).to_encoded_object_key()
    assert delete.entries[0].size_bytes == 0
    assert all(b.tier == Tier.L1 for b in batches)
    assert store.backend == "dram" and delete.backend == "dram"
    assert [b.seq for b in batches] == [1, 2, 3]
    assert all(b.instance_id == "node-a" and b.incarnation == 7 for b in batches)


def test_l1_events_split_batches_by_medium():
    """A hybrid DRAM+DAX store emits one batch per medium, and deletes
    target the same per-medium identity the stores reported."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={
                "keys": [_key(1), _key(2), _key(3)],
                "meta": [
                    _meta(100, L1BackendType.DRAM),
                    _meta(200, L1BackendType.DEVDAX),
                    _meta(300, L1BackendType.DRAM),
                ],
            },
        ),
        Event(
            event_type=EventType.L1_KEYS_EVICTED,
            metadata={"keys": [_key(2)], "meta": [_meta(200, L1BackendType.DEVDAX)]},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [(b.event_type, b.backend) for b in batches] == [
        (CacheEventType.STORE, "dram"),
        (CacheEventType.STORE, "devdax"),
        (CacheEventType.DELETE, "devdax"),
    ]
    dram_store, devdax_store, devdax_delete = batches
    assert [e.size_bytes for e in dram_store.entries] == [100, 300]
    assert [e.size_bytes for e in devdax_store.entries] == [200]
    assert devdax_delete.entries[0].key == _key(2).to_encoded_object_key()


def test_l1_access_batches_have_empty_backend():
    """ACCESS refreshes key-level recency only, so its batches carry no
    placement identity: the backend is empty by contract."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(1), _key(2)]},
        ),
    )
    subscriber.flush()
    [[batch]] = sink.published
    assert batch.event_type == CacheEventType.ACCESS
    assert batch.tier == Tier.L1
    assert batch.backend == ""
    assert all(e.size_bytes == 0 for e in batch.entries)


def test_read_finished_is_not_consumed():
    """L1_READ_FINISHED is covered by the request-end unified touch
    (L1_KEYS_ACCESSED); consuming both would duplicate ACCESS events."""
    subscriber = _subscriber(_RecordingSink())
    assert EventType.L1_READ_FINISHED not in subscriber.get_subscriptions()


def test_l2_events_map_with_backend_and_sizes():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1), _key(2)], "sizes": [100, 200], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_ACCESSED,
            metadata={"keys": [_key(1)], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(2)], "backend": "fs"},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200]
    assert access.entries[0].key == _key(1).to_encoded_object_key()
    assert delete.entries[0].key == _key(2).to_encoded_object_key()
    assert all(b.tier == Tier.L2 and b.backend == "fs" for b in batches)


def test_interleaved_events_preserve_total_order():
    # store k1, delete k1, re-store k1: the re-store must not be
    # reordered before the delete, or the directory ends up empty.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(1)], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [150], "backend": "fs"},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.DELETE,
        CacheEventType.STORE,
    ]
    assert [b.seq for b in batches] == [1, 2, 3]


def test_mismatched_l1_meta_raises():
    subscriber = _subscriber(_RecordingSink())
    with pytest.raises(ValueError):
        _dispatch(
            subscriber,
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": [_key(1), _key(2)], "meta": [_meta(100)]},
            ),
        )


# -- Emitter flush / seq semantics --------------------------------------------


def test_flush_with_empty_buffer_publishes_nothing():
    sink = _RecordingSink()
    _subscriber(sink).flush()
    assert sink.published == []


def test_publish_failure_drops_batches_and_leaves_a_seq_gap():
    # Failed flushes consume their seq numbers so the directory sees a
    # gap and can flag the instance for resync.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    sink.fail_next = True
    subscriber.flush()
    assert sink.published == []

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(2)], "sizes": [200], "backend": "fs"},
        ),
    )
    subscriber.flush()
    [[batch]] = sink.published
    assert batch.seq == 2


def test_negative_flush_interval_rejected():
    with pytest.raises(ValueError):
        _subscriber(_RecordingSink(), flush_interval=-1.0)


def test_events_self_pace_flushing():
    """With interval 0 every event flushes; a long interval holds events
    back until an explicit flush."""
    store_event = Event(
        event_type=EventType.L2_KEYS_STORED,
        metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
    )
    eager_sink = _RecordingSink()
    _dispatch(_subscriber(eager_sink, flush_interval=0.0), store_event, store_event)
    assert len(eager_sink.published) == 2

    lazy_sink = _RecordingSink()
    _dispatch(_subscriber(lazy_sink), store_event)
    assert lazy_sink.published == []


def test_eviction_tick_flushes_buffered_tail():
    """A buffered tail (burst-ending events) is flushed by the eviction
    loop's tick once the flush interval elapses, without new cache
    events."""
    # Standard
    import time as _time

    sink = _RecordingSink()
    subscriber = _subscriber(sink, flush_interval=0.05)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    assert sink.published == []  # interval not yet elapsed: tail buffered
    _time.sleep(0.06)
    _dispatch(
        subscriber,
        Event(event_type=EventType.L1_EVICTION_LOOP_TICK, metadata={"usage": 0.0}),
    )
    assert len(sink.published) == 1


def test_shutdown_flushes_and_closes_sink():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    subscriber.shutdown()
    assert len(sink.published) == 1
    assert sink.closed is True


# -- Bus integration -----------------------------------------------------------


def test_subscriber_on_a_real_bus_flushes_on_events():
    """End-to-end through a real EventBus: published events reach the
    sink via the drain thread's event callbacks, with no dedicated
    emission thread."""
    sink = _RecordingSink()
    bus = EventBus(EventBusConfig(enabled=True))
    bus.register_subscriber(_subscriber(sink, flush_interval=0.0))
    bus.start()
    try:
        bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": [_key(1)], "meta": [_meta(100)]},
            )
        )
        waiter = threading.Event()
        for _ in range(100):
            if sink.published:
                break
            waiter.wait(0.05)
        assert sink.published, "event-driven flush never delivered the batch"
        [[batch]] = sink.published
        assert batch.event_type == CacheEventType.STORE
        assert batch.entries[0].size_bytes == 100
    finally:
        bus.stop()
    # Shutdown closed the sink via the subscriber hook.
    assert sink.closed is True


def test_bus_stop_flushes_buffered_events():
    """Events recorded but not yet flushed are delivered by the
    subscriber's shutdown hook during ``EventBus.stop()``."""
    sink = _RecordingSink()
    bus = EventBus(EventBusConfig(enabled=True))
    bus.register_subscriber(_subscriber(sink))
    bus.start()
    bus.publish(
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [64], "backend": "fs"},
        )
    )
    bus.stop()
    assert len(sink.published) == 1
    assert sink.closed is True


# -- HTTP sink end-to-end -------------------------------------------------------


class _SyncASGITransport(httpx.BaseTransport):
    """Bridge httpx's sync client onto an in-process ASGI app."""

    def __init__(self, asgi: httpx.ASGITransport) -> None:
        self._asgi = asgi

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _roundtrip() -> tuple[int, httpx.Headers, bytes]:
            response = await self._asgi.handle_async_request(request)
            content = await response.aread()
            return response.status_code, response.headers, content

        status_code, headers, content = asyncio.run(_roundtrip())
        return httpx.Response(status_code=status_code, headers=headers, content=content)


def test_http_sink_feeds_the_directory_end_to_end():
    """Subscriber events -> emitter -> HTTP sink -> coordinator app ->
    directory lookup, all with the synchronous sink."""
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    asgi = httpx.ASGITransport(app=create_app(config))

    sink = HttpCacheEventSink("http://coordinator")
    # Point the sink's client at the in-process app (same public API).
    sink._client = httpx.Client(  # noqa: SLF001 — test-only transport swap
        transport=_SyncASGITransport(asgi), base_url="http://coordinator"
    )
    subscriber = _subscriber(sink, incarnation=3)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1), _key(2)], "sizes": [100, 200], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(2)], "backend": "fs"},
        ),
    )
    subscriber.flush()

    async def _verify() -> None:
        async with httpx.AsyncClient(
            transport=asgi, base_url="http://coordinator"
        ) as client:
            resp = await client.post(
                "/directory/lookup",
                json={
                    "keys": [
                        asdict(_key(1).to_encoded_object_key()),
                        asdict(_key(2).to_encoded_object_key()),
                    ]
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            [placement] = results[0]["placements"]
            assert placement["instance_id"] == "node-a"
            assert placement["incarnation"] == 3
            assert placement["tier"] == "l2"
            assert placement["backend"] == "fs"
            assert placement["size_bytes"] == 100
            assert results[1]["placements"] == []

            stats = (await client.get("/directory/stats")).json()
            instance = stats["instances"]["node-a"]
            assert instance["last_seq"] == 2
            assert instance["gap_detected"] is False

    asyncio.run(_verify())


def test_http_sink_raises_publish_error_on_http_failure():
    sink = HttpCacheEventSink("http://127.0.0.1:1")  # nothing listens here
    batch = CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[_entry(1, 100)],
    )
    with pytest.raises(CacheEventPublishError):
        sink.publish([batch])
    sink.close()
