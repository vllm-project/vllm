# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import random
import threading
import time
from collections import Counter

import msgspec
import pytest
import zmq

from examples.features.kv_events.kv_events_snapshot_subscriber import (
    ResyncRequired,
    SnapshotClient,
)
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.distributed.kv_events_snapshot import KVCacheSnapshot, KVEventSnapshotRecorder

pytestmark = pytest.mark.skip_global_cleanup


def stored(hashes, parent=None, medium="GPU", group=None):
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=[h * 4 + j for h in hashes for j in range(4)]
        if medium == "GPU"
        else [],
        block_size=4 if medium == "GPU" else 0,
        lora_id=None,
        lora_name=None,
        medium=medium,
        group_idx=group,
    )


def counts(events, live: Counter | None = None):
    if live is None:
        live = Counter()
    for e in events:
        if isinstance(e, BlockStored):
            live.update((e.medium, e.group_idx, h) for h in e.block_hashes)
        elif isinstance(e, BlockRemoved):
            for h in e.block_hashes:
                key = (e.medium, e.group_idx, h)
                if live[key] > 1:
                    live[key] -= 1
                else:
                    live.pop(key, None)
        else:
            for key in list(live):
                if key[0] in ("GPU", None):
                    del live[key]
    return live


def random_batches(seed, n=400):
    rng = random.Random(seed)
    live: Counter = Counter()
    for i in range(n):
        events = []
        if rng.random() < 0.5 or not live:
            start = 1 + 8 * rng.randrange(30)
            events.append(stored(list(range(start, start + 8))))
        else:
            key = rng.choice(list(live))
            if key[0] == "GPU" and rng.random() < 0.3:
                events.append(stored([key[2]], medium="CPU"))
            else:
                events.append(
                    BlockRemoved(block_hashes=[key[2]], medium=key[0], group_idx=key[1])
                )
        if rng.random() < 0.02:
            events.append(AllBlocksCleared())
        counts(events, live)
        yield events


@pytest.mark.parametrize("seed", range(5))
def test_snapshot_references_match_full_history(seed):
    snap = KVCacheSnapshot()
    expected: Counter = Counter()
    for i, events in enumerate(random_batches(seed)):
        counts(events, expected)
        snap.apply(events)
        if i % 31 == 0:
            assert counts(snap.export()) == expected
    assert counts(snap.export()) == expected


def test_orphan_metadata_is_retained_within_budget(monkeypatch):
    snap = KVCacheSnapshot()
    for h in range(1, 1001):
        snap.apply([stored([h])])
        snap.apply([BlockRemoved(block_hashes=[h], medium="GPU")])
    assert snap._sources and snap._known and snap._orphan_metadata_bytes
    monkeypatch.setattr(snap, "MAX_ORPHAN_METADATA_BYTES", 0)
    snap.apply([])
    assert not snap._sources and not snap._known
    assert snap._metadata_bytes == 0


def test_dependencies_released_iteratively(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "MAX_ORPHAN_METADATA_BYTES", 0)
    for h in range(1, 1501):
        snap.apply([stored([h], parent=h - 1 if h > 1 else None)])
        if h > 1:
            snap.apply([BlockRemoved(block_hashes=[h - 1], medium="GPU")])
    assert len(snap) == 1
    assert len(snap._sources) == 1500
    snap.apply([BlockRemoved(block_hashes=[1500], medium="GPU")])
    assert not snap._sources and not snap._known


def test_delayed_transfer_preserves_metadata():
    snap = KVCacheSnapshot()
    snap.apply([stored([1, 2])])
    snap.apply([BlockRemoved(block_hashes=[1, 2], medium="GPU")])
    snap.apply([stored([2], parent=1, medium="CPU")])
    exported = list(snap.export())
    assert isinstance(exported[0], BlockStored) and exported[0].token_ids
    assert counts(exported) == Counter({("CPU", None, 2): 1})


def test_unknown_parent_without_known_block_fails_closed():
    with pytest.raises(ValueError, match="Missing reconstruction"):
        KVCacheSnapshot().apply([stored([2], parent=1)])


def test_budget_collection_preserves_selected_dependency(monkeypatch):
    snap = KVCacheSnapshot()
    parent = stored([1])
    unrelated = stored([9])
    child = stored([2], parent=1)
    snap.apply([parent])
    snap.apply([BlockRemoved(block_hashes=[1], medium="GPU")])
    snap.apply([unrelated])
    snap.apply([BlockRemoved(block_hashes=[9], medium="GPU")])
    parent_cost = snap._sources[0].cost
    child_cost = 512 + 64 * (len(child.block_hashes) + len(child.token_ids))
    child_cost += len(msgspec.msgpack.encode(child))
    monkeypatch.setattr(snap, "MAX_METADATA_BYTES", parent_cost + child_cost)
    snap.apply([child])
    assert 1 in snap._known
    assert 2 in snap._known
    assert 9 not in snap._known
    assert counts(snap.export()) == Counter({("GPU", None, 2): 1})


@pytest.fixture
def publisher(monkeypatch):
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    endpoints = None

    def create():
        return ZmqEventPublisher(
            0,
            endpoint=endpoints[0].replace("127.0.0.1", "*")
            if endpoints
            else "tcp://*:0",
            snapshot_endpoint=endpoints[1] if endpoints else "tcp://*:0",
        )

    pub = create()
    config = pub.get_publisher_config()
    endpoints = (config.endpoint, config.snapshot_endpoint)
    yield pub, endpoints, create
    pub.shutdown()


def client(endpoints):
    return SnapshotClient(*endpoints)


def request(endpoints):
    with zmq.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoints[1])
        sock.send(b"snapshot")
        assert sock.poll(5000)
        return sock.recv_multipart()


def publish(pub, events):
    pub.publish(KVEventBatch(ts=time.time(), events=events))
    pub._event_queue.join()


def test_empty_snapshot_and_idle_subscription(publisher):
    pub, port, _ = publisher
    reply = request(port)
    assert int.from_bytes(reply[0], "big", signed=True) == -1
    assert reply[1] == pub._snapshot_stream_id
    with_client = client(port)
    try:
        seq, payloads = with_client.bootstrap()
        assert seq >= 0  # heartbeat establishes SUB delivery
        assert payloads == []
    finally:
        with_client.close()


def test_initial_live_message_obeys_bootstrap_buffer_limit(publisher, monkeypatch):
    _, endpoints, _ = publisher
    c = client(endpoints)
    monkeypatch.setattr(c, "MAX_BUFFER_BYTES", 1)
    try:
        with pytest.raises(ResyncRequired, match="buffer exhausted"):
            c.bootstrap()
        assert not c.ready
    finally:
        c.close()


def test_record_happens_before_send(publisher, monkeypatch):
    pub, port, _ = publisher
    entered, release = threading.Event(), threading.Event()
    record = pub._snapshot_recorder.record

    def blocked(seq, payload):
        record(seq, payload)
        entered.set()
        assert release.wait(5)

    c = client(port)
    c.bootstrap()
    monkeypatch.setattr(pub._snapshot_recorder, "record", blocked)
    try:
        pub.publish(KVEventBatch(ts=0, events=[stored([1])]))
        assert entered.wait(5)
        assert not c.sub.poll(100)
        reply = request(port)
        assert any(msgspec.msgpack.decode(chunk)[1] for chunk in reply[2:])
    finally:
        release.set()
        c.close()


def test_overflow_disables_snapshot_without_losing_live_batch(publisher, monkeypatch):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        monkeypatch.setattr(pub._snapshot_recorder, "MAX_PENDING_BYTES", 1)
        publish(pub, [stored([1])])
        assert c.poll() is not None
        assert int.from_bytes(request(port)[0], "big", signed=True) == -2
    finally:
        c.close()


def test_dequeue_releases_pending_bytes_before_capacity_check():
    first = msgspec.msgpack.encode(KVEventBatch(ts=0, events=[stored([1])]))
    second = msgspec.msgpack.encode(KVEventBatch(ts=0, events=[stored([2])]))
    recorder = KVEventSnapshotRecorder.__new__(KVEventSnapshotRecorder)
    recorder._inbox = queue.Queue()
    recorder._inbox.put_nowait((0, first))
    recorder._pending_bytes = len(first)
    recorder._lock = threading.Lock()
    recorder._snapshot = KVCacheSnapshot()
    recorder._seq = -1
    recorder._failed = threading.Event()
    recorder._stop = threading.Event()
    recorder.MAX_PENDING_BYTES = max(len(first), len(second))

    dequeued = threading.Event()
    release = threading.Event()
    get_nowait = recorder._inbox.get_nowait

    def blocked_get_nowait():
        item = get_nowait()
        dequeued.set()
        assert release.wait(5)
        return item

    recorder._inbox.get_nowait = blocked_get_nowait
    drain = threading.Thread(
        target=recorder._drain,
        args=(msgspec.msgpack.Decoder(type=KVEventBatch),),
    )
    drain.start()
    assert dequeued.wait(5)
    record = threading.Thread(target=recorder.record, args=(1, second))
    record.start()
    release.set()
    drain.join(5)
    record.join(5)

    assert not recorder._failed.is_set()
    assert recorder._pending_bytes == len(second)
    assert recorder._inbox.qsize() == 1


@pytest.mark.parametrize("budget", ["metadata", "reply"])
def test_resource_budget_fails_closed(publisher, monkeypatch, budget):
    pub, port, _ = publisher
    recorder = pub._snapshot_recorder
    if budget == "metadata":
        monkeypatch.setattr(recorder._snapshot, "MAX_METADATA_BYTES", 1)
    else:
        monkeypatch.setattr(recorder, "MAX_REPLY_BYTES", 1)
    publish(pub, [stored([1])])
    reply = request(port)
    assert int.from_bytes(reply[0], "big", signed=True) == -2
    assert len(reply) == 2


def test_concurrent_requests(publisher):
    from concurrent.futures import ThreadPoolExecutor

    pub, port, _ = publisher
    publish(pub, [stored([1, 2, 3])])
    with ThreadPoolExecutor(max_workers=8) as executor:
        replies = list(executor.map(lambda _: request(port), range(32)))
    for reply in replies:
        assert int.from_bytes(reply[0], "big", signed=True) >= 0
        events = (
            event
            for chunk in reply[2:]
            for event in msgspec.msgpack.decode(chunk, type=KVEventBatch).events
        )
        assert counts(events) == Counter({("GPU", None, h): 1 for h in (1, 2, 3)})


def test_data_parallel_rank_offsets_and_tags(random_port):
    live = f"inproc://snapshot-live-{random_port}"
    snapshot = f"inproc://snapshot-state-{random_port}"
    pub = ZmqEventPublisher(2, endpoint=live, snapshot_endpoint=snapshot)
    try:
        config = pub.get_publisher_config()
        assert config.snapshot_endpoint == snapshot + "_dp2"
        publish(pub, [stored([1])])
        reply = request((config.endpoint, config.snapshot_endpoint))
        assert reply[1] == pub._snapshot_stream_id
        assert (
            msgspec.msgpack.decode(reply[2], type=KVEventBatch).data_parallel_rank == 2
        )
    finally:
        pub.shutdown()


def test_replay_preserves_snapshot_stream_identity(random_port):
    pub = ZmqEventPublisher(
        0,
        endpoint=f"inproc://snapshot-live-{random_port}",
        replay_endpoint=f"inproc://snapshot-replay-{random_port}",
        snapshot_endpoint=f"inproc://snapshot-state-{random_port}",
    )
    try:
        publish(pub, [stored([1])])
        with zmq.Context.instance().socket(zmq.DEALER) as replay:
            replay.setsockopt(zmq.LINGER, 0)
            replay.connect(pub.get_publisher_config().replay_endpoint)
            replay.send_multipart([b"", (0).to_bytes(8, "big")])
            assert replay.poll(2000)
            frames = replay.recv_multipart()
            assert frames[2] == (0).to_bytes(8, "big") + pub._snapshot_stream_id
    finally:
        pub.shutdown()


def test_drain_takes_finite_cut():
    recorder = KVEventSnapshotRecorder.__new__(KVEventSnapshotRecorder)
    import queue

    recorder._inbox = queue.Queue()
    recorder._lock = threading.Lock()
    recorder._failed = threading.Event()
    payload = msgspec.msgpack.encode(KVEventBatch(ts=0, events=[]))
    recorder._inbox.put((0, payload))
    recorder._pending_bytes = len(payload)

    class Replenishing:
        def apply(self, events):
            recorder._inbox.put((1, payload))
            recorder._pending_bytes += len(payload)

    recorder._snapshot = Replenishing()
    recorder._drain(msgspec.msgpack.Decoder(type=KVEventBatch))
    assert recorder._seq == 0
    assert recorder._inbox.qsize() == 1


def test_midstream_bootstrap_converges(publisher):
    pub, port, _ = publisher
    batches = list(random_batches(0, 600))
    expected: Counter = Counter()
    for events in batches:
        counts(events, expected)

    def produce():
        for events in batches:
            publish(pub, events)
            time.sleep(0.001)

    producer = threading.Thread(target=produce)
    c = client(port)
    producer.start()
    try:
        time.sleep(0.1)
        _, payloads = c.bootstrap()
        producer.join()
        target = pub._buffer[-1][0] + 1
        while c.next_seq < target:
            payload = c.poll()
            if payload is not None:
                payloads.append(payload)
        actual: Counter = Counter()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        for payload in payloads:
            counts(decoder.decode(payload).events, actual)
        assert actual == expected
    finally:
        producer.join()
        c.close()


def test_gap_requires_fresh_snapshot(publisher):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        publish(pub, [stored([1])])
        assert c.sub.poll(2000)
        c.sub.recv_multipart()  # lose the last data batch
        with pytest.raises(ResyncRequired, match="gap"):
            # The idle heartbeat exposes the lost tail.
            while True:
                c.poll(2000)
        assert not c.ready
        _, payloads = c.bootstrap()
        assert counts(
            e
            for p in payloads
            for e in msgspec.msgpack.decode(p, type=KVEventBatch).events
        ) == Counter({("GPU", None, 1): 1})
    finally:
        c.close()


def test_restart_changes_epoch(publisher):
    pub, port, create = publisher
    c = client(port)
    replacement = None
    try:
        c.bootstrap()
        old = c.stream_id
        pub.shutdown()
        replacement = create()
        with pytest.raises(ResyncRequired, match="restart"):
            while True:
                c.poll(2000)
        c.bootstrap()
        assert c.stream_id != old
    finally:
        c.close()
        if replacement:
            replacement.shutdown()
