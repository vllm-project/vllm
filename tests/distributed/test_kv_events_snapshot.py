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


def wire(events):
    """Decode exported events as a snapshot consumer receives them."""
    batch = msgspec.msgpack.encode(KVEventBatch(ts=0, events=list(events)))
    return msgspec.msgpack.decode(batch, type=KVEventBatch).events


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
            assert counts(wire(snap.export())) == expected
    assert counts(wire(snap.export())) == expected


def test_dead_records_are_retained_for_recent_batches():
    snap = KVCacheSnapshot()
    for h in range(1, 101):
        snap.apply([stored([h]), BlockRemoved(block_hashes=[h], medium="GPU")])
    window = range(101 - KVCacheSnapshot.RING_BATCHES, 101)
    assert set(snap._records) == set(window)
    snap.apply([stored([window[0]], medium="CPU")])
    assert counts(wire(snap.export())) == Counter({("CPU", None, window[0]): 1})


def test_dead_records_are_retained_within_ring(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "RING_BATCHES", 10**9)
    for h in range(1, 1001):
        snap.apply([stored([h])])
        snap.apply([BlockRemoved(block_hashes=[h], medium="GPU")])
    assert len(snap._records) == 1000 and snap._ring_size == 1000
    monkeypatch.setattr(snap, "MAX_RING_BLOCKS", 10)
    snap.apply([])
    assert len(snap._records) == 10 and set(snap._records) == set(range(991, 1001))
    monkeypatch.setattr(snap, "MAX_RING_BLOCKS", 0)
    snap.apply([])
    assert not snap._records and not snap._ring_size


def test_ancestors_of_live_blocks_are_retained(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "MAX_RING_BLOCKS", 0)
    for h in range(1, 1501):
        snap.apply([stored([h], parent=h - 1 if h > 1 else None)])
        if h > 1:
            snap.apply([BlockRemoved(block_hashes=[h - 1], medium="GPU")])
    assert len(snap) == 1
    assert len(snap._records) == 1500
    exported = wire(snap.export())
    assert counts(exported) == Counter({("GPU", None, 1500): 1})
    snap.apply([BlockRemoved(block_hashes=[1500], medium="GPU")])
    assert not snap._records


def test_delayed_transfer_preserves_metadata():
    snap = KVCacheSnapshot()
    snap.apply([stored([1, 2])])
    snap.apply([BlockRemoved(block_hashes=[1, 2], medium="GPU")])
    snap.apply([stored([2], parent=1, medium="CPU")])
    exported = wire(snap.export())
    assert isinstance(exported[0], BlockStored) and exported[0].token_ids
    assert counts(exported) == Counter({("CPU", None, 2): 1})


def test_unknown_parent_without_known_block_fails_closed():
    with pytest.raises(ValueError, match="Missing reconstruction"):
        KVCacheSnapshot().apply([stored([2], parent=1)])


def test_store_after_ring_overflow_fails_closed(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "MAX_RING_BLOCKS", 1)
    snap.apply([stored([1]), BlockRemoved(block_hashes=[1], medium="GPU")])
    snap.apply([stored([9]), BlockRemoved(block_hashes=[9], medium="GPU")])
    assert set(snap._records) == {9}
    with pytest.raises(ValueError, match="Missing reconstruction"):
        snap.apply([stored([1], medium="CPU")])


def test_offloaded_history_exports_live_state():
    """A full CPU tier keeps records for evicted GPU stores, not the events."""
    rng = random.Random(0)
    snap = KVCacheSnapshot()
    expected: Counter = Counter()
    for prompt in range(2 * 29_093 // 64 // 20 + 1):
        hashes = list(range(1 + 36 * prompt, 1 + 36 * (prompt + 1)))
        tokens = [rng.randrange(151_552) for _ in range(64 * len(hashes))]
        offloaded = [stored([h], medium="CPU") for h in hashes[-20:]]
        history = [
            BlockStored(
                block_hashes=hashes,
                parent_block_hash=None,
                token_ids=tokens,
                block_size=64,
                lora_id=None,
                lora_name=None,
                medium="GPU",
            ),
            *offloaded,
            BlockRemoved(block_hashes=hashes, medium="GPU"),
        ]
        snap.apply(history)
        counts(history, expected)
    assert counts(wire(snap.export())) == expected


class ConsumerFailure(Exception):
    pass


class RouterModel:
    """The llm-d router's snapshot consumer, reduced to its key semantics.

    Request keys chain over (parent request key, tokens, extra key). A strict
    consumer fails on any engine hash it cannot resolve, as a snapshot pool
    does for its whole life; stores and removes are reference counted per
    (tier, group, hash) and only the last remove evicts.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.keys: dict = {}
        self.entries: set = set()
        self.refs: Counter = Counter()

    def resolve(self, h):
        if h not in self.keys:
            raise ConsumerFailure(f"engine key not found: {h!r}")
        return self.keys[h]

    def apply(self, events):
        for e in events:
            if isinstance(e, BlockStored):
                scope = ((e.medium or "GPU").lower(), e.group_idx)
                if e.token_ids:
                    key = (
                        ()
                        if e.parent_block_hash is None
                        else self.resolve(e.parent_block_hash)
                    )
                    size = e.block_size
                    for i, h in enumerate(e.block_hashes):
                        extra = e.extra_keys[i] if e.extra_keys else None
                        key = hash(
                            (key, tuple(e.token_ids[i * size : (i + 1) * size]), extra)
                        )
                        self.keys[h] = key
                        self.entries.add((scope, key))
                else:
                    for h in e.block_hashes:
                        self.entries.add((scope, self.resolve(h)))
                for h in e.block_hashes:
                    self.refs[(scope, h)] += 1
            elif isinstance(e, BlockRemoved):
                scope = ((e.medium or "GPU").lower(), e.group_idx)
                for h in e.block_hashes:
                    if self.refs[(scope, h)] > 1:
                        self.refs[(scope, h)] -= 1
                        continue
                    self.refs.pop((scope, h), None)
                    self.entries.discard((scope, self.resolve(h)))
            else:
                for scope, h in [k for k in self.refs if k[0][0] == "gpu"]:
                    self.entries.discard((scope, self.resolve(h)))
                    del self.refs[(scope, h)]

    def state(self):
        return self.entries, +self.refs


def cache_history(seed, steps=300, lag=3):
    """Prefix-sharing requests over a small GPU pool and an LRU CPU tier.

    GPU blocks are evicted tail first, CPU blocks head first, offload stores
    complete up to `lag` steps late (after their GPU copy may be gone), and a
    step's block-pool events precede its connector events, as in vLLM.
    """
    rng = random.Random(seed)
    prefixes = [
        [rng.randrange(1000) for _ in range(4 * rng.randrange(1, 6))] for _ in range(8)
    ]
    gpu: dict = {}  # hash -> (refs, order)
    cpu: dict = {}  # hash -> order
    pending: list = []  # (due step, hash)
    tick = 0
    for step in range(steps):
        pool_events, connector_events = [], []
        prompt = list(rng.choice(prefixes)) + [
            rng.randrange(1000) for _ in range(4 * rng.randrange(0, 8))
        ]
        parent: int | None = None
        new: list[tuple[int, int | None, tuple[int, ...]]] = []
        for i in range(0, len(prompt), 4):
            block = tuple(prompt[i : i + 4])
            h = hash((parent, block)) & ((1 << 63) - 1)
            tick += 1
            if new or h not in gpu:
                # The prefix hit ends at the first miss; every later block is
                # computed and cached again, a second copy if still resident.
                new.append((h, parent, block))
                gpu[h] = (gpu.get(h, (0, 0))[0] + 1, tick)
            else:
                gpu[h] = (gpu[h][0], tick)
            parent = h
        # store new blocks in chunks, each chunk chained to the previous block
        while new:
            n = rng.randrange(1, len(new) + 1)
            chunk, new = new[:n], new[n:]
            pool_events.append(
                BlockStored(
                    block_hashes=[h for h, _, _ in chunk],
                    parent_block_hash=chunk[0][1],
                    token_ids=[t for _, _, b in chunk for t in b],
                    block_size=4,
                    lora_id=None,
                    lora_name=None,
                    medium="GPU",
                    extra_keys=[None] * len(chunk),
                    group_idx=0,
                    kv_cache_spec_kind="full_attention",
                )
            )
            for h, _, _ in chunk:
                if h not in cpu and rng.random() < 0.8:
                    pending.append((step + rng.randrange(0, lag + 1), h))
        # evict GPU down to 40 blocks, youngest first within the oldest request
        while len(gpu) > 40:
            victim = min(gpu, key=lambda h: (gpu[h][1] // 64, -gpu[h][1]))
            refs = gpu.pop(victim)[0]
            pool_events.append(
                BlockRemoved(block_hashes=[victim] * refs, medium="GPU", group_idx=0)
            )
        # offload completions, then CPU LRU eviction oldest first
        due = [h for s, h in pending if s <= step]
        pending = [(s, h) for s, h in pending if s > step]
        for h in due:
            if h not in cpu:
                tick += 1
                cpu[h] = tick
                connector_events.append(
                    BlockStored(
                        block_hashes=[h],
                        parent_block_hash=None,
                        token_ids=[],
                        block_size=0,
                        lora_id=None,
                        lora_name=None,
                        medium="CPU",
                        group_idx=0,
                    )
                )
        while len(cpu) > 120:
            victim = min(cpu, key=lambda h: cpu[h])
            del cpu[victim]
            connector_events.append(
                BlockRemoved(block_hashes=[victim], medium="CPU", group_idx=0)
            )
        if rng.random() < 0.01:
            pool_events.append(AllBlocksCleared())
            gpu.clear()
        yield pool_events + connector_events


@pytest.mark.parametrize("seed", range(20))
def test_strict_consumer_follows_any_cut(seed, monkeypatch):
    monkeypatch.setattr(KVCacheSnapshot, "MAX_RING_BLOCKS", 512)
    history = list(cache_history(seed))
    reference = RouterModel()
    for events in history:
        reference.apply(events)
    snap = KVCacheSnapshot()
    for cut, events in enumerate(history):
        snap.apply(events)
        if cut % 7:
            continue
        consumer = RouterModel()
        consumer.apply(wire(snap.export(max_blocks_per_event=5)))
        for later in history[cut + 1 :]:
            consumer.apply(later)
        assert consumer.state() == reference.state()


def test_ring_too_small_fails_recorder_before_consumer(monkeypatch):
    """With a ring that cannot cover the offload lag, the recorder fails
    closed on the same event a strict consumer would reject."""
    monkeypatch.setattr(KVCacheSnapshot, "MAX_RING_BLOCKS", 0)
    snap = KVCacheSnapshot()
    with pytest.raises(ValueError, match="Missing reconstruction"):
        for events in cache_history(0, lag=5):
            snap.apply(events)


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


@pytest.mark.parametrize("budget", ["records", "reply"])
def test_resource_budget_fails_closed(publisher, monkeypatch, budget):
    pub, port, _ = publisher
    recorder = pub._snapshot_recorder
    if budget == "records":
        monkeypatch.setattr(recorder._snapshot, "MAX_RECORDS", 0)
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
