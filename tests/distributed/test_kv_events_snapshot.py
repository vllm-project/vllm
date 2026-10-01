# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import random
import threading
import time
import uuid
from collections import Counter

import msgspec
import pytest
import zmq

from examples.features.kv_events.kv_events_snapshot_subscriber import (
    ResyncRequired,
    SnapshotClient,
)
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
    ZmqEventPublisher,
)
from vllm.distributed.kv_events_snapshot import KVCacheSnapshot, KVEventSnapshotRecorder

pytestmark = pytest.mark.skip_global_cleanup


def stored(
    hashes,
    parent=None,
    medium="GPU",
    group=None,
    *,
    tokens=None,
    extra_keys=None,
    locality=None,
    ownership=None,
):
    if tokens is None:
        tokens = medium == "GPU"
    return BlockStored(
        block_hashes=hashes,
        parent_block_hash=parent,
        token_ids=[h * 4 + j for h in hashes for j in range(4)] if tokens else [],
        block_size=4 if tokens else 0,
        lora_id=None,
        lora_name=None,
        medium=medium,
        extra_keys=extra_keys,
        group_idx=group,
        locality=locality,
        ownership=ownership,
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


def test_dead_records_are_retained_for_recent_batches(monkeypatch):
    monkeypatch.setattr(KVCacheSnapshot, "RING_BATCHES", 8)
    snap = KVCacheSnapshot()
    for h in range(1, 101):
        snap.apply([stored([h]), BlockRemoved(block_hashes=[h], medium="GPU")])
    window = range(101 - KVCacheSnapshot.RING_BATCHES, 101)
    assert set(snap._records) == set(window)
    snap.apply([stored([window[0]], medium="CPU")])
    assert counts(wire(snap.export())) == Counter({("CPU", None, window[0]): 1})


def test_heartbeats_do_not_age_dead_records():
    snap = KVCacheSnapshot()
    snap.apply([stored([1]), BlockRemoved(block_hashes=[1], medium="GPU")])
    for _ in range(10 * KVCacheSnapshot.RING_BATCHES):
        snap.apply([])
    snap.apply([stored([1], medium="CPU")])
    assert counts(wire(snap.export())) == Counter({("CPU", None, 1): 1})


def test_ring_budget_drops_oldest_dead_records_early(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "MAX_RING_BLOCKS", 10)
    for h in range(1, 13):
        snap.apply([stored([h]), BlockRemoved(block_hashes=[h], medium="GPU")])
    assert set(snap._records) == set(range(3, 13))
    assert snap.expired_early == 2 and not snap.tainted


def test_ancestors_of_live_blocks_are_retained(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "RING_BATCHES", 0)
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


def test_unknown_parent_taints_until_its_children_leave(monkeypatch):
    monkeypatch.setattr(KVCacheSnapshot, "RING_BATCHES", 8)
    snap = KVCacheSnapshot()
    snap.apply([stored([2], parent=1)])
    assert snap.tainted == 1 and "reconstruction metadata" in snap.taint_reason
    snap.apply([BlockRemoved(block_hashes=[2], medium="GPU")])
    for h in range(10, 10 + KVCacheSnapshot.RING_BATCHES):
        snap.apply([stored([h])])
    assert not snap.tainted and 1 not in snap._records


def test_store_after_ring_window_taints_until_evicted(monkeypatch):
    """An offload store later than the ring window, as after a GPU-only
    prefix cache reset with stores in flight."""
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "RING_BATCHES", 1)
    snap.apply([stored([1]), BlockRemoved(block_hashes=[1], medium="GPU")])
    snap.apply([stored([9]), BlockRemoved(block_hashes=[9], medium="GPU")])
    assert set(snap._records) == {9}
    snap.apply([stored([1], medium="CPU")])
    assert snap.tainted == 1
    snap.apply([BlockRemoved(block_hashes=[1], medium="CPU")])
    snap.apply([stored([2])])
    assert not snap.tainted
    assert counts(wire(snap.export())) == Counter({("GPU", None, 2): 1})


def test_store_after_ring_window_heals_when_restated(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "RING_BATCHES", 1)
    snap.apply([stored([1]), BlockRemoved(block_hashes=[1], medium="GPU")])
    snap.apply([stored([9])])
    snap.apply([stored([1], medium="CPU")])
    assert snap.tainted == 1
    snap.apply([stored([1])])
    assert not snap.tainted
    consumer = RouterModel()
    consumer.apply(wire(snap.export()))
    assert consumer.refs == Counter(
        {
            (("cpu", None, None, None), 1): 1,
            (("gpu", None, None, None), 1): 1,
            (("gpu", None, None, None), 9): 1,
        }
    )


def test_remove_after_ring_window_is_counted(monkeypatch):
    snap = KVCacheSnapshot()
    monkeypatch.setattr(snap, "RING_BATCHES", 1)
    snap.apply([stored([1]), BlockRemoved(block_hashes=[1], medium="GPU")])
    snap.apply([stored([9])])
    snap.apply([BlockRemoved(block_hashes=[1], medium="CPU")])
    assert snap.forgotten_removals == 1 and not snap.tainted


def test_conflicting_metadata_taints_until_the_block_leaves(monkeypatch):
    monkeypatch.setattr(KVCacheSnapshot, "RING_BATCHES", 8)
    snap = KVCacheSnapshot()
    snap.apply([stored([1])])
    restated = stored([1])
    restated.token_ids[0] += 1
    snap.apply([restated])
    assert snap.tainted == 1 and "conflicting" in snap.taint_reason
    snap.apply([BlockRemoved(block_hashes=[1, 1], medium="GPU")])
    for h in range(10, 10 + KVCacheSnapshot.RING_BATCHES):
        snap.apply([stored([h])])
    assert not snap.tainted
    # A group with twice the block size names the second block's hash for
    # both blocks' tokens.
    snap = KVCacheSnapshot()
    snap.apply([stored([1, 2], group=0)])
    coarse = stored([1, 2], group=1)
    snap.apply(
        [
            BlockStored(
                block_hashes=[2],
                parent_block_hash=None,
                token_ids=coarse.token_ids,
                block_size=8,
                lora_id=None,
                lora_name=None,
                medium="GPU",
                group_idx=1,
            )
        ]
    )
    assert snap.tainted == 1 and "conflicting" in snap.taint_reason


def test_groups_share_block_records():
    """KV cache groups store the same hash with the same inputs. A store that
    skips null or masked blocks lists fewer hashes than its token span and
    only adds residency."""
    snap = KVCacheSnapshot()
    snap.apply([stored([1, 2, 3], group=0), stored([1, 2, 3], group=1)])
    full = stored([4, 5, 6], parent=3, group=0)
    window = BlockStored(
        block_hashes=[6],
        parent_block_hash=3,
        token_ids=full.token_ids,
        block_size=4,
        lora_id=None,
        lora_name=None,
        medium="GPU",
        group_idx=1,
    )
    snap.apply([full, window])
    assert not snap.tainted
    expected = Counter({("GPU", 0, h): 1 for h in range(1, 7)})
    expected.update(("GPU", 1, h) for h in (1, 2, 3, 6))
    assert counts(wire(snap.export())) == expected


def test_store_skipping_blocks_needs_another_store():
    window = BlockStored(
        block_hashes=[3],
        parent_block_hash=None,
        token_ids=stored([1, 2, 3]).token_ids,
        block_size=4,
        lora_id=None,
        lora_name=None,
        medium="GPU",
        group_idx=1,
    )
    snap = KVCacheSnapshot()
    snap.apply([window])
    assert snap.tainted == 1 and "no reconstruction metadata" in snap.taint_reason
    snap.apply([stored([1, 2, 3], group=0)])
    assert not snap.tainted
    # The same batch, in either order.
    snap = KVCacheSnapshot()
    snap.apply([window, stored([1, 2, 3], group=0)])
    assert not snap.tainted


def test_omitted_extra_keys_do_not_conflict():
    """Offload stores can omit extra keys. The recorder keeps keys a store
    stated, learns them from a later store, and compares them once known."""
    salted = stored([1], extra_keys=[("salt",)])
    offloaded = stored([1], medium="CPU", tokens=True)
    snap = KVCacheSnapshot()
    snap.apply([salted, offloaded])
    assert not snap.tainted
    assert wire(snap.export())[0].extra_keys == [("salt",)]

    snap = KVCacheSnapshot()
    snap.apply([offloaded])
    snap.apply([salted])
    assert not snap.tainted
    assert wire(snap.export())[0].extra_keys == [("salt",)]
    snap.apply([stored([1], extra_keys=[("other",)])])
    assert snap.tainted == 1


def test_records_are_not_tracked_by_the_garbage_collector():
    snap = KVCacheSnapshot()
    snap.apply([stored([1, 2], extra_keys=[("salt",), None])])
    for record in snap._records.values():
        assert not gc.is_tracked(record) and not gc.is_tracked(record.tokens)


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
    """The llm-d router's event consumer, reduced to its key semantics.

    Request keys chain over (parent request key, tokens, extra key, adapter).
    The router forgets an engine hash when it is evicted while no scope holds
    its request key, so an offload store that lands after that is lost; with
    `forget` off the model keeps every hash and holds the engine's residency.
    Stores and removes are reference counted per (tier, group, locality,
    ownership, hash) and only the last remove evicts. A store whose token
    span is not one block per hash, as from a group that skips null or masked
    blocks, only names blocks, like a token-less store. A strict consumer
    fails on any engine hash it cannot resolve; otherwise the event, or the
    hash of a token-less store, is skipped, as the router does.
    """

    def __init__(self, strict: bool = True, forget: bool = True):
        self.strict = strict
        self.forget = forget
        self.keys: dict = {}
        self.entries: set = set()
        self.refs: Counter = Counter()

    def resolve(self, h):
        if h not in self.keys and self.strict:
            raise ConsumerFailure(f"engine key not found: {h!r}")
        return self.keys.get(h)

    @staticmethod
    def scope(e):
        return ((e.medium or "GPU").lower(), e.group_idx, e.locality, e.ownership)

    def apply(self, events):
        for e in events:
            if isinstance(e, BlockStored):
                scope = self.scope(e)
                size = e.block_size
                if e.token_ids and len(e.token_ids) == size * len(e.block_hashes):
                    key = (
                        ()
                        if e.parent_block_hash is None
                        else self.resolve(e.parent_block_hash)
                    )
                    if key is None:
                        continue
                    for i, h in enumerate(e.block_hashes):
                        extra = e.extra_keys[i] if e.extra_keys else None
                        tokens = tuple(e.token_ids[i * size : (i + 1) * size])
                        key = hash((key, tokens, extra, e.lora_name))
                        self.keys[h] = key
                        self.entries.add((scope, key))
                    hashes = e.block_hashes
                else:
                    hashes = [h for h in e.block_hashes if self.resolve(h) is not None]
                    for h in hashes:
                        self.entries.add((scope, self.keys[h]))
                for h in hashes:
                    self.refs[(scope, h)] += 1
            elif isinstance(e, BlockRemoved):
                scope = self.scope(e)
                for h in e.block_hashes:
                    if self.refs[(scope, h)] > 1:
                        self.refs[(scope, h)] -= 1
                        continue
                    self.refs.pop((scope, h), None)
                    key = self.resolve(h)
                    if key is None:
                        continue
                    self.entries.discard((scope, key))
                    if self.forget and all(k != key for _, k in self.entries):
                        del self.keys[h]
            else:
                for scope, h in [k for k in self.refs if k[0][0] == "gpu"]:
                    self.entries.discard((scope, self.resolve(h)))
                    del self.refs[(scope, h)]

    def state(self):
        return frozenset(self.entries), frozenset((+self.refs).items())

    def resolved(self):
        """The request key of each resident engine hash."""
        return frozenset((h, self.keys.get(h)) for _, h in +self.refs)


def test_dead_alias_does_not_remove_live_block():
    """Hashes whose inputs differ only where the consumer does not look
    share a consumer key; removing the dead one must not evict the live one."""
    history = [
        [stored([1])],
        [BlockRemoved(block_hashes=[1], medium="GPU")],
        [
            BlockStored(
                block_hashes=[2],
                parent_block_hash=None,
                token_ids=[4, 5, 6, 7],
                block_size=4,
                lora_id=None,
                lora_name=None,
                medium="GPU",
            )
        ],
    ]
    reference, snap = RouterModel(), KVCacheSnapshot()
    for events in history:
        reference.apply(events)
        snap.apply(events)
    consumer = RouterModel()
    consumer.apply(wire(snap.export()))
    assert consumer.state() == reference.state() and reference.entries


def cache_history(
    seed,
    steps=300,
    lag=3,
    groups=False,
    adapters=False,
    self_describing=False,
    locality=None,
):
    """Prefix-sharing requests over a small GPU pool and an LRU CPU tier.

    GPU blocks are evicted tail first, CPU blocks head first, offload stores
    complete up to `lag` steps late (after their GPU copy may be gone), and a
    step's block-pool events precede its connector events, as in vLLM. Idle
    heartbeats come between steps.

    With `groups`, a second KV cache group stores each chunk after the first
    group, leaving out some blocks as sliding-window and Mamba groups do, and
    evicts on its own. `adapters` gives requests LoRA adapters and per-block
    extra keys, which enter the block hash. Offload stores carry the blocks'
    inputs with `self_describing`, and the CPU tier's `locality`.
    """
    rng = random.Random(seed)
    prefixes = [
        [rng.randrange(1000) for _ in range(4 * rng.randrange(1, 6))] for _ in range(8)
    ]
    gpu: dict = {}  # hash -> (refs, order)
    window: dict = {}  # second group: hash -> (refs, order)
    cpu: dict = {}  # hash -> order
    blocks: dict = {}  # hash -> (parent, tokens, adapter, extra key)
    pending: list = []  # (due step, hash)
    tick = 0
    for step in range(steps):
        pool_events, connector_events = [], []
        prompt = list(rng.choice(prefixes)) + [
            rng.randrange(1000) for _ in range(4 * rng.randrange(0, 8))
        ]
        adapter = salt = None
        if adapters:
            adapter = rng.choice((None, (1, "a"), (2, "b")))
            salt = rng.choice((None, ("salt-0",), ("salt-1",)))
        parent: int | None = None
        new: list[int] = []
        for i in range(0, len(prompt), 4):
            block = tuple(prompt[i : i + 4])
            extra = salt if i == 0 else None
            h = hash((parent, block, adapter, extra)) & ((1 << 63) - 1)
            blocks[h] = (parent, block, adapter, extra)
            tick += 1
            if new or h not in gpu:
                # The prefix hit ends at the first miss; every later block is
                # computed and cached again, a second copy if still resident.
                new.append(h)
                gpu[h] = (gpu.get(h, (0, 0))[0] + 1, tick)
            else:
                gpu[h] = (gpu[h][0], tick)
            parent = h

        def store(hashes, parent, tokens, medium, group, kind, adapter, **scope):
            return BlockStored(
                block_hashes=hashes,
                parent_block_hash=parent,
                token_ids=tokens,
                block_size=4 if tokens else 0,
                lora_id=adapter[0] if adapter and tokens else None,
                lora_name=adapter[1] if adapter and tokens else None,
                medium=medium,
                extra_keys=[blocks[h][3] for h in hashes] if tokens else None,
                group_idx=group,
                kv_cache_spec_kind=kind,
                **scope,
            )

        # store new blocks in chunks, each chunk chained to the previous block
        while new:
            n = rng.randrange(1, len(new) + 1)
            chunk, new = new[:n], new[n:]
            first_parent = blocks[chunk[0]][0]
            tokens = [t for h in chunk for t in blocks[h][1]]
            pool_events.append(
                store(chunk, first_parent, tokens, "GPU", 0, "full_attention", adapter)
            )
            if groups:
                # Left-out blocks leave no hash; token_ids keep the span.
                kept = [h for h in chunk if rng.random() < 0.7]
                for h in kept:
                    window[h] = (window.get(h, (0, 0))[0] + 1, tick)
                pool_events.append(
                    store(
                        kept, first_parent, tokens, "GPU", 1, "sliding_window", adapter
                    )
                )
            for h in chunk:
                if h not in cpu and rng.random() < 0.8:
                    pending.append((step + rng.randrange(0, lag + 1), h))
        # evict GPU down to 40 blocks, youngest first within the oldest request
        for pool, group, size in ((gpu, 0, 40), (window, 1, 30)):
            while len(pool) > size:
                victim = min(pool, key=lambda h: (pool[h][1] // 64, -pool[h][1]))
                refs = pool.pop(victim)[0]
                pool_events.append(
                    BlockRemoved(
                        block_hashes=[victim] * refs, medium="GPU", group_idx=group
                    )
                )
        # offload completions, then CPU LRU eviction oldest first
        due = [h for s, h in pending if s <= step]
        pending = [(s, h) for s, h in pending if s > step]
        for h in due:
            if h not in cpu:
                tick += 1
                cpu[h] = tick
                parent, tokens, block_adapter, _ = blocks[h]
                payload = list(tokens) if self_describing else []
                connector_events.append(
                    store(
                        [h],
                        parent if payload else None,
                        payload,
                        "CPU",
                        0,
                        None,
                        block_adapter,
                        locality=locality,
                    )
                )
        while len(cpu) > 120:
            victim = min(cpu, key=lambda h: cpu[h])
            del cpu[victim]
            connector_events.append(
                BlockRemoved(
                    block_hashes=[victim], medium="CPU", group_idx=0, locality=locality
                )
            )
        if rng.random() < 0.01:
            pool_events.append(AllBlocksCleared())
            gpu.clear()
            window.clear()
        yield pool_events + connector_events
        for _ in range(rng.choice((0, 0, 0, 1, 5, 40))):
            yield []


HISTORIES = {
    "plain": {},
    "groups": {"groups": True},
    "adapters": {"adapters": True},
    "offload": {"adapters": True, "self_describing": True, "locality": "LOCAL"},
    "everything": {"groups": True, "adapters": True, "self_describing": True},
}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("variant", HISTORIES)
def test_snapshot_then_live_follows_the_full_stream(variant, seed):
    """Load a snapshot at every cut and follow the live stream for longer
    than a dead block's record is kept. A consumer that keeps engine hashes
    matches one that read the whole stream, and resolves every late offload
    store, including those of blocks that died before the cut. A router that
    forgets hashes matches at the cut and never holds residency the engine
    lacks."""
    lag = 3
    history = list(cache_history(seed, steps=150, lag=lag, **HISTORIES[variant]))
    full_stream = RouterModel(forget=False)
    states = []
    for events in history:
        full_stream.apply(events)
        states.append(
            (full_stream.state(), full_stream.resolved(), Counter(full_stream.refs))
        )
    carrying = [i for i, events in enumerate(history) if events]
    follow = KVCacheSnapshot.RING_BATCHES + lag + 1
    snap = KVCacheSnapshot()
    for n, cut in enumerate(carrying):
        for events in history[carrying[n - 1] + 1 if n else 0 : cut + 1]:
            snap.apply(events)
        assert not snap.tainted, snap.taint_reason
        exported = wire(snap.export(max_blocks_per_event=5))
        exact, router = RouterModel(forget=False), RouterModel(strict=False)
        exact.apply(exported)
        router.apply(exported)
        state, resolved, _ = states[cut]
        assert (exact.state(), exact.resolved()) == (state, resolved)
        assert router.state() == state
        end = carrying[min(n + follow, len(carrying) - 1)]
        for events in history[cut + 1 : end + 1]:
            exact.apply(events)
            router.apply(events)
        state, resolved, refs = states[end]
        assert (exact.state(), exact.resolved()) == (state, resolved)
        assert router.entries <= state[0] and router.refs <= refs


@pytest.mark.parametrize("seed", range(3))
def test_short_ring_is_unavailable_then_heals(monkeypatch, seed):
    """With a ring that cannot cover the offload lag, snapshots are
    unavailable while a block cannot be rebuilt, and every snapshot exported
    while available reproduces the full-history state."""
    monkeypatch.setattr(KVCacheSnapshot, "RING_BATCHES", 1)
    reference = RouterModel(forget=False)
    snap = KVCacheSnapshot()
    available = []
    for events in cache_history(seed, lag=5):
        reference.apply(events)
        snap.apply(events)
        available.append(not snap.tainted)
        if not snap.tainted:
            consumer = RouterModel()
            consumer.apply(wire(snap.export()))
            assert consumer.state() == reference.state()
    first_taint = available.index(False)
    assert any(available[first_taint:])


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


@pytest.fixture
def idle_recorder():
    """A recorder whose thread is stopped, so the test drives it."""
    recorder = KVEventSnapshotRecorder(f"inproc://snapshot-{uuid.uuid4().hex}", 0)
    recorder._stop.set()
    recorder._thread.join(5)
    recorder._stop.clear()
    yield recorder
    recorder.shutdown(timeout=1)


def encoded(events):
    return msgspec.msgpack.encode(KVEventBatch(ts=0, events=events))


def test_record_waits_for_room(idle_recorder, monkeypatch):
    monkeypatch.setattr(idle_recorder, "MAX_PENDING_BATCHES", 1)
    idle_recorder.record(0, encoded([stored([1])]))
    waiting = threading.Thread(
        target=idle_recorder.record, args=(1, encoded([stored([2])]))
    )
    waiting.start()
    waiting.join(0.2)
    assert waiting.is_alive()
    idle_recorder._drain(msgspec.msgpack.Decoder(type=KVEventBatch))
    waiting.join(5)
    assert not waiting.is_alive() and not idle_recorder._failed.is_set()
    assert [seq for seq, _ in idle_recorder._inbox] == [1]


def test_recorder_that_stays_behind_fails_without_losing_live_batch(
    publisher, monkeypatch
):
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        recorder = pub._snapshot_recorder
        monkeypatch.setattr(recorder, "MAX_RECORD_WAIT_S", 0.01)
        monkeypatch.setattr(recorder, "_fits", lambda size: False)
        publish(pub, [stored([1])])
        assert c.poll() is not None
        assert int.from_bytes(request(port)[0], "big", signed=True) == -2
    finally:
        c.close()


def test_unavailable_period_keeps_identity_and_live_followers(publisher, monkeypatch):
    """A consumer following the live stream keeps its state while snapshots
    are unavailable and after they heal; requesters retry."""
    monkeypatch.setattr(KVCacheSnapshot, "RING_BATCHES", 8)
    pub, port, _ = publisher
    c = client(port)
    try:
        c.bootstrap()
        identity = c.stream_id
        publish(pub, [stored([1], medium="CPU")])
        reply = request(port)
        assert int.from_bytes(reply[0], "big", signed=True) == -2
        assert reply[1] == identity
        publish(pub, [BlockRemoved(block_hashes=[1], medium="CPU")])
        for h in range(10, 10 + KVCacheSnapshot.RING_BATCHES):
            publish(pub, [stored([h])])
        reply = request(port)
        assert int.from_bytes(reply[0], "big", signed=True) >= 0
        assert reply[1] == identity == pub._snapshot_stream_id
        target = pub._buffer[-1][0] + 1
        while c.next_seq < target:
            c.poll()
        assert c.ready and c.stream_id == identity
    finally:
        c.close()


@pytest.mark.parametrize(
    "limit", ["snapshot_max_blocks", "snapshot_max_response_bytes"]
)
def test_configured_budget_fails_closed(random_port, limit):
    config = KVEventsConfig(
        enable_kv_cache_events=True,
        endpoint=f"inproc://snapshot-live-{random_port}",
        snapshot_endpoint=f"inproc://snapshot-state-{random_port}",
        **{limit: 1},
    )
    pub = EventPublisherFactory.create(config)
    try:
        resolved = pub.get_publisher_config()
        assert getattr(resolved, limit) == 1
        publish(pub, [stored([1, 2])])
        reply = request((resolved.endpoint, resolved.snapshot_endpoint))
        assert int.from_bytes(reply[0], "big", signed=True) == -2
        assert len(reply) == 2
    finally:
        pub.shutdown()


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


def test_drain_takes_finite_cut(idle_recorder, monkeypatch):
    payload = encoded([])
    idle_recorder.record(0, payload)
    apply = idle_recorder._snapshot.apply

    def replenishing(events):
        idle_recorder.record(1, payload)
        apply(events)

    monkeypatch.setattr(idle_recorder._snapshot, "apply", replenishing)
    idle_recorder._drain(msgspec.msgpack.Decoder(type=KVEventBatch))
    assert idle_recorder._seq == 0
    assert [seq for seq, _ in idle_recorder._inbox] == [1]


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
