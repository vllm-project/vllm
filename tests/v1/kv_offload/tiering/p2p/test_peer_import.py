# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bulk import of a peer's CPU-tier cache over real P2P sessions."""

import json
import mmap
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.tiering.p2p.test_manager import (  # noqa: F401
    drain_peers,
    drain_transports,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    StoreTrigger,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering import peer_import as peer_import_module
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.p2p.session import session as session_module
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    ListKeysMsg,
    ListKeysRespMsg,
)
from vllm.v1.kv_offload.tiering.peer_import import PeerImport

_STEP = ScheduleEndContext([], ())


def _tiering(tier, num_chunks=8):
    region = SharedOffloadRegion(
        engine_id=str(uuid.uuid4()),
        num_chunks=num_chunks,
        rank=0,
        kv_bytes_per_chunk=mmap.PAGESIZE,
        cpu_page_size=mmap.PAGESIZE,
    )
    manager = TieringOffloadingManager(
        CPUPrimaryTierOffloadingManager(num_chunks=num_chunks, mmap_region=region),
        [tier],
        tier_poll_interval_s=0,
    )
    return manager, region


def _keys(prefix, n):
    return [OffloadKey(f"{prefix}-{i}".encode()) for i in range(n)]


def _seed(manager, keys):
    ctx = ReqContext(req_id="seed")
    manager.on_new_request(ctx)
    store = manager.prepare_store(keys, ctx, trigger=StoreTrigger.ON_COMPUTE)
    assert store is not None
    manager.complete_store(store.keys_to_store, ctx, trigger=StoreTrigger.ON_COMPUTE)
    manager.on_request_finished(ctx)
    manager.on_schedule_end(_STEP)


def _step(consumer, producer):
    consumer.on_schedule_end(_STEP)
    producer.on_schedule_end(_STEP)
    time.sleep(0.001)  # The source orders its listing on another thread.


def _run(consumer, producer, import_id, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _step(consumer, producer)
        (status,) = consumer.peer_import_status(import_id)
        if status["state"] in ("done", "failed"):
            return status
    raise AssertionError(f"import did not finish: {status}")


def _import(consumer, producer, **kwargs):
    started = consumer.start_peer_import("B", 2, **kwargs)
    return _run(consumer, producer, started["import_id"])


def _hits(manager, keys):
    ctx = ReqContext(req_id="check")
    return [manager.primary_tier.lookup(k, ctx) is LookupResult.HIT for k in keys]


@pytest.fixture
def pair(drain_peers):  # noqa: F811
    built = [_tiering(tier) for tier in drain_peers]
    try:
        yield tuple(manager for manager, _ in built)
    finally:
        for manager, region in built:
            manager.shutdown()
            region.cleanup()


@pytest.mark.parametrize("num_keys", [0, 6])
def test_import_copies_peer_cache_in_recency_order(pair, monkeypatch, num_keys):
    """Several pages and batches; the source's hottest keys stay hottest."""
    consumer, producer = pair
    monkeypatch.setattr(peer_import_module, "_LIST_PAGE", 2)
    monkeypatch.setattr(peer_import_module, "_MAX_BATCH_CHUNKS", 2)
    keys = _keys("key", num_keys)
    for key in keys:  # One per step: each a different recency.
        _seed(producer, [key])
    looked_up = []
    monkeypatch.setattr(
        consumer._metrics, "_observe_resolved_lookup", lambda *a: looked_up.append(a)
    )

    started = consumer.start_peer_import("B", 2)
    assert consumer.has_pending_work()
    status = _run(consumer, producer, started["import_id"])

    assert status["state"] == "done", status
    assert status["listed"] == status["imported"] == num_keys
    assert status["missed"] == status["failed"] == 0
    assert all(_hits(consumer, keys))
    order = producer.primary_tier._policy.key_order_snapshot()()
    assert consumer.primary_tier._policy.key_order_snapshot()() == order
    assert looked_up == []  # An import is not traffic.
    assert consumer._peer_imports[started["import_id"]].keys == []  # Freed.


def test_max_chunks_takes_most_recent_and_skips_present(pair):
    consumer, producer = pair
    old, new = _keys("old", 3), _keys("new", 3)
    _seed(producer, old)
    _seed(producer, new)
    _seed(consumer, new[:1])

    status = _import(consumer, producer, max_chunks=3)

    assert status["state"] == "done", status
    assert status["listed"] == 3
    assert status["already_present"] == 1
    assert status["imported"] == 2
    assert all(_hits(consumer, new))
    assert not any(_hits(consumer, old))


def test_batch_waits_for_slow_peer_answers(pair):
    """Promoting before every probe is answered would drop the rest."""
    consumer, producer = pair
    _seed(producer, _keys("key", 4))
    started = consumer.start_peer_import("B", 2)
    imp = consumer._peer_imports[started["import_id"]]
    deadline = time.monotonic() + 10.0
    while not imp._batches and time.monotonic() < deadline:
        _step(consumer, producer)
    assert imp._batches, "import never started a batch"
    for _ in range(5):  # The consumer steps while its probes are unanswered.
        consumer.on_schedule_end(_STEP)

    status = _run(consumer, producer, started["import_id"])
    assert status["imported"] == 4
    assert status["missed"] == 0


def test_small_tier_imports_without_losing_keys(drain_peers):  # noqa: F811
    """In-flight reservations must fit the tier, or keys are dropped."""
    (consumer, consumer_region), (producer, producer_region) = (
        _tiering(drain_peers[0], num_chunks=4),
        _tiering(drain_peers[1], num_chunks=8),
    )
    try:
        _seed(producer, _keys("key", 8))
        status = _import(consumer, producer)
        assert status["state"] == "done", status
        assert status["listed"] == status["imported"] == 4
        assert status["missed"] == status["failed"] == 0
    finally:
        for manager, region in (
            (consumer, consumer_region),
            (producer, producer_region),
        ):
            manager.shutdown()
            region.cleanup()


class _DeadPeerTier:
    """A peer whose session dies before the handshake (e.g. NIXL rejects it)."""

    block_size_bytes = 4096

    def __init__(self):
        self.opens = 0

    def peer_ready(self, peer_id, connect=False):
        self.opens += connect
        return False if connect else None

    def peer_down_reason(self, peer_id):
        return "block_len mismatch"

    def take_peer_keys(self, peer_id, list_id, abandon=False):
        return None


def test_dead_session_fails_import_without_reconnecting():
    tier = _DeadPeerTier()
    imp = PeerImport("i", "B", 2, tier, max_chunks=8, timeout_s=60, capacity=8)
    for _ in range(5):
        imp.advance(SimpleNamespace())
    assert tier.opens == 1
    assert imp.state == "failed"
    assert imp.error == (
        "session to peer closed before the handshake completed: block_len mismatch"
    )


class _BusyPeerTier(_DeadPeerTier):
    """A peer at its session limit: opening a session fails, with no cause."""

    def __init__(self, busy_for):
        super().__init__()
        self.busy_for = busy_for

    def peer_ready(self, peer_id, connect=False):
        self.opens += connect
        return None if self.opens <= self.busy_for else True

    def peer_down_reason(self, peer_id):
        return None


def test_transient_connect_failure_is_retried(monkeypatch):
    monkeypatch.setattr(peer_import_module, "_CONNECT_RETRY_S", 0)
    tier = _BusyPeerTier(busy_for=2)
    imp = PeerImport("i", "B", 2, tier, max_chunks=8, timeout_s=60, capacity=8)
    for _ in range(3):
        imp.advance(SimpleNamespace())
    assert (tier.opens, imp.state) == (3, "importing")

    imp = PeerImport("j", "B", 2, _BusyPeerTier(busy_for=99), 8, 60, capacity=8)
    imp.started_at -= peer_import_module._CONNECT_TIMEOUT_S + 1
    imp.advance(SimpleNamespace())
    assert imp.error == "could not open a session to peer"


def test_session_lost_mid_import_fails_the_import(pair):
    consumer, producer = pair
    _seed(producer, _keys("key", 4))
    started = consumer.start_peer_import("B", 2)
    imp = consumer._peer_imports[started["import_id"]]
    for _ in range(50):
        if imp.state == "importing":
            break
        _step(consumer, producer)
    consumer.secondary_tiers[0]._sessions["B:2"]._mark_dead("test cause")
    consumer.on_schedule_end(_STEP)
    assert imp.state == "failed"
    assert imp.error == "session to peer lost: test cause"


def test_source_that_cannot_list_fails_the_import(pair, monkeypatch):
    """A listing failure on the source must not break its scheduler step."""
    consumer, producer = pair
    _seed(producer, _keys("key", 1))

    def cannot_list():
        raise NotImplementedError("custom policy")

    monkeypatch.setattr(producer.primary_tier, "key_snapshot", cannot_list)
    status = _import(consumer, producer)
    assert status["state"] == "failed", status
    assert "cannot list" in status["error"]


def test_listing_lost_on_peer_is_not_reported_done(pair, monkeypatch):
    consumer, producer = pair
    monkeypatch.setattr(peer_import_module, "_LIST_PAGE", 2)
    _seed(producer, _keys("key", 6))
    started = consumer.start_peer_import("B", 2)
    imp = consumer._peer_imports[started["import_id"]]
    deadline = time.monotonic() + 10.0
    while not imp.keys and time.monotonic() < deadline:
        _step(consumer, producer)
    assert 0 < len(imp.keys) < 6
    for session in producer.secondary_tiers[0]._sessions.values():
        session._listing_snapshots.clear()
    status = _run(consumer, producer, started["import_id"])
    assert status["state"] == "failed", status
    assert "listing not found" in status["error"]


def test_listing_requests_are_bounded(pair, monkeypatch):
    """A peer cannot pile up snapshots: each copies the key set under lock."""
    consumer, producer = pair
    _import(consumer, producer)  # Opens the sessions.
    monkeypatch.setattr(session_module, "_MIN_SNAPSHOT_GAP_S", 0)
    session = next(iter(producer.secondary_tiers[0]._sessions.values()))
    sent: list[dict] = []
    monkeypatch.setattr(session, "_send", sent.append)
    release = threading.Event()
    snapshots = []

    def key_snapshot():
        snapshots.append(1)
        return lambda: release.wait(10) and [b"k"]

    parent = SimpleNamespace(key_snapshot=key_snapshot)

    def ask(list_id):
        session._dispatch_message(
            {
                "type": ListKeysMsg.TYPE,
                ListKeysMsg.LIST_ID: list_id,
                ListKeysMsg.CURSOR: 0,
                ListKeysMsg.LIMIT: 8,
            }
        )

    for list_id in ("a", "a", "b"):
        ask(list_id)
    session._serve_listings(parent)
    assert len(snapshots) == 1  # "a" once; "b" waits for it.
    monkeypatch.setattr(session_module, "_MAX_PENDING_LISTINGS", 3)
    ask("c")
    assert sent[-1][ListKeysRespMsg.ERROR] == "too many listing requests"
    release.set()

    def page_for(list_id):
        return next((m for m in sent if m[ListKeysRespMsg.LIST_ID] == list_id), None)

    def serve_until(list_id):
        deadline = time.monotonic() + 10.0
        while page_for(list_id) is None and time.monotonic() < deadline:
            session._serve_listings(parent)
            time.sleep(0.001)

    serve_until("a")
    monkeypatch.setattr(session_module, "_MIN_SNAPSHOT_GAP_S", 60)
    for _ in range(100):
        session._serve_listings(parent)
        time.sleep(0.002)
    assert page_for("b") is None  # Too soon after the last snapshot.
    monkeypatch.setattr(session_module, "_MIN_SNAPSHOT_GAP_S", 0)
    serve_until("b")
    assert page_for("b")[ListKeysRespMsg.KEYS] == [b"k"]


def test_ended_import_stops_waiting_for_its_page(pair):
    consumer, producer = pair
    _seed(producer, _keys("key", 1))
    started = consumer.start_peer_import("B", 2)
    imp = consumer._peer_imports[started["import_id"]]
    for _ in range(1000):  # Stop once asked, before the peer can answer.
        consumer.on_schedule_end(_STEP)
        if imp._page_requested_at is not None:
            break
        producer.on_schedule_end(_STEP)
        time.sleep(0.001)
    session = consumer.secondary_tiers[0]._sessions["B:2"]
    assert imp.import_id in session._requested_listings
    consumer.reset_cache()
    assert imp.import_id not in session._requested_listings


def test_unrequested_key_pages_are_dropped(pair):
    consumer, producer = pair
    _import(consumer, producer)
    session = consumer.secondary_tiers[0]._sessions["B:2"]
    session._dispatch_message(
        {
            "type": ListKeysRespMsg.TYPE,
            ListKeysRespMsg.LIST_ID: "never-asked",
            ListKeysRespMsg.KEYS: [b"k"],
            ListKeysRespMsg.NEXT_CURSOR: -1,
        }
    )
    assert session.take_key_page("never-asked") is None


def test_import_waits_out_a_cache_reset_and_is_aborted_by_one(pair):
    consumer, producer = pair
    _seed(producer, _keys("key", 1))
    consumer.reset_cache()
    started = consumer.start_peer_import("B", 2)
    for _ in range(20):
        _step(consumer, producer)
    (status,) = consumer.peer_import_status(started["import_id"])
    assert status["imported"] == status["missed"] == 0  # Held, not missed.
    consumer.on_reset_flushed()
    assert _run(consumer, producer, started["import_id"])["imported"] == 1

    started = consumer.start_peer_import("B", 2)
    consumer.reset_cache()
    (status,) = consumer.peer_import_status(started["import_id"])
    assert (status["state"], status["error"]) == ("failed", "cache reset")


def test_one_import_at_a_time_from_the_same_peer_rank(pair):
    consumer, _ = pair
    consumer.secondary_tiers[0]._dp_index = 3
    first = consumer.start_peer_import("B", 7777)
    assert first["peer"] == "B:7780"  # Base port plus this rank's DP index.
    assert consumer.start_peer_import("B", 7777)["import_id"] == first["import_id"]
    with pytest.raises(ValueError, match="still running"):
        consumer.start_peer_import("C", 7777)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": "", "base_port": 2},
        {"host": "B", "base_port": 0},
        {"host": "B", "base_port": 70000},
        {"host": "B", "base_port": 2, "max_chunks": 0},
        {"host": "B", "base_port": 2, "timeout_s": 0},
        {"host": "B", "base_port": 2, "timeout_s": float("inf")},
        {"host": "B", "base_port": 2, "timeout_s": 7200},
    ],
)
def test_bad_import_arguments_are_rejected(pair, kwargs):
    consumer, _ = pair
    with pytest.raises(ValueError):
        consumer.start_peer_import(**kwargs)
    assert not consumer._peer_imports


@pytest.mark.parametrize(
    "msg_type, msg",
    [
        (ListKeysMsg, {"list_id": "x", "cursor": -1, "limit": 1}),
        (ListKeysMsg, {"list_id": "x", "cursor": 0, "limit": 0}),
        (ListKeysMsg, {"list_id": 1, "cursor": 0, "limit": 1}),
        (ListKeysRespMsg, {"list_id": "x", "keys": [], "next_cursor": -2}),
        (ListKeysRespMsg, {"list_id": "x", "keys": ["str"], "next_cursor": -1}),
        (ListKeysRespMsg, {"list_id": "x", "keys": [b"k"] * 3, "next_cursor": -1}),
    ],
)
def test_bad_listing_messages_are_rejected(msg_type, msg):
    with pytest.raises(ValueError):
        if msg_type is ListKeysMsg:
            msg_type.validate(msg)
        else:
            msg_type.validate(msg, max_keys=2)


def test_connector_rpc_drives_an_import(pair):
    """The import is reached through the generic connector RPC (on_rpc)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    consumer, producer = pair
    keys = [f"k{i}".encode() for i in range(6)]
    _seed(producer, keys)
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = consumer
    scheduler._step_region = None

    def rpc(msg):
        reply = scheduler.on_rpc(json.dumps(msg).encode())
        return None if reply is None else json.loads(reply)

    started = rpc({"op": "import_from_peer", "host": "B", "port": 2})
    assert "import_id" in started, started
    status = _run(consumer, producer, started["import_id"])
    assert status["state"] == "done" and status["imported"] == len(keys), status
    assert rpc({"op": "import_status", "import_id": started["import_id"]}) == [status]
    assert all(_hits(consumer, keys))

    assert "error" in rpc({"op": "import_from_peer", "host": "B", "port": 0})
    # A JSON float must not reach the listing, where it would raise in the step.
    for bad in (2.5, 20000.0, True):
        msg = {"op": "import_from_peer", "host": "B", "port": 2, "max_chunks": bad}
        assert "error" in rpc(msg), bad
    # Not ours: a MultiConnector passes these to its next child.
    assert rpc({"op": "migrate"}) is None
    assert rpc([1, 2]) is None
    assert rpc({"op": ["import_status"]}) is None
    assert scheduler.on_rpc(b"\xff not json") is None


def test_connector_rpc_without_tiering_replies_an_error():
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.manager = SimpleNamespace(lock=threading.Lock())  # Not tiering.
    scheduler._step_region = None
    for op in ("import_from_peer", "import_status"):
        reply = json.loads(scheduler.on_rpc(json.dumps({"op": op}).encode()))
        assert "TieringOffloadingSpec" in reply["error"], reply


def test_rejected_handshake_reports_its_cause(pair, monkeypatch):
    """A config mismatch must show in the status, not only in the logs."""
    consumer, producer = pair
    data = producer.secondary_tiers[0]._data
    monkeypatch.setattr(data, "block_len", data.block_len * 4)  # Block 64 vs 16.
    status = _import(consumer, producer)
    assert status["state"] == "failed", status
    assert "block_len mismatch" in status["error"], status
