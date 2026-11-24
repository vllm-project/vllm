# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PeerLookupClient against a scripted fake producer and the real listener."""

import os
import socket
import threading
import time
from dataclasses import dataclass

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    LOOKUP_PIN_MSG,
    LookupPinReply,
    LookupPinRequest,
    NixlHandshakePayload,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_lookup_client import (
    LookupResult,
    PeerLookupClient,
    _Job,
)
from vllm.utils.network_utils import get_open_port

from .test_nixl_peer_pins import (
    COMPAT,
    _init_hash,  # noqa: F401  (autouse fixture)
    cached_manager,
    lookup,
    make_wired_scheduler,
)

pytestmark = pytest.mark.cpu_test

HOST = "127.0.0.1"
REQ = lookup([b"h1", b"h2"])


def hit(pin_id: str = "pin-1", tokens: int = 32) -> LookupPinReply:
    return LookupPinReply(
        hit_tokens=tokens,
        remote_block_ids=[[1, 2]],
        remote_request_id=pin_id,
        miss_reason=PeerMissReason.NONE,
    )


def miss(reason: PeerMissReason = PeerMissReason.NOT_CACHED) -> LookupPinReply:
    return LookupPinReply(0, [], "", reason)


class FakeProducer:
    """A ROUTER on a free port. Tests call recv() / reply() from their thread.

    recv() answers the client's capability probes itself (counted in `probes`),
    advertising `peer_lookup` as `advertise` says, and returns the first lookup.
    """

    def __init__(self, advertise: bool = True) -> None:
        self.advertise = advertise
        # The engine id its handshake payload reports.
        self.engine_id = "e1"
        # Sent instead of the payload when set.
        self.garbage: bytes | None = None
        self.probes = 0
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port(f"tcp://{HOST}")

    def recv(self, timeout_ms: int = 5000) -> tuple[bytes, LookupPinRequest]:
        while True:
            assert self.sock.poll(timeout_ms), "no request"
            identity, _, raw = self.sock.recv_multipart()
            decoded = msgspec.msgpack.decode(raw)
            if decoded[0] == GET_META_MSG:
                self.probes += 1
                payload = NixlHandshakePayload(
                    COMPAT,
                    msgspec.msgpack.encode({"engine_id": self.engine_id}),
                    peer_lookup=self.advertise,
                )
                self.sock.send_multipart(
                    (
                        identity,
                        b"",
                        self.garbage or msgspec.msgpack.encode(payload),
                        msgspec.msgpack.encode(time.perf_counter()),
                    )
                )
                continue
            assert decoded[0] == LOOKUP_PIN_MSG
            return identity, msgspec.msgpack.decode(decoded[1], type=LookupPinRequest)

    def reply(self, identity: bytes, reply: LookupPinReply | bytes) -> None:
        raw = reply if isinstance(reply, bytes) else msgspec.msgpack.encode(reply)
        self.sock.send_multipart((identity, b"", raw))

    def close(self) -> None:
        self.ctx.destroy(linger=0)


@dataclass
class OldHandshakePayload:
    """NixlHandshakePayload as engines wrote it before `peer_lookup` existed."""

    compatibility_hash: str
    agent_metadata_bytes: bytes


class OldListener:
    """The side-channel listener of an engine that predates lookups.

    It decodes every frame as `(msg, pp_rank, tp_rank)` outside any try, so a
    LOOKUP_PIN 2-tuple raises and ends its thread for good, ordinary P/D
    handshakes included. It answers GET_META with a payload that has no
    `peer_lookup` field.
    """

    def __init__(self) -> None:
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port(f"tcp://{HOST}")
        self.frames: list[tuple] = []
        self.get_metas = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stop.is_set():
            if not self.sock.poll(50):
                continue
            identity, _, raw = self.sock.recv_multipart()
            decoded = msgspec.msgpack.decode(raw)
            self.frames.append(tuple(decoded))
            msg, target_pp_rank, target_tp_rank = decoded  # Raises on a 2-tuple.
            assert msg == GET_META_MSG
            self.get_metas += 1
            payload = OldHandshakePayload(COMPAT, b"x")
            self.sock.send_multipart(
                (
                    identity,
                    b"",
                    msgspec.msgpack.encode(payload),
                    msgspec.msgpack.encode(time.perf_counter()),
                )
            )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(5)
        self.ctx.destroy(linger=0)


@pytest.fixture
def peer():
    p = FakeProducer()
    yield p
    p.close()


@pytest.fixture
def client():
    c = PeerLookupClient(timeout_s=5.0)
    yield c
    c.close()


def wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def wait_result(client, key: str, timeout: float = 5.0) -> LookupResult:
    deadline = time.monotonic() + timeout
    while True:
        result = client.poll(key)
        if result is not None:
            return result
        assert time.monotonic() < deadline, "no result"
        time.sleep(0.005)


def test_hit(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    assert client.has_in_flight()
    identity, request = peer.recv()
    assert request == REQ
    peer.reply(identity, hit())
    result = wait_result(client, "a")
    assert result == LookupResult(hit())
    assert not client.has_in_flight()


def test_miss(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss(PeerMissReason.EVICTED))
    result = wait_result(client, "a")
    assert result.error is None
    assert result.reply is not None
    assert result.reply.miss_reason == PeerMissReason.EVICTED


def test_out_of_order_replies_reach_their_own_lookup(client, peer):
    client.submit("first", HOST, peer.port, REQ)
    id_first, _ = peer.recv()
    client.submit("second", HOST, peer.port, REQ)
    id_second, _ = peer.recv()
    assert id_first != id_second
    # The later lookup is answered first, with a different reply.
    peer.reply(id_second, miss())
    second = wait_result(client, "second")
    assert second.reply is not None and second.reply.hit_tokens == 0
    assert client.poll("first") is None
    peer.reply(id_first, hit("pin-first"))
    first = wait_result(client, "first")
    assert first.reply is not None
    assert first.reply.remote_request_id == "pin-first"


def test_timeout_when_peer_never_replies(peer):
    client = PeerLookupClient(timeout_s=0.2)
    try:
        client.submit("a", HOST, peer.port, REQ)
        peer.recv()
        assert wait_result(client, "a") == LookupResult(None, "timeout")
    finally:
        client.close()


def test_timeout_when_nothing_listens():
    client = PeerLookupClient(timeout_s=0.2)
    try:
        port = get_open_port()
        client.submit("a", HOST, port, REQ)
        assert wait_result(client, "a") == LookupResult(None, "probe_timeout")
        assert not client.has_in_flight()
        # The failed probe is cached briefly: the next lookup does not wait out
        # another timeout.
        client.submit("b", HOST, port, REQ)
        assert wait_result(client, "b", timeout=0.15) == LookupResult(
            None, "probe_timeout"
        )
    finally:
        client.close()


def test_garbage_reply_is_bad_reply(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, b"\xc1not msgpack")
    assert wait_result(client, "a") == LookupResult(None, "bad_reply")


def test_wrong_shape_reply_is_bad_reply(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, msgspec.msgpack.encode([1, 2]))
    assert wait_result(client, "a") == LookupResult(None, "bad_reply")


def test_submit_is_idempotent_and_does_not_block(client, peer):
    start = time.perf_counter()
    for _ in range(50):
        client.submit("a", HOST, peer.port, REQ)
    assert (time.perf_counter() - start) / 50 < 0.01
    # An unreachable peer does not slow submit either.
    start = time.perf_counter()
    client.submit("b", HOST, get_open_port(), REQ)
    assert time.perf_counter() - start < 0.01

    identity, _ = peer.recv()
    assert not peer.sock.poll(200)  # Exactly one request went out for "a".
    peer.reply(identity, hit())
    assert wait_result(client, "a").reply == hit()
    # Untaken results also make submit a no-op.
    client.submit("c", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss())
    wait_for(lambda: not client.has_in_flight())
    client.submit("c", HOST, peer.port, REQ)
    assert not peer.sock.poll(200)
    assert client.poll("c") is not None


def test_poll_forgets_key(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss())
    wait_result(client, "a")
    assert client.poll("a") is None
    assert client.poll("never-submitted") is None
    # The key can be used again.
    client.submit("a", HOST, peer.port, REQ)
    peer.recv()
    assert client.has_in_flight()


def test_cancel_in_flight_closes_socket_and_counts_lost(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    client.cancel("a")
    wait_for(lambda: client.num_lost_after_cancel == 1)
    assert not client.has_in_flight()
    peer.reply(identity, hit())  # Too late: the socket is gone.
    assert client.poll("a") is None
    assert client.take_orphaned_pins() == []
    assert client.num_lost_after_cancel == 1


def test_cancel_before_thread_starts_the_lookup(peer):
    client = PeerLookupClient(timeout_s=5.0)
    try:
        # Hold the lock so the thread cannot start the job before the cancel.
        with client._lock:
            job = _Job(0, "a", HOST, peer.port, b"")
            client._live["a"] = job
            client._cancelled.add(0)
            client._jobs.put(job)
        client._wake()
        wait_for(lambda: not client._cancelled)
        assert not peer.sock.poll(200)  # Never sent.
        assert client.num_lost_after_cancel == 0
    finally:
        client.close()


def test_cancel_after_untaken_hit_records_orphan(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, hit("pin-a"))
    wait_for(lambda: not client.has_in_flight())
    client.cancel("a")
    assert client.poll("a") is None
    assert client.take_orphaned_pins() == [(HOST, peer.port, hit("pin-a"))]
    assert client.take_orphaned_pins() == []


def test_cancel_after_untaken_miss_records_nothing(client, peer):
    client.submit("a", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss())
    wait_for(lambda: not client.has_in_flight())
    client.cancel("a")
    assert client.poll("a") is None
    assert client.take_orphaned_pins() == []


def test_reply_racing_a_cancel_is_orphaned(client):
    # The thread finishes a lookup whose cancel it has not yet acted on.
    job = _Job(7, "a", HOST, 1234, b"")
    with client._lock:
        client._live["a"] = job
        client._live.pop("a")
        client._cancelled.add(7)
    client._complete(job, LookupResult(hit("pin-race")))
    assert client.poll("a") is None
    assert client.take_orphaned_pins() == [(HOST, 1234, hit("pin-race"))]
    assert not client._cancelled
    # A miss, or a hit with no pin id, leaves nothing behind.
    with client._lock:
        client._cancelled.add(8)
    client._complete(_Job(8, "b", HOST, 1, b""), LookupResult(miss()))
    assert client.take_orphaned_pins() == []


def test_close_resolves_in_flight_as_closed(peer):
    client = PeerLookupClient(timeout_s=30.0)
    client.submit("a", HOST, peer.port, REQ)
    peer.recv()
    thread = client._thread
    client.close()
    assert not thread.is_alive()
    assert client.poll("a") == LookupResult(None, "closed")
    assert not client.has_in_flight()
    client.submit("b", HOST, peer.port, REQ)  # No-op, returns at once.
    assert client.poll("b") is None
    client.close()  # Idempotent.
    for fd in (client._wake_r, client._wake_w):
        with pytest.raises(OSError):
            os.fstat(fd)


def test_thread_survives_a_peer_that_resets(client, peer):
    srv = socket.socket()
    srv.bind((HOST, 0))
    srv.listen()
    port = srv.getsockname()[1]
    stop = threading.Event()

    def reset_forever() -> None:
        srv.settimeout(0.05)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.close()

    t = threading.Thread(target=reset_forever, daemon=True)
    t.start()
    try:
        bad = PeerLookupClient(timeout_s=0.3)
        try:
            bad.submit("bad", HOST, port, REQ)
            assert wait_result(bad, "bad") == LookupResult(None, "probe_timeout")
            assert bad._thread.is_alive()
            bad.submit("ok", HOST, peer.port, REQ)
            identity, _ = peer.recv()
            peer.reply(identity, hit())
            assert wait_result(bad, "ok").reply == hit()
        finally:
            bad.close()
    finally:
        stop.set()
        t.join()
        srv.close()


def test_bad_peer_address_fails_that_lookup_only(client, peer):
    client.submit("bad", "not a host!", 1, REQ)
    assert wait_result(client, "bad") == LookupResult(None, "probe_send_failed")
    client.submit("ok", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss())
    assert wait_result(client, "ok").reply == miss()


def test_real_listener_disabled_engine_is_not_sent_a_lookup(client):
    manager, _ = cached_manager()
    sched = make_wired_scheduler(manager, peer_lookup=False)
    port = get_open_port()
    sched.side_channel_host = HOST
    sched.side_channel_port = port
    sched._stop_event = threading.Event()
    sched._handshake_data = {}
    sched._nixl_handshake_listener_t = None
    sched.set_xfer_handshake_metadata({(0, 0): NixlHandshakePayload(COMPAT, b"x")})
    try:
        client.submit("a", HOST, port, REQ)
        result = wait_result(client, "a")
    finally:
        sched.shutdown()
    assert result == LookupResult(None, "peer_unsupported")


def test_probe_comes_first_is_shared_cached_and_repeated_on_expiry(peer):
    now = [0.0]
    client = PeerLookupClient(timeout_s=5.0, clock=lambda: now[0], cap_ttl_s=60.0)
    try:

        def one_lookup(key: str, engine_id: str) -> None:
            client.submit(key, HOST, peer.port, REQ, engine_id=engine_id)
            identity, _ = peer.recv()
            peer.reply(identity, miss())
            wait_result(client, key)

        # Two lookups at once wait for one probe.
        client.submit("a", HOST, peer.port, REQ, engine_id="e1")
        client.submit("b", HOST, peer.port, REQ, engine_id="e1")
        id_a, _ = peer.recv()
        id_b, _ = peer.recv()
        assert peer.probes == 1
        peer.reply(id_a, miss())
        peer.reply(id_b, miss())
        wait_result(client, "a")
        wait_result(client, "b")
        one_lookup("c", "e1")  # The verdict is cached.
        assert peer.probes == 1
        now[0] = 61.0  # The yes expired.
        one_lookup("d", "e1")
        assert peer.probes == 2
        peer.engine_id = "e2"
        one_lookup("e", "e2")  # Another engine at the same address.
        assert peer.probes == 3
        # The engine behind the address is not the one the router named (a
        # restart or downgrade): its yes is not trusted and no lookup is sent.
        peer.engine_id = "someone-else"
        now[0] = 200.0
        client.submit("f", HOST, peer.port, REQ, engine_id="e2")
        with pytest.raises(AssertionError, match="no request"):
            peer.recv(timeout_ms=300)  # Answers the probe, sees no lookup.
        assert wait_result(client, "f") == LookupResult(None, "peer_engine_mismatch")
    finally:
        client.close()


def test_old_listener_gets_no_lookup_and_keeps_serving():
    old = OldListener()
    now = [0.0]
    client = PeerLookupClient(timeout_s=5.0, clock=lambda: now[0], negative_ttl_s=30.0)
    try:
        client.submit("a", HOST, old.port, REQ)
        assert wait_result(client, "a") == LookupResult(None, "peer_unsupported")
        # Only the probe was sent, so the old listener is alive and still
        # answers an ordinary handshake request.
        assert old.frames == [(GET_META_MSG, 0, 0)]
        assert old.thread.is_alive()
        # A no is cached: no second probe, still no lookup.
        client.submit("b", HOST, old.port, REQ)
        assert wait_result(client, "b") == LookupResult(None, "peer_unsupported")
        assert old.get_metas == 1
        # After the negative TTL the peer is probed again (it may have upgraded).
        now[0] = 31.0
        client.submit("c", HOST, old.port, REQ)
        assert wait_result(client, "c") == LookupResult(None, "peer_unsupported")
        assert old.get_metas == 2
        assert all(frame[0] == GET_META_MSG for frame in old.frames)
        assert old.thread.is_alive()
    finally:
        client.close()
        old.close()


def test_undecodable_payload_is_peer_unsupported(client, peer, caplog):
    peer.garbage = b"\xc1not a payload"
    client.submit("a", HOST, peer.port, REQ)
    with pytest.raises(AssertionError, match="no request"):
        peer.recv(timeout_ms=300)
    assert wait_result(client, "a") == LookupResult(None, "peer_unsupported")
    assert client._caps[(HOST, peer.port)].error == "peer_unsupported"


def test_lookup_id_is_sent_echoed_and_checked(client, peer):
    request = LookupPinRequest(REQ.block_hashes, REQ.identity, lookup_id="want")
    client.submit("ok", HOST, peer.port, request)
    identity, seen = peer.recv()
    assert seen.lookup_id == "want"
    echoed = hit("pin-ok")
    echoed.lookup_id = "want"
    peer.reply(identity, echoed)
    assert wait_result(client, "ok").reply == echoed

    # A reply for another lookup, or one that echoes nothing, fails the lookup
    # and is not handed out as a hit.
    for key, wrong in (("other", "someone-else"), ("none", None)):
        client.submit(key, HOST, peer.port, request)
        identity, _ = peer.recv()
        bad = hit(f"pin-{key}")
        bad.lookup_id = wrong
        peer.reply(identity, bad)
        assert wait_result(client, key) == LookupResult(None, "lookup_id_mismatch")
        assert client.take_orphaned_pins() == []


def test_scheduler_lazily_creates_and_closes_client():
    manager, _ = cached_manager()
    sched = make_wired_scheduler(manager)
    sched._stop_event = threading.Event()
    sched._nixl_handshake_listener_t = None
    sched._peer_lookup_timeout = 1.5
    assert sched._lookup_client is None
    lc = sched._get_lookup_client()
    assert lc is sched._get_lookup_client()
    assert lc._timeout_s == 1.5
    sched.shutdown()
    assert not lc._thread.is_alive()
    assert sched._lookup_client is None
