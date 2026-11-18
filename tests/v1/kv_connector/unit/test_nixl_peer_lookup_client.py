# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PeerLookupClient against a scripted fake producer and the real listener."""

import os
import socket
import threading
import time

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    LOOKUP_PIN_MSG,
    LookupPinReply,
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
    """A ROUTER on a free port. Tests call recv() / reply() from their thread."""

    def __init__(self) -> None:
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port(f"tcp://{HOST}")

    def recv(self, timeout_ms: int = 5000) -> tuple[bytes, object]:
        assert self.sock.poll(timeout_ms), "no request"
        identity, _, raw = self.sock.recv_multipart()
        msg, body = msgspec.msgpack.decode(raw)
        assert msg == LOOKUP_PIN_MSG
        return identity, msgspec.msgpack.decode(body, type=type(REQ))

    def reply(self, identity: bytes, reply: LookupPinReply | bytes) -> None:
        raw = reply if isinstance(reply, bytes) else msgspec.msgpack.encode(reply)
        self.sock.send_multipart((identity, b"", raw))

    def close(self) -> None:
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
        client.submit("a", HOST, get_open_port(), REQ)
        assert wait_result(client, "a") == LookupResult(None, "timeout")
        assert not client.has_in_flight()
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
            assert wait_result(bad, "bad") == LookupResult(None, "timeout")
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
    assert wait_result(client, "bad") == LookupResult(None, "send_failed")
    client.submit("ok", HOST, peer.port, REQ)
    identity, _ = peer.recv()
    peer.reply(identity, miss())
    assert wait_result(client, "ok").reply == miss()


def test_real_listener_disabled_engine_answers_miss(client):
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
    assert result.error is None
    assert result.reply is not None
    assert result.reply.miss_reason == PeerMissReason.IDENTITY_MISMATCH
    assert result.reply.remote_request_id == ""


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
