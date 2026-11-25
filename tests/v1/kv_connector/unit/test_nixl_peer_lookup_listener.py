# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL peer lookup over the side channel: listener, scheduler drain, wake-up."""

import threading
import time

import msgspec
import pytest
import zmq

from tests.v1.core.test_prefix_pin import BLOCK_SIZE
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    LOOKUP_PIN_MSG,
    LookupPinReply,
    LookupPinRequest,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    PeerKVIdentity,
    PeerMissReason,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.utils.network_utils import get_open_port, make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput

from .test_nixl_peer_lookup_client import OldHandshakePayload
from .test_nixl_peer_pins import (
    COMPAT,
    Clock,
    _init_hash,  # noqa: F401  (autouse fixture)
    cached_manager,
    lookup,
    make_wired_connector,
    make_wired_scheduler,
)

pytestmark = pytest.mark.cpu_test

HOST = "127.0.0.1"
ENGINE = "producer-engine"
# What the probe reads the engine id from.
AGENT = msgspec.msgpack.encode({"engine_id": ENGINE})


@pytest.fixture
def served():
    """A wired scheduler with a real listener; yields (sched, connector, manager,
    hashes, path)."""
    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager)
    port = get_open_port()
    sched.side_channel_host = HOST
    sched.side_channel_port = port
    sched._stop_event = threading.Event()
    sched._handshake_data = {}
    sched._nixl_handshake_listener_t = None
    connector = make_wired_connector(sched)
    connector.bind_kv_cache_manager(manager)
    sched.set_xfer_handshake_metadata({(0, 0): NixlHandshakePayload(COMPAT, AGENT)})
    try:
        yield sched, connector, manager, hashes, make_zmq_path("tcp", HOST, port)
    finally:
        sched.shutdown()


def send_lookup(sock, request) -> None:
    sock.send(msgspec.msgpack.encode((LOOKUP_PIN_MSG, msgspec.msgpack.encode(request))))


def recv_reply(sock, timeout_ms: int = 5000) -> LookupPinReply:
    assert sock.poll(timeout_ms), "no reply"
    return msgspec.msgpack.decode(sock.recv(), type=LookupPinReply)


def cycle(connector) -> NixlConnectorMetadata:
    """One schedule (build_connector_meta) + processed (take_events) tick."""
    meta = connector.build_connector_meta(SchedulerOutput.make_empty())
    assert isinstance(meta, NixlConnectorMetadata)
    list(connector.take_events())
    return meta


def wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def get_meta(sock):
    sock.send(msgspec.msgpack.encode((GET_META_MSG, 0, 0)))
    assert sock.poll(5000), "no GET_META reply"
    return sock.recv_multipart()


def test_lookup_round_trip(served):
    sched, connector, manager, hashes, path = served
    with zmq_ctx(zmq.REQ, path) as sock:
        send_lookup(sock, lookup(hashes, lookup_id="id-1"))
        wait_for(lambda: connector.has_pending_push_work())
        meta = cycle(connector)  # drains, fences this step, ships nothing yet
        assert meta.reqs_to_send == {}
        assert sock.poll(100) == 0
        meta = cycle(connector)  # the fence step is processed: ships and replies
        reply = recv_reply(sock)
    assert reply.hit_tokens == len(hashes) * BLOCK_SIZE
    assert reply.miss_reason == PeerMissReason.NONE
    assert reply.lookup_id == "id-1"
    assert len(reply.remote_block_ids) == 1
    assert len(reply.remote_block_ids[0]) == len(hashes)
    (pin_id,) = meta.reqs_to_send
    assert reply.remote_request_id == pin_id
    # The shipped pin keeps the engine stepping until its read is reported.
    assert connector.has_pending_push_work()


def test_get_meta_survives_bad_messages(served):
    _, _, _, _, path = served
    with zmq_ctx(zmq.REQ, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        frames = get_meta(sock)
        assert len(frames) == 2  # payload, then the listener's clock
        # The capability probe: this engine serves lookups.
        payload = msgspec.msgpack.decode(frames[0], type=NixlHandshakePayload)
        assert payload.peer_lookup is True

    # Each bad message on its own socket: a REQ cannot send twice unanswered.
    for bad in (
        b"\xc1not msgpack",  # undecodable
        msgspec.msgpack.encode(("no_such_message", 1, 2)),  # unknown type
        msgspec.msgpack.encode((GET_META_MSG, 7, 7)),  # unknown rank
        msgspec.msgpack.encode((LOOKUP_PIN_MSG, b"\xc1")),  # bad request body
        msgspec.msgpack.encode(5),  # not a sequence
        msgspec.msgpack.encode([]),  # empty
    ):
        with zmq_ctx(zmq.REQ, path) as bad_sock:
            bad_sock.send(bad)
        with zmq_ctx(zmq.REQ, path) as sock:
            frames = get_meta(sock)
            assert len(frames) == 2


def test_lookup_disabled_gets_immediate_miss():
    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager, peer_lookup=False)
    port = get_open_port()
    sched.side_channel_host = HOST
    sched.side_channel_port = port
    sched._stop_event = threading.Event()
    sched._handshake_data = {}
    sched._nixl_handshake_listener_t = None
    make_wired_connector(sched).bind_kv_cache_manager(manager)
    sched.set_xfer_handshake_metadata({(0, 0): NixlHandshakePayload(COMPAT, b"x")})
    try:
        path = make_zmq_path("tcp", HOST, port)
        with zmq_ctx(zmq.REQ, path) as sock:
            # A disabled engine does not advertise lookups to the probe...
            payload = msgspec.msgpack.decode(
                get_meta(sock)[0], type=NixlHandshakePayload
            )
            assert payload.peer_lookup is False
        # ...and refuses a lookup that is sent anyway, echoing its id.
        with zmq_ctx(zmq.REQ, path) as sock:
            send_lookup(sock, lookup(hashes, lookup_id="id-2"))
            reply = recv_reply(sock)
        assert reply.lookup_id == "id-2"
        assert reply.hit_tokens == 0
        assert reply.remote_block_ids == []
        assert reply.remote_request_id == ""
        assert reply.miss_reason == PeerMissReason.IDENTITY_MISMATCH
        assert not sched.has_pending_push_work()
    finally:
        sched.shutdown()


def test_wakeup_called_and_pending_work_tracks_inbox(served):
    sched, connector, _, hashes, path = served
    woken = threading.Event()
    connector.set_engine_wakeup(woken.set)
    assert not connector.has_pending_push_work()
    with zmq_ctx(zmq.REQ, path) as sock:
        send_lookup(sock, lookup(hashes))
        assert woken.wait(5)
        assert connector.has_pending_push_work()
        cycle(connector)
        cycle(connector)
        recv_reply(sock)
    # Served and flushed: only the shipped pin keeps the engine stepping.
    assert sched._lookup.inbox.empty() and sched._lookup.outbox.empty()
    assert connector.has_pending_push_work()


def test_idle_engine_answers_within_two_cycles(served):
    """Nothing is scheduled: only the lookup keeps the loop turning."""
    _, connector, _, hashes, path = served
    with zmq_ctx(zmq.REQ, path) as sock:
        send_lookup(sock, lookup(hashes))
        cycles = 0
        while sock.poll(0) == 0 and cycles < 10:
            wait_for(connector.has_pending_push_work)
            cycle(connector)
            cycles += 1
            sock.poll(50)
        assert cycles == 2
        assert recv_reply(sock).hit_tokens > 0


def test_miss_is_answered_in_the_first_cycle(served):
    _, connector, _, _, path = served
    with zmq_ctx(zmq.REQ, path) as sock:
        send_lookup(sock, lookup([b"\x00" * 32]))
        wait_for(connector.has_pending_push_work)
        cycle(connector)
        reply = recv_reply(sock)
    assert reply.hit_tokens == 0
    assert reply.miss_reason == PeerMissReason.NOT_CACHED
    assert not connector.has_pending_push_work()


def test_lookup_before_binding_is_pin_refused_not_stuck():
    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager)
    sched._lookup.inbox.put((b"id", lookup(hashes, lookup_id="id-3")))
    assert sched.has_pending_push_work()
    make_wired_connector(sched).build_connector_meta(SchedulerOutput.make_empty())
    identity, reply = sched._lookup.outbox.get_nowait()
    assert identity == b"id"
    # Transient (lookups are enabled, just not bound yet), unlike a disabled
    # engine, which refuses for good with IDENTITY_MISMATCH.
    assert reply.miss_reason == PeerMissReason.PIN_REFUSED
    assert reply.lookup_id == "id-3"
    sched._lookup.close()


def test_pending_pin_times_out_with_pin_refused():
    manager, hashes = cached_manager()
    sched = make_wired_scheduler(manager, serve_timeout=5.0)
    clock = Clock()
    connector = make_wired_connector(sched)
    connector.bind_kv_cache_manager(manager)
    server = sched._peer_pins
    assert server is not None
    server._clock = clock
    server._max_pending_seconds = 5.0
    server.set_compatibility_hash(COMPAT)

    sched._lookup.inbox.put((b"id", lookup(hashes, lookup_id="id-4")))
    # Scheduled but never processed (a stalled step): the fence cannot pass.
    connector.build_connector_meta(SchedulerOutput.make_empty())
    assert manager.num_pinned_blocks == len(hashes)
    assert sched._lookup.outbox.empty()

    clock.now += 4.0
    connector.build_connector_meta(SchedulerOutput.make_empty())
    assert manager.num_pinned_blocks == len(hashes)

    clock.now += 2.0
    meta = connector.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.reqs_to_send == {}
    assert manager.num_pinned_blocks == 0
    identity, reply = sched._lookup.outbox.get_nowait()
    assert identity == b"id"
    assert reply.hit_tokens == 0
    assert reply.miss_reason == PeerMissReason.PIN_REFUSED
    assert reply.lookup_id == "id-4"
    assert not connector.has_pending_push_work() or not sched._lookup.outbox.empty()
    sched._lookup.close()


def test_serve_timeout_config_default_is_five_seconds():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.peer_pins import (
        PeerPinServer,
    )

    manager, _ = cached_manager()
    server = PeerPinServer(manager, BLOCK_SIZE, "sha256", 30.0, 1.0, 0.0)
    assert server._max_pending_seconds is None  # no timeout unless configured


def test_multi_connector_forwards_set_engine_wakeup():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    class Child:
        def __init__(self):
            self.cb = None

        def set_engine_wakeup(self, callback):
            self.cb = callback

    multi = object.__new__(MultiConnector)
    children = [Child(), Child()]
    multi._connectors = children

    def callback():
        return None

    multi.set_engine_wakeup(callback)
    assert [c.cb for c in children] == [callback, callback]


def test_base_connector_set_engine_wakeup_is_a_noop():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

    assert KVConnectorBase_V1.set_engine_wakeup(object(), lambda: None) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "msg",
    [
        LookupPinRequest(
            block_hashes=[b"\x01" * 32, b"\x02" * 32],
            identity=PeerKVIdentity(
                "abc123", 16, hash_algo="sha256", cache_salt="salt"
            ),
            lease_duration=30.0,
            lookup_id="id-1",
        ),
        # From a consumer or producer that sends no id.
        LookupPinRequest(
            block_hashes=[b"\x03" * 32],
            identity=PeerKVIdentity("abc123", 16, hash_algo="sha256"),
        ),
        NixlHandshakePayload("abc123", b"agent", peer_lookup=True),
        LookupPinReply(
            hit_tokens=32,
            remote_block_ids=[[3, 4], [5, 6]],
            remote_request_id="pin-1",
            miss_reason=PeerMissReason.NONE,
            remote_blocks_expiry_time=1234.5,
            remote_lease_extension=20,
            lookup_id="id-1",
        ),
        LookupPinReply(
            hit_tokens=0,
            remote_block_ids=[],
            remote_request_id="pin-2",
            miss_reason=PeerMissReason.IDENTITY_MISMATCH,
        ),
    ],
)
def test_lookup_pin_wire_roundtrip(msg):
    """Lookup-and-pin messages survive the side channel's msgpack encoding."""
    decoded = msgspec.msgpack.Decoder(type(msg)).decode(msgspec.msgpack.encode(msg))
    assert decoded == msg


def test_handshake_payload_peer_lookup_is_compatible_both_ways():
    """`peer_lookup` is optional on the wire, so mixed versions interoperate.

    An engine that predates it must keep decoding a new engine's payload (its
    handshake would otherwise break), and a new consumer must read an old
    engine's payload as "serves no lookups".
    """
    new = NixlHandshakePayload("abc123", b"agent", peer_lookup=True)
    seen_by_old = msgspec.msgpack.decode(
        msgspec.msgpack.encode(new), type=OldHandshakePayload
    )
    assert seen_by_old == OldHandshakePayload("abc123", b"agent")

    old = OldHandshakePayload("abc123", b"agent")
    seen_by_new = msgspec.msgpack.decode(
        msgspec.msgpack.encode(old), type=NixlHandshakePayload
    )
    assert seen_by_new == NixlHandshakePayload("abc123", b"agent", peer_lookup=False)
