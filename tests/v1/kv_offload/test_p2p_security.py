# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline regression for disclosure of the CPU KV region to untrusted peers."""

import socket

import msgspec
import numpy as np
import pytest
import zmq
from zmq.auth.base import Authenticator

from tests.v1.kv_offload.conftest import make_offloading_config
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.p2p import manager as p2p
from vllm.v1.kv_offload.tiering.p2p.control import zmq as zmq_control
from vllm.v1.kv_offload.tiering.p2p.control.base import (
    ControlConnection,
    ControlTransport,
)
from vllm.v1.kv_offload.tiering.p2p.data import nixl as nixl_data
from vllm.v1.kv_offload.tiering.p2p.peer_policy import PeerAllowlist
from vllm.v1.kv_offload.tiering.p2p.session.protocol import TYPE_KEY, ConnectMsg

# These tests create no accelerator or distributed state to clean up.
pytestmark = pytest.mark.skip_global_cleanup

NUM_BLOCKS, BLOCK_LEN = 8, 4096
AGENT_METADATA = b"synthetic NIXL worker address and region key"


class RecordingNixlAgent:
    def __init__(self, name, cfg):
        self.registered = []

    def create_backend(self, backend, params):
        pass

    def register_memory(self, descs, mem_type):
        self.registered.extend(descs)
        return descs

    def get_xfer_descs(self, descs, mem_type):
        return descs

    def prep_xfer_dlist(self, agent, descs):
        return descs

    def get_agent_metadata(self):
        return AGENT_METADATA

    def release_dlist_handle(self, handle):
        pass

    def deregister_memory(self, handle):
        pass


class RecordingConnection(ControlConnection):
    def __init__(self, peer_id):
        super().__init__(peer_id)
        self.sent = []
        self._alive = True

    @property
    def alive(self):
        return self._alive

    def send(self, msg):
        assert self.alive
        self.sent.append(msg)

    def recv(self):
        return []

    def mark_dead(self):
        self._alive = False

    def close(self):
        self._alive = False


class RecordingControl(ControlTransport):
    def __init__(self, local_id, host, port, **kwargs):
        self.connections = {}
        self.inbound = []

    def connect(self, peer_id):
        conn = RecordingConnection(peer_id)
        self.connections[peer_id] = conn
        return conn

    def poll(self):
        inbound, self.inbound = self.inbound, []
        self.connections.update((conn.peer_id, conn) for conn in inbound)
        return inbound

    def close(self):
        for conn in self.connections.values():
            conn.close()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("security regression must not use network sockets or DNS")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(zmq, "Context", no_network)


@pytest.fixture
def manager_factory(monkeypatch):
    monkeypatch.setattr(nixl_data, "_NixlAgent", RecordingNixlAgent)
    monkeypatch.setattr(nixl_data, "_NixlAgentConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(p2p, "ZmqTransport", RecordingControl)
    init_none_hash(sha256)
    managers = []

    def build(**tier_config):
        spec = CPUOffloadingSpec(
            make_offloading_config(worker_kv_bytes_per_block=BLOCK_LEN)
        )
        view = memoryview(bytearray(NUM_BLOCKS * BLOCK_LEN)).cast(
            "B", (NUM_BLOCKS, BLOCK_LEN)
        )
        manager = p2p.P2PSecondaryTierManager(spec, view, **tier_config)
        managers.append(manager)
        return manager

    yield build
    for manager in managers:
        manager.shutdown()


@pytest.mark.parametrize(
    "direction,role",
    [
        ("outbound", "remote_kv_source"),
        ("outbound", "remote_prefiller"),
        ("inbound", None),
    ],
)
@pytest.mark.parametrize(
    "tier_config,host,allowed",
    [
        ({}, "203.0.113.66", False),
        ({}, "127.0.0.2", True),
        ({}, "::1", True),
        ({"allowed_peers": []}, "127.0.0.1", False),
        ({"allowed_peers": ["10.0.0.0/24"]}, "203.0.113.77", False),
        ({"allowed_peers": ["10.0.0.0/24"]}, "10.0.0.2", True),
        ({"allowed_peers": ["10.0.0.2"]}, "10.0.0.3", False),
        ({"allowed_peers": ["2001:db8::/64"]}, "2001:db8::2", True),
    ],
)
def test_only_authorized_peers_receive_region_metadata(
    manager_factory, tier_config, host, allowed, direction, role
):
    """Both admission paths must authorize before the real session sends ConnectMsg."""
    manager = manager_factory(**tier_config)
    peer_id = f"{host}:5710"
    control = manager._control
    if direction == "outbound":
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            kv_transfer_params={
                role: {
                    "kv_request_id": "request-1",
                    "remote_host": host,
                    "remote_port": 5710,
                }
            },
        )
        sampling_params = request.to_sampling_params(16, {})
        assert sampling_params.extra_args is not None
        params = sampling_params.extra_args["kv_transfer_params"]
        ctx = ReqContext(req_id="request-1", kv_transfer_params=params)
        manager.on_new_request(ctx)
    else:
        control.inbound.append(RecordingConnection(peer_id))

    assert not list(manager.get_finished_jobs())
    conn = control.connections.get(peer_id)
    if not allowed:
        assert conn is None or (not conn.alive and not conn.sent)
        assert peer_id not in manager._sessions
        if direction == "outbound":
            assert manager.lookup(OffloadKey(b"block"), ctx) == LookupResult.MISS
            manager.submit_load(
                TransferJob(1, [OffloadKey(b"block")], np.array([0]), False, ctx)
            )
            assert list(manager.get_finished_jobs()) == [JobResult(1, False)]
    else:
        assert conn is not None and conn.alive
        msg = conn.sent[0]
        assert msg[TYPE_KEY] == ConnectMsg.TYPE
        assert msg[ConnectMsg.AGENT_METADATA] == AGENT_METADATA
        assert msg[ConnectMsg.BASE_ADDR] == manager._data.base_addr
        assert msg[ConnectMsg.NUM_BLOCKS] == NUM_BLOCKS
        assert msg[ConnectMsg.BLOCK_LEN] == BLOCK_LEN


@pytest.mark.parametrize("peers", ["10.0.0.2", [1], [""], ["10.0.0.0/99"]])
def test_invalid_allowlist_fails_startup(manager_factory, peers):
    with pytest.raises(ValueError, match="allowed_peers"):
        manager_factory(allowed_peers=peers)


def test_unresolvable_peer_fails_closed(manager_factory, monkeypatch):
    def resolve(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="allowed_peers"):
        manager_factory(allowed_peers=["missing.example"])
    manager = manager_factory()
    manager.on_new_request(
        ReqContext(
            req_id="unresolvable",
            kv_transfer_params={
                "remote_kv_source": {
                    "kv_request_id": "unresolvable",
                    "remote_host": "missing.example",
                    "remote_port": 5710,
                }
            },
        )
    )
    assert not manager._control.connections


class RecordingSocket:
    def __init__(self):
        self.options = {}
        self.connected = []
        self.inbox = []
        self.sent = []

    def setsockopt(self, key, value):
        self.options[key] = value

    def bind(self, address):
        pass

    def connect(self, address):
        self.connected.append(address)

    def monitor(self, *args):
        pass

    def disable_monitor(self):
        pass

    def recv_multipart(self, *args):
        if not self.inbox:
            raise zmq.Again()
        return self.inbox.pop(0)

    def send_multipart(self, frames):
        self.sent.append(frames)

    def close(self):
        pass


class RecordingContext:
    def __init__(self):
        self.sockets = []

    def socket(self, *args, **kwargs):
        sock = RecordingSocket()
        self.sockets.append(sock)
        return sock

    def destroy(self, **kwargs):
        pass


@pytest.mark.parametrize(
    "source,status", [("203.0.113.77", b"400"), ("10.0.0.2", b"200")]
)
def test_inbound_authorization_uses_source_ip(source, status):
    """A claimed allowlisted identity must not bypass source-IP authorization."""
    context = RecordingContext()
    auth = zmq_control._PeerAuthenticator(context, PeerAllowlist(["10.0.0.0/24"]))
    # Drive the real handler synchronously with an in-memory ZAP socket.
    Authenticator.start(auth)
    try:
        request = [
            b"1.0",
            b"1",
            b"vllm-p2p",
            source.encode(),
            b"10.0.0.2:5710",
            b"NULL",
        ]
        with pytest.raises(StopIteration):
            auth.handle_zap_message(request).send(None)
        assert context.sockets[0].sent[0][2] == status
    finally:
        auth.stop()


def test_transport_pins_dns_and_rejects_untrusted_dialback(monkeypatch):
    """Neither asynchronous DNS nor a claimed identity can redirect a connection."""
    context = RecordingContext()
    monkeypatch.setattr(zmq, "Context", lambda: context)

    class RecordingAuthenticator:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(zmq_control, "_PeerAuthenticator", RecordingAuthenticator)
    addresses = ["10.0.0.2"]

    def resolve(host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, 0)) for addr in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    policy = PeerAllowlist(["trusted.example"])
    transport = zmq_control.ZmqTransport(
        "10.0.0.1:5710", "10.0.0.1", 5710, peer_allowlist=policy
    )
    try:
        transport.connect("trusted.example:5710")
        router, dealer, _ = context.sockets
        assert router.options[zmq.ZAP_DOMAIN] == b"vllm-p2p"
        assert dealer.connected == ["tcp://10.0.0.2:5710"]

        addresses[:] = ["10.0.0.2", "203.0.113.66"]
        with pytest.raises(ValueError, match="allowed_peers"):
            transport.connect("trusted.example:5711")
        addresses[:] = ["203.0.113.66"]
        with pytest.raises(ValueError, match="allowed_peers"):
            transport.connect("trusted.example:5711")

        for identity in (b"203.0.113.77:7777", b"127.0.0.1:0", b"malformed", b"\xff"):
            router.inbox.append([identity, msgspec.msgpack.encode({})])
        assert not transport.poll()
        assert len(context.sockets) == 3
    finally:
        transport.close()
