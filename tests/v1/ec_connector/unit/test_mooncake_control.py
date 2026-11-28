# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource bounds for the Mooncake control client."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import pytest
import zmq

from vllm.distributed.ec_transfer.ec_connector.mooncake import control


@pytest.fixture
def mooncake_control_server() -> Iterator[tuple[list[str], list[int]]]:
    ready: queue.Queue[list[str] | Exception] = queue.Queue(maxsize=1)
    stop = threading.Event()
    requests = [0]

    def serve() -> None:
        context = zmq.Context()
        sockets: list[zmq.Socket] = []
        ports: list[int] = []
        ready_sent = False
        try:
            for _ in range(3):
                socket = context.socket(zmq.REP)
                ports.append(socket.bind_to_random_port("tcp://127.0.0.1"))
                sockets.append(socket)
            poller = zmq.Poller()
            for socket in sockets:
                poller.register(socket, zmq.POLLIN)
            addresses = [f"tcp://127.0.0.1:{port}" for port in ports]
            ready.put(addresses)
            ready_sent = True
            while not stop.is_set():
                for socket, events in poller.poll(timeout=50):
                    if events & zmq.POLLIN:
                        socket.recv_json()
                        requests[0] += 1
                        socket.send_json(
                            {"ok": True, "result": {"ports": [19020, 19021]}}
                        )
        except Exception as exc:
            if not ready_sent:
                ready.put(exc)
        finally:
            for socket in sockets:
                socket.close(linger=0)
            context.term()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    result = ready.get(timeout=10)
    if isinstance(result, Exception):
        stop.set()
        server.join(timeout=5)
        raise result
    try:
        yield result, requests
    finally:
        stop.set()
        server.join(timeout=5)
        assert not server.is_alive()


def test_control_client_evicts_and_recreates_least_recently_used_socket(
    mooncake_control_server: tuple[list[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "_MAX_CONTROL_SOCKETS_PER_THREAD", 2, raising=False)
    addresses, _ = mooncake_control_server
    address_a, address_b, address_c = addresses
    client = control.ControlClient(timeout_ms=5000)
    try:
        client.request(address_a, {"op": "ping"})
        client.request(address_b, {"op": "ping"})
        socket_a = client._sockets()[address_a]
        socket_b = client._sockets()[address_b]

        client.request(address_a, {"op": "ping"})
        client.request(address_c, {"op": "ping"})

        assert len(client._sockets()) == 2
        assert client._sockets()[address_a] is socket_a
        assert socket_b.closed
        assert address_b not in client._sockets()

        client.request(address_b, {"op": "ping"})
        assert len(client._sockets()) == 2
        assert client._sockets()[address_b] is not socket_b
        assert not client._sockets()[address_b].closed
    finally:
        client.close()


def test_control_client_bounds_topology_cache(
    mooncake_control_server: tuple[list[str], list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "_MAX_CACHED_TOPOLOGIES", 2, raising=False)
    addresses, requests = mooncake_control_server
    address_a, address_b, address_c = addresses
    client = control.ControlClient(timeout_ms=5000)
    try:
        client.discover_shards(address_a)
        client.discover_shards(address_b)
        client.discover_shards(address_c)
        client.discover_shards(address_b)
        client.discover_shards(address_a)

        # C evicts A; the hit on B refreshes its position before A is queried.
        assert requests[0] == 4
    finally:
        client.close()


def test_control_client_returns_copies_of_cached_topologies(
    mooncake_control_server: tuple[list[str], list[int]],
) -> None:
    addresses, _ = mooncake_control_server
    address_a = addresses[0]
    expected = [
        f"{address_a.rsplit(':', 1)[0]}:19020",
        f"{address_a.rsplit(':', 1)[0]}:19021",
    ]
    client = control.ControlClient(timeout_ms=5000)
    try:
        discovered = client.discover_shards(address_a)
        assert discovered == expected
        assert discovered is not None
        discovered.append("tcp://corrupted:19020")
        assert client.discover_shards(address_a) == expected
    finally:
        client.close()
