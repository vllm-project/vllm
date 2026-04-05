# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace


def _reload_core_client_module():
    module = importlib.import_module("vllm.v1.engine.core_client")
    return importlib.reload(module)


def _reload_engine_utils_module():
    module = importlib.import_module("vllm.v1.engine.utils")
    return importlib.reload(module)


def test_mp_client_releases_held_sockets_before_bind(monkeypatch):
    core_client_mod = _reload_core_client_module()

    class ShadowSocket:
        def poll(self, timeout: int) -> int:
            return 1

        def recv_multipart(self):
            return (b"\x00\x00", b"ready")

    class DummySocket:
        def send_multipart(self, _msg, *, copy: bool = False, track: bool = False):
            if track:
                return SimpleNamespace(done=True)

        def recv_multipart(self, *, copy: bool = False):
            return (b"", b"")

        def close(self, *, linger: int = 0):
            pass

        def bind(self, _address):
            pass

        def connect(self, _address):
            pass

        def setsockopt(self, *_args, **_kwargs):
            pass

    class HeldSocket:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class ReadyEvent:
        def __init__(self):
            self.wait_calls = 0

        def wait(self):
            self.wait_calls += 1

    held_input_socket = HeldSocket()
    held_output_socket = HeldSocket()
    held_socket_ready = ReadyEvent()

    def make_zmq_socket(_ctx, address, *_args, **_kwargs):
        if address == "inproc://input":
            assert held_input_socket.closed
            assert not held_output_socket.closed
        elif address == "inproc://output":
            assert held_output_socket.closed
        return DummySocket()

    monkeypatch.setattr(core_client_mod.zmq.Socket, "shadow", lambda *_: ShadowSocket())
    monkeypatch.setattr(core_client_mod, "make_zmq_socket", make_zmq_socket)

    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_index=0,
        data_parallel_size_local=1,
        data_parallel_rank_local=None,
        data_parallel_hybrid_lb=False,
        data_parallel_external_lb=False,
        local_engines_only=False,
        enable_elastic_ep=False,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    client = core_client_mod.MPClient(
        asyncio_mode=False,
        vllm_config=vllm_config,
        executor_class=object,
        log_stats=False,
        client_addresses={
            "input_address": "inproc://input",
            "output_address": "inproc://output",
            "held_input_socket": held_input_socket,
            "held_output_socket": held_output_socket,
            "held_socket_ready": held_socket_ready,
        },
    )
    try:
        assert held_socket_ready.wait_calls == 1
        assert held_input_socket.closed
        assert held_output_socket.closed
    finally:
        client.shutdown()


def test_get_engine_zmq_addresses_reserves_ipv6_sockets(monkeypatch):
    engine_utils_mod = _reload_engine_utils_module()

    created_sockets = []

    class ReservedSocket:
        next_port = 5500

        def __init__(self, family, socktype):
            self.family = family
            self.socktype = socktype
            self.bound_address = None
            self.port = ReservedSocket.next_port
            ReservedSocket.next_port += 1
            created_sockets.append(self)

        def bind(self, address):
            self.bound_address = address

        def getsockname(self):
            return ("::1", self.port, 0, 0)

        def close(self):
            pass

    monkeypatch.setattr(engine_utils_mod.stdlib_socket, "socket", ReservedSocket)

    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_rank_local=None,
        data_parallel_size=2,
        data_parallel_master_ip="::1",
        local_engines_only=False,
        enable_elastic_ep=False,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config)

    addresses = engine_utils_mod.get_engine_zmq_addresses(
        vllm_config, num_api_servers=2
    )

    assert addresses.inputs == ["tcp://[::1]:5500", "tcp://[::1]:5501"]
    assert addresses.outputs == ["tcp://[::1]:5502", "tcp://[::1]:5503"]
    assert all(
        sock.family == engine_utils_mod.stdlib_socket.AF_INET6
        for sock in created_sockets
    )
    assert all(sock.bound_address == ("::1", 0, 0, 0) for sock in created_sockets)
