# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
import sys
from types import ModuleType, SimpleNamespace

from vllm.distributed import parallel_state
from vllm.utils.network_utils import get_distributed_init_method


def test_init_distributed_environment_uses_reserved_listen_socket(
    monkeypatch,
):
    init_process_group_calls: list[dict] = []
    store_calls: list[dict] = []
    store = object()

    config_stub = ModuleType("vllm.config")
    config_stub.get_current_vllm_config_or_none = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.config", config_stub)
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setattr(
        parallel_state.torch.distributed, "is_initialized", lambda: False
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "is_backend_available",
        lambda backend: True,
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "init_process_group",
        lambda **kwargs: init_process_group_calls.append(kwargs),
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "get_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda ranks, local_rank, backend: SimpleNamespace(cpu_group=object()),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda group: 1)

    def fake_create_tcp_store(host: str, port: int, **kwargs):
        store_calls.append(
            {
                "host": host,
                "port": port,
                **kwargs,
            }
        )
        return store

    monkeypatch.setattr(parallel_state, "create_tcp_store", fake_create_tcp_store)

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    _, port = listen_socket.getsockname()
    distributed_init_method = get_distributed_init_method("127.0.0.1", port)

    parallel_state.init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=distributed_init_method,
        local_rank=0,
        backend="gloo",
        distributed_listen_socket=listen_socket,
    )

    assert store_calls == [
        {
            "host": "127.0.0.1",
            "port": port,
            "listen_socket": listen_socket,
            "world_size": 1,
            "is_master": True,
        }
    ]
    assert init_process_group_calls == [
        {
            "backend": "gloo",
            "store": store,
            "world_size": 1,
            "rank": 0,
            "timeout": None,
        }
    ]


def test_init_distributed_environment_closes_unused_reserved_socket(monkeypatch):
    config_stub = ModuleType("vllm.config")
    config_stub.get_current_vllm_config_or_none = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.config", config_stub)
    monkeypatch.setattr(
        parallel_state.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        parallel_state,
        "_WORLD",
        SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(parallel_state.torch.distributed, "get_world_size", lambda: 1)

    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen()
    _, port = listen_socket.getsockname()

    parallel_state.init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=get_distributed_init_method("127.0.0.1", port),
        local_rank=0,
        backend="gloo",
        distributed_listen_socket=listen_socket,
    )

    assert listen_socket.fileno() == -1


def test_init_distributed_environment_uses_tcp_store_for_nonzero_ranks(monkeypatch):
    init_process_group_calls: list[dict] = []
    store_calls: list[dict] = []
    store = object()

    config_stub = ModuleType("vllm.config")
    config_stub.get_current_vllm_config_or_none = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.config", config_stub)
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setattr(
        parallel_state.torch.distributed, "is_initialized", lambda: False
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "is_backend_available",
        lambda backend: True,
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "init_process_group",
        lambda **kwargs: init_process_group_calls.append(kwargs),
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "get_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        parallel_state,
        "init_world_group",
        lambda ranks, local_rank, backend: SimpleNamespace(cpu_group=object()),
    )
    monkeypatch.setattr(parallel_state, "_node_count", lambda group: 1)

    def fake_create_tcp_store(host: str, port: int, **kwargs):
        store_calls.append(
            {
                "host": host,
                "port": port,
                **kwargs,
            }
        )
        return store

    monkeypatch.setattr(parallel_state, "create_tcp_store", fake_create_tcp_store)

    distributed_init_method = get_distributed_init_method("127.0.0.1", 23456)

    parallel_state.init_distributed_environment(
        world_size=2,
        rank=1,
        distributed_init_method=distributed_init_method,
        local_rank=1,
        backend="gloo",
    )

    assert store_calls == [
        {
            "host": "127.0.0.1",
            "port": 23456,
            "world_size": 2,
            "is_master": False,
        }
    ]
    assert init_process_group_calls == [
        {
            "backend": "gloo",
            "store": store,
            "world_size": 2,
            "rank": 1,
            "timeout": None,
        }
    ]
