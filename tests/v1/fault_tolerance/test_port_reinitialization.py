# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel


class _MemoryStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def set(self, key: str, value: bytes) -> None:
        self.values[key] = value

    def get(self, key: str) -> bytes:
        return self.values[key]


def _assert_port_is_held(listen_socket: socket.socket) -> None:
    host, port = listen_socket.getsockname()[:2]
    competing_socket = socket.socket(listen_socket.family, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            competing_socket.bind((host, port))
    finally:
        competing_socket.close()


def test_engine_dp_reinit_hands_prebound_socket_to_store(monkeypatch) -> None:
    old_store = _MemoryStore()
    parallel_config = SimpleNamespace(
        data_parallel_master_ip="127.0.0.1",
        data_parallel_rank=0,
        data_parallel_size=2,
    )
    engine = SimpleNamespace(
        engine_index=0,
        dp_group=object(),
        dp_store=old_store,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
    )
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine = engine
    sentinel._dp_reinit_epoch = 0

    def fake_init(*args, listen_socket=None, **kwargs):
        assert listen_socket is not None
        _assert_port_is_held(listen_socket)
        listen_socket.close()
        return object(), _MemoryStore()

    monkeypatch.setattr(
        "vllm.v1.fault_tolerance.engine_core_sentinel."
        "stateless_destroy_torch_distributed_process_group",
        Mock(),
    )
    monkeypatch.setattr(
        "vllm.v1.fault_tolerance.engine_core_sentinel."
        "stateless_init_torch_distributed_process_group",
        fake_init,
    )

    params = sentinel._reinit_dp_group()

    assert params["new_stateless_dp_group_coord_port"] > 0
    assert params["new_stateless_dp_group_epoch"] == 0
    assert (
        old_store.values["ft_engine_dp_port_0"]
        == str(params["new_stateless_dp_group_coord_port"]).encode()
    )


def test_worker_dp_reinit_hands_prebound_socket_to_store(monkeypatch) -> None:
    coord_store = _MemoryStore()
    dp_group = SimpleNamespace(cpu_group=object())
    worker = SimpleNamespace(
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
    )
    sentinel = WorkerSentinel.__new__(WorkerSentinel)
    sentinel.worker = worker
    sentinel.dp_rank = 0
    sentinel.dp_size = 2
    sentinel.data_parallel_master_ip = "127.0.0.1"
    sentinel._clean_worker_state = Mock()

    def fake_init(*args, listen_socket=None, **kwargs):
        assert listen_socket is not None
        _assert_port_is_held(listen_socket)
        listen_socket.close()
        return object()

    monkeypatch.setattr("torch.accelerator.synchronize", Mock())
    monkeypatch.setattr(
        "vllm.v1.worker.sentinel.gpu_worker_sentinel.get_ep_all2all_manager",
        lambda: SimpleNamespace(clean_buffers=Mock()),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.sentinel.gpu_worker_sentinel.get_dp_group",
        lambda: dp_group,
    )
    monkeypatch.setattr(
        "vllm.distributed.utils.get_cached_tcp_store_client",
        lambda *args: coord_store,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.sentinel.gpu_worker_sentinel."
        "stateless_destroy_torch_distributed_process_group",
        Mock(),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.sentinel.gpu_worker_sentinel."
        "stateless_init_torch_distributed_process_group",
        fake_init,
    )

    sentinel.retry(
        FaultToleranceRequest(
            instruction="retry",
            params={
                "new_stateless_dp_group_coord_port": 12345,
                "new_stateless_dp_group_epoch": 3,
            },
        )
    )

    worker_port = int(coord_store.values["ft_worker_dp_port_3_0"].decode())
    assert worker_port > 0
    assert dp_group.cpu_group is not None
