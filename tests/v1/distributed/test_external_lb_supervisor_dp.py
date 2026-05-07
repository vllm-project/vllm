# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from types import SimpleNamespace

import pytest

import vllm.entrypoints.openai.dp_supervisor as dp_supervisor
from vllm.entrypoints.openai.dp_supervisor import (
    DPSupervisor,
    build_multi_port_external_lb_child_args,
    infer_multi_port_external_lb_start_rank,
)


def _make_args(**overrides) -> argparse.Namespace:
    base = {
        "host": None,
        "port": 8000,
        "data_parallel_multi_port_external_lb": True,
        "data_parallel_supervisor_port": 9256,
        "data_parallel_probe_interval_s": 5.0,
        "data_parallel_probe_timeout_s": 5.0,
        "data_parallel_size": 8,
        "data_parallel_size_local": 4,
        "data_parallel_start_rank": None,
        "data_parallel_rank": None,
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
        "api_server_count": None,
        "headless": False,
        "grpc": False,
        "uds": None,
        "ssl_keyfile": None,
        "ssl_certfile": None,
        "ssl_ca_certs": None,
        "node_rank": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "uvicorn_log_level": "info",
        "shutdown_timeout": 5.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_infer_multi_port_external_lb_start_rank_uses_node_rank():
    args = _make_args()
    assert infer_multi_port_external_lb_start_rank(args) == 4


def test_build_multi_port_external_lb_child_args_sets_external_rank_server():
    args = _make_args(data_parallel_start_rank=8, api_server_count=None)
    child_args = build_multi_port_external_lb_child_args(args, local_rank=2)

    assert child_args.port == 8002
    assert child_args.data_parallel_rank == 10
    assert child_args.data_parallel_size_local == 1
    assert child_args.data_parallel_external_lb is True
    assert child_args.data_parallel_hybrid_lb is False
    assert child_args.data_parallel_multi_port_external_lb is False
    assert child_args.api_server_count == 1


def test_dp_supervisor_aggregates_health():
    supervisor = DPSupervisor(_make_args())

    supervisor.children_healthy = True

    assert supervisor.is_healthy() is True


def test_dp_supervisor_is_unhealthy_after_shutdown_requested():
    supervisor = DPSupervisor(_make_args())
    supervisor.children_healthy = True
    supervisor._shutdown_event.set()

    assert supervisor.is_healthy() is False


@pytest.mark.asyncio
async def test_dp_supervisor_monitor_children_raises_when_child_exits(
    monkeypatch: pytest.MonkeyPatch,
):
    supervisor = DPSupervisor(_make_args())
    failed_process = SimpleNamespace(name="ExternalLBRank_5", exitcode=17)
    supervisor.processes = [
        SimpleNamespace(name="ExternalLBRank_4", exitcode=None),
        failed_process,
    ]

    async def fake_probe(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(dp_supervisor, "_probe_endpoint", fake_probe)

    with pytest.raises(
        RuntimeError, match="Multi-port external LB child exited unexpectedly"
    ) as exc_info:
        await supervisor._monitor_children()

    assert "ExternalLBRank_5" in str(exc_info.value)
    assert "exit code 17" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dp_supervisor_run_propagates_supervisor_server_error_before_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeLoop:
        def add_signal_handler(self, *_args, **_kwargs):
            pass

        def remove_signal_handler(self, *_args, **_kwargs):
            pass

    class FakeServer:
        def __init__(self, _config):
            self.started = False
            self.should_exit = False

        async def serve(self):
            raise ValueError("supervisor boom")

    async def fake_shutdown_children(self):
        return None

    monkeypatch.setattr(dp_supervisor.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(dp_supervisor.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(DPSupervisor, "_shutdown_children", fake_shutdown_children)

    supervisor = DPSupervisor(_make_args())

    with pytest.raises(ValueError, match="supervisor boom"):
        await supervisor.run()
