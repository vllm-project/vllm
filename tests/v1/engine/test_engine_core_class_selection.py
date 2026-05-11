# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import DeviceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.engine.utils import CoreEngineProcManager


class SelectedInprocCore:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class SelectedEngineCoreProc:
    @staticmethod
    def run_engine_core(*args, **kwargs):
        raise NotImplementedError


class FakeProcess:
    sentinel = object()
    exitcode = None
    pid = None

    def __init__(self, target, name, kwargs):
        self.target = target
        self.name = name
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True

    def is_alive(self):
        return False

    def terminate(self):
        raise AssertionError("FakeProcess should not be terminated")

    def join(self, timeout=None):
        return None


class FakeMpContext:
    def __init__(self):
        self.processes: list[FakeProcess] = []

    def Process(self, target, name, kwargs):
        process = FakeProcess(target=target, name=name, kwargs=kwargs)
        self.processes.append(process)
        return process


@pytest.mark.skip_global_cleanup
def test_inproc_client_uses_configured_engine_core_cls():
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            engine_core_cls=f"{__name__}.SelectedInprocCore"
        ),
    )

    client = InprocClient(vllm_config=vllm_config)

    assert isinstance(client.engine_core, SelectedInprocCore)
    assert client.engine_core.kwargs["vllm_config"] is vllm_config


@pytest.mark.skip_global_cleanup
def test_proc_manager_uses_configured_engine_core_proc_cls(monkeypatch):
    fake_context = FakeMpContext()
    monkeypatch.setattr("vllm.v1.engine.utils.get_mp_context", lambda: fake_context)
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            engine_core_proc_cls=f"{__name__}.SelectedEngineCoreProc"
        ),
    )

    CoreEngineProcManager(
        local_engine_count=1,
        start_index=0,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=True,
        handshake_address="inproc://test",
        executor_class=object,
        log_stats=False,
        tensor_queue=None,
    )

    assert len(fake_context.processes) == 1
    assert fake_context.processes[0].target is SelectedEngineCoreProc.run_engine_core
    assert fake_context.processes[0].started
