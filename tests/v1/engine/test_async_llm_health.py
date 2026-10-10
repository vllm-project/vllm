# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import vllm.envs as envs
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreReadyState,
    EngineCoreRequestType,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    READY_PROGRESS_BROADCAST_CLIENT_INDEX,
    DPEngineCoreProc,
    EngineCore,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreReadyProgress
from vllm.v1.engine.exceptions import EngineSleepingError, EngineUnhealthyError


def make_async_engine(core, data_parallel_size: int = 1) -> AsyncLLM:
    if not hasattr(core, "get_sleeping_engine_ranks"):
        core.get_sleeping_engine_ranks = Mock(return_value=[])
    if not hasattr(core, "get_in_progress_operations"):
        core.get_in_progress_operations = Mock(return_value=[])
    engine = object.__new__(AsyncLLM)
    engine.engine_core = core
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size)
    )
    core.resources = SimpleNamespace(engine_dead=False)
    core.shutdown = Mock()
    engine.output_handler = None
    engine.check_health = AsyncMock()  # type: ignore[method-assign]
    engine._idle_health_probe_tasks = {}
    return engine


@pytest.mark.asyncio
async def test_check_health_gpu_sleeping_is_not_ready():
    core = SimpleNamespace(
        get_sleeping_engine_ranks=Mock(return_value=[0]),
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[]),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core)

    with pytest.raises(EngineSleepingError, match=r"engine ranks: \[0\]"):
        await engine.check_health_gpu()

    core.get_stalled_engine_ranks.assert_not_called()
    core.check_health_gpu_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_gpu_busy_with_progress():
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[]),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core)

    await engine.check_health_gpu()

    core.check_health_gpu_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_parallel_size", [1, 2])
async def test_check_health_gpu_busy_stalled(data_parallel_size: int):
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[1]),
        get_idle_engine_ranks=Mock(return_value=[]),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core, data_parallel_size=data_parallel_size)

    with pytest.raises(EngineUnhealthyError, match=r"engine ranks: \[1\]"):
        await engine.check_health_gpu()

    core.check_health_gpu_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_gpu_reports_reason_and_in_progress():
    operation = {
        "operation": "collective_rpc:update_weights",
        "engine_rank": 0,
        "elapsed_s": 70.0,
    }
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[0]),
        get_idle_engine_ranks=Mock(return_value=[]),
        check_health_gpu_async=AsyncMock(),
        get_in_progress_operations=Mock(return_value=[operation]),
    )
    engine = make_async_engine(core)

    with pytest.raises(EngineUnhealthyError) as exc_info:
        await engine.check_health_gpu()

    assert exc_info.value.reason == "stalled"
    assert exc_info.value.details["engine_ranks"] == [0]
    assert exc_info.value.details["in_progress"] == [operation]


@pytest.mark.asyncio
async def test_idle_gpu_probe_delegates_cache_to_engine_core(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_READY_IDLE_PROBE_CACHE_TTL_S", 10.0)
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[0]),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core)

    await engine.check_health_gpu()
    await engine.check_health_gpu()

    assert core.check_health_gpu_async.await_count == 2
    core.check_health_gpu_async.assert_awaited_with(10.0, 0)


@pytest.mark.asyncio
async def test_check_health_gpu_dp_probes_each_idle_rank():
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[0, 1]),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core, data_parallel_size=2)

    await engine.check_health_gpu()

    core.get_stalled_engine_ranks.assert_called_once()
    assert core.check_health_gpu_async.await_count == 2


@pytest.mark.asyncio
async def test_check_health_gpu_probes_idle_rank_while_another_is_busy(
    monkeypatch: pytest.MonkeyPatch,
):
    """A dense DP rank wedged while IDLE is caught even when another rank is
    busy and making progress."""
    monkeypatch.setattr(envs, "VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.01)

    async def probe(_cache_ttl_s: float, _engine_rank: int):
        await asyncio.Future()

    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[1]),
        check_health_gpu_async=AsyncMock(side_effect=probe),
    )
    engine = make_async_engine(core, data_parallel_size=2)

    with pytest.raises(EngineUnhealthyError) as exc_info:
        await engine.check_health_gpu()

    assert exc_info.value.reason == "probe_timeout"
    assert exc_info.value.details["engine_ranks"] == [1]


@pytest.mark.asyncio
async def test_concurrent_idle_gpu_probes_share_one_task(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_READY_IDLE_PROBE_CACHE_TTL_S", 10.0)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def probe(_cache_ttl_s: float, _engine_rank: int):
        started.set()
        await finish.wait()

    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[0]),
        check_health_gpu_async=AsyncMock(side_effect=probe),
    )
    engine = make_async_engine(core)

    probes = [asyncio.create_task(engine.check_health_gpu()) for _ in range(2)]
    await started.wait()
    await asyncio.sleep(0)
    assert core.check_health_gpu_async.await_count == 1

    finish.set()
    await asyncio.gather(*probes)


@pytest.mark.asyncio
async def test_idle_gpu_probe_timeout_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.01)

    async def probe(_cache_ttl_s: float, _engine_rank: int):
        await asyncio.Future()

    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        get_idle_engine_ranks=Mock(return_value=[0]),
        check_health_gpu_async=AsyncMock(side_effect=probe),
    )
    engine = make_async_engine(core)

    with pytest.raises(EngineUnhealthyError, match="timed out"):
        await engine.check_health_gpu()

    assert not engine.errored


@pytest.mark.asyncio
async def test_async_client_tracks_progress_and_recovery():
    client = object.__new__(AsyncMPClient)
    client._ready_progress = {0: EngineCoreReadyProgress()}

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=1,
            ready_state=EngineCoreReadyState.BUSY,
        ),
    )
    progress = client._ready_progress[0]
    assert progress.state == EngineCoreReadyState.BUSY
    assert progress.seq == 1
    assert client.get_stalled_engine_ranks(60) == []

    progress.last_progress_at = time.monotonic() - 61
    assert client.get_stalled_engine_ranks(60) == [0]

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=2,
            ready_state=EngineCoreReadyState.BUSY,
        ),
    )
    assert client.get_stalled_engine_ranks(60) == []

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=2,
            ready_state=EngineCoreReadyState.IDLE,
        ),
    )
    assert client.get_idle_engine_ranks() == [0]

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=2,
            ready_state=EngineCoreReadyState.SLEEPING,
        ),
    )
    assert client.get_sleeping_engine_ranks() == [0]
    assert client.get_idle_engine_ranks() == []


def test_async_client_detects_one_stalled_dp_engine():
    now = time.monotonic()
    client = object.__new__(AsyncMPClient)
    client._ready_progress = {
        0: EngineCoreReadyProgress(
            seq=3, state=EngineCoreReadyState.BUSY, last_progress_at=now
        ),
        1: EngineCoreReadyProgress(
            seq=7,
            state=EngineCoreReadyState.BUSY,
            last_progress_at=now - 61,
        ),
    }

    assert client.get_stalled_engine_ranks(60) == [1]
    assert client.get_idle_engine_ranks() == []


@pytest.mark.asyncio
async def test_async_client_ignores_untracked_engines():
    """Engines removed by an elastic EP scale-down are paused on their way
    out; their SLEEPING broadcast must not make `/ready` sleeping forever."""
    client = object.__new__(AsyncMPClient)
    client._ready_progress = {0: EngineCoreReadyProgress()}

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=1,
            ready_progress_seq=0,
            ready_state=EngineCoreReadyState.SLEEPING,
        ),
    )

    assert client.get_sleeping_engine_ranks() == []


@pytest.mark.asyncio
async def test_idle_gpu_probe_targets_engine_rank():
    client = object.__new__(AsyncMPClient)
    client._call_utility_async = AsyncMock()

    await AsyncMPClient.check_health_gpu_async(client, 10, 1)

    client._call_utility_async.assert_awaited_once_with(
        "check_health_gpu", 10, engine=b"\x01\x00"
    )


def make_core() -> EngineCore:
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=False))
    core.batch_queue = None
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    core.model_executor = SimpleNamespace(check_health_gpu=Mock())
    core._last_health_dummy_batch_at = 0.0
    core._ready_progress_seq = 0
    return core


def test_engine_core_idle_probe_cache(monkeypatch: pytest.MonkeyPatch):
    core = make_core()
    now = 100.0
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic", lambda: now)

    core.check_health_gpu(10)
    core.check_health_gpu(10)
    now = 111.0
    core.check_health_gpu(10)

    assert core.model_executor.check_health_gpu.call_count == 2


def test_engine_core_busy_probe_does_not_run_dummy():
    core = make_core()
    core.scheduler.has_requests.return_value = True

    core.check_health_gpu(10)

    core.model_executor.check_health_gpu.assert_not_called()


def test_engine_core_sleeping_probe_does_not_run_dummy():
    core = make_core()
    core.is_sleeping.return_value = True

    core.check_health_gpu(10)

    core.model_executor.check_health_gpu.assert_not_called()


def test_engine_core_dp_wave_does_not_run_probe_dummy():
    core = make_core()
    core.engines_running = True

    core.check_health_gpu(10)

    core.model_executor.check_health_gpu.assert_not_called()


def test_engine_core_probe_failure_is_not_cached():
    core = make_core()
    core.model_executor.check_health_gpu.side_effect = [RuntimeError(), None]

    with pytest.raises(RuntimeError):
        core.check_health_gpu(10)
    core.check_health_gpu(10)

    assert core.model_executor.check_health_gpu.call_count == 2


def test_engine_core_progress_sequence():
    core = make_core()
    core._last_health_dummy_batch_at = 100.0

    core._record_ready_progress()
    core._record_ready_progress()

    assert core._ready_progress_seq == 2
    assert core._last_health_dummy_batch_at == 0.0


@pytest.mark.parametrize("has_unfinished_requests", [True, False])
def test_engine_core_connector_cleanup_is_not_stalled(has_unfinished_requests: bool):
    """Zero-token steps count as progress only when no unfinished requests
    remain, e.g. finished requests awaiting delayed KV connector frees."""
    core = object.__new__(EngineCoreProc)
    core.output_queue = queue.Queue()
    core._ready_progress_seq = 0
    core._last_ready_published_state = EngineCoreReadyState.BUSY
    core._last_ready_published_seq = 0
    core._last_ready_published_at = 0.0
    core._ready_operation = None
    core._last_health_dummy_batch_at = 0.0
    core.engines_running = False
    core.scheduler = SimpleNamespace(
        has_requests=Mock(return_value=True),
        has_unfinished_requests=Mock(return_value=has_unfinished_requests),
    )
    core.batch_queue = None
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    core.step_fn = Mock(return_value=({}, False))
    core.post_step = Mock()  # type: ignore[method-assign]

    for _ in range(3):
        core._process_engine_step()

    assert core._ready_progress_seq == (0 if has_unfinished_requests else 3)


def test_engine_core_broadcasts_busy_and_throttles_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    core = object.__new__(EngineCoreProc)
    core.output_queue = queue.Queue()
    core._ready_progress_seq = 0
    core._last_ready_published_state = EngineCoreReadyState.IDLE
    core._last_ready_published_seq = 0
    core._last_ready_published_at = 0.0
    core._ready_operation = None
    core._last_health_dummy_batch_at = 100.0
    core.engines_running = True
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=False))
    core.batch_queue = None
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    now = 1.0
    monkeypatch.setattr("vllm.v1.engine.core.time.monotonic", lambda: now)

    core._maybe_publish_ready_progress()
    client_index, output = core.output_queue.get_nowait()
    assert client_index == READY_PROGRESS_BROADCAST_CLIENT_INDEX
    assert output.ready_state == EngineCoreReadyState.BUSY
    assert output.ready_progress_seq == 0
    assert core._last_health_dummy_batch_at == 0.0

    core._ready_progress_seq = 1
    now = 1.5
    core._maybe_publish_ready_progress()
    with pytest.raises(queue.Empty):
        core.output_queue.get_nowait()

    now = 2.1
    core._maybe_publish_ready_progress()
    _, output = core.output_queue.get_nowait()
    assert output.ready_progress_seq == 1

    core.is_sleeping.return_value = True
    core._maybe_publish_ready_progress()
    _, output = core.output_queue.get_nowait()
    assert output.ready_state == EngineCoreReadyState.SLEEPING


def test_dp_engine_core_probe_is_liveness_only():
    """A dummy batch outside the wave loop could deadlock MoE DP peers."""
    core = object.__new__(DPEngineCoreProc)
    core.model_executor = SimpleNamespace(check_health_gpu=Mock())

    core.check_health_gpu(10)

    core.model_executor.check_health_gpu.assert_not_called()


def test_engine_core_publishes_operation_around_utility_call():
    """The running utility is broadcast before it starts and cleared after, so
    every API server can report what a blocked busy loop is doing."""
    core = object.__new__(EngineCoreProc)
    core.output_queue = queue.Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    core._ready_progress_seq = 0
    core._last_ready_published_state = EngineCoreReadyState.IDLE
    core._last_ready_published_seq = 0
    core._last_ready_published_at = 0.0
    core._ready_operation = None
    core._last_health_dummy_batch_at = 0.0
    core.engines_running = False
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=False))
    core.batch_queue = None
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    seen_during_call = []
    core.collective_rpc = lambda method: seen_during_call.append(  # type: ignore[method-assign]
        core._ready_operation
    )

    core._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 1, "collective_rpc", ("update_weights",))
    )

    assert seen_during_call == ["collective_rpc:update_weights"]
    published = [
        output.ready_operation
        for client_index, output in core.output_queue.queue
        if client_index == READY_PROGRESS_BROADCAST_CLIENT_INDEX
    ]
    assert published == ["collective_rpc:update_weights", None]


@pytest.mark.asyncio
async def test_async_client_reports_engine_operations():
    client = object.__new__(AsyncMPClient)
    client._ready_progress = {
        0: EngineCoreReadyProgress(),
        1: EngineCoreReadyProgress(),
    }

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=0,
            ready_state=EngineCoreReadyState.IDLE,
            ready_operation="collective_rpc:update_weights",
        ),
    )
    (operation,) = client.get_in_progress_operations()
    assert operation["operation"] == "collective_rpc:update_weights"
    assert operation["engine_rank"] == 0

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0, ready_progress_seq=0, ready_state=EngineCoreReadyState.IDLE
        ),
    )
    assert client.get_in_progress_operations() == []
