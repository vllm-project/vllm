# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import vllm.envs as envs
from vllm.v1.engine import EngineCoreOutputs, EngineCoreReadyState
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    READY_PROGRESS_BROADCAST_CLIENT_INDEX,
    EngineCore,
    EngineCoreProc,
)
from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreReadyProgress
from vllm.v1.engine.exceptions import EngineSleepingError, EngineUnhealthyError


def make_async_engine(core, data_parallel_size: int = 1) -> AsyncLLM:
    if not hasattr(core, "get_sleeping_engine_ranks"):
        core.get_sleeping_engine_ranks = Mock(return_value=[])
    engine = object.__new__(AsyncLLM)
    engine.engine_core = core
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size)
    )
    core.resources = SimpleNamespace(engine_dead=False)
    core.shutdown = Mock()
    engine.output_handler = None
    engine.check_health = AsyncMock()  # type: ignore[method-assign]
    engine._idle_health_probe_lock = asyncio.Lock()
    engine._idle_health_probe_task = None
    return engine


@pytest.mark.asyncio
async def test_check_health_gpu_sleeping_is_not_ready():
    core = SimpleNamespace(
        get_sleeping_engine_ranks=Mock(return_value=[0]),
        get_stalled_engine_ranks=Mock(return_value=[]),
        all_engines_idle=Mock(return_value=False),
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
        all_engines_idle=Mock(return_value=False),
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
        all_engines_idle=Mock(return_value=False),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core, data_parallel_size=data_parallel_size)

    with pytest.raises(EngineUnhealthyError, match=r"engine ranks: \[1\]"):
        await engine.check_health_gpu()

    core.check_health_gpu_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_gpu_probe_delegates_cache_to_engine_core(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_READY_IDLE_PROBE_CACHE_TTL_S", 10.0)
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        all_engines_idle=Mock(return_value=True),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core)

    await engine.check_health_gpu()
    await engine.check_health_gpu()

    assert core.check_health_gpu_async.await_count == 2
    core.check_health_gpu_async.assert_awaited_with(10.0)


@pytest.mark.asyncio
async def test_check_health_gpu_dp_only_checks_rank_progress():
    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        all_engines_idle=Mock(return_value=True),
        check_health_gpu_async=AsyncMock(),
    )
    engine = make_async_engine(core, data_parallel_size=2)

    await engine.check_health_gpu()

    core.get_stalled_engine_ranks.assert_called_once()
    core.all_engines_idle.assert_not_called()
    core.check_health_gpu_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_idle_gpu_probes_share_one_task(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_READY_IDLE_PROBE_CACHE_TTL_S", 10.0)
    started = asyncio.Event()
    finish = asyncio.Event()

    async def probe(_cache_ttl_s: float):
        started.set()
        await finish.wait()

    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        all_engines_idle=Mock(return_value=True),
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

    async def probe(_cache_ttl_s: float):
        await asyncio.Future()

    core = SimpleNamespace(
        get_stalled_engine_ranks=Mock(return_value=[]),
        all_engines_idle=Mock(return_value=True),
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
    assert client.all_engines_idle()

    await AsyncMPClient.process_engine_outputs(
        client,
        EngineCoreOutputs(
            engine_index=0,
            ready_progress_seq=2,
            ready_state=EngineCoreReadyState.SLEEPING,
        ),
    )
    assert client.get_sleeping_engine_ranks() == [0]
    assert not client.all_engines_idle()


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
    assert not client.all_engines_idle()


def test_async_client_keeps_sleeping_state_when_request_is_queued():
    client = object.__new__(AsyncMPClient)
    engine = b"\x00\x00"
    client._ready_engine_ranks = {engine: 0}
    client._ready_progress = {
        0: EngineCoreReadyProgress(state=EngineCoreReadyState.SLEEPING)
    }

    client._mark_engine_busy(engine)

    assert client.get_sleeping_engine_ranks() == [0]


@pytest.mark.asyncio
async def test_idle_gpu_probe_covers_all_managed_engines():
    client = object.__new__(AsyncMPClient)
    client.core_engines = [b"\x00\x00", b"\x01\x00"]
    client._call_utility_async = AsyncMock()

    await AsyncMPClient.check_health_gpu_async(client, 10)

    assert client._call_utility_async.await_count == 2
    client._call_utility_async.assert_any_await(
        "check_health_gpu", 10, engine=b"\x00\x00"
    )
    client._call_utility_async.assert_any_await(
        "check_health_gpu", 10, engine=b"\x01\x00"
    )


def make_core() -> EngineCore:
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=False))
    core.batch_queue = None
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    core.model_executor = SimpleNamespace(execute_dummy_batch=Mock())
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

    assert core.model_executor.execute_dummy_batch.call_count == 2


def test_engine_core_busy_probe_does_not_run_dummy():
    core = make_core()
    core.scheduler.has_requests.return_value = True

    core.check_health_gpu(10)

    core.model_executor.execute_dummy_batch.assert_not_called()


def test_engine_core_sleeping_probe_does_not_run_dummy():
    core = make_core()
    core.is_sleeping.return_value = True

    core.check_health_gpu(10)

    core.model_executor.execute_dummy_batch.assert_not_called()


def test_engine_core_dp_wave_does_not_run_probe_dummy():
    core = make_core()
    core.engines_running = True

    core.check_health_gpu(10)

    core.model_executor.execute_dummy_batch.assert_not_called()


def test_engine_core_probe_failure_is_not_cached():
    core = make_core()
    core.model_executor.execute_dummy_batch.side_effect = [RuntimeError(), None]

    with pytest.raises(RuntimeError):
        core.check_health_gpu(10)
    core.check_health_gpu(10)

    assert core.model_executor.execute_dummy_batch.call_count == 2


def test_engine_core_progress_sequence():
    core = make_core()
    core._last_health_dummy_batch_at = 100.0

    core._record_ready_progress()
    core._record_ready_progress()

    assert core._ready_progress_seq == 2
    assert core._last_health_dummy_batch_at == 0.0


def test_engine_core_broadcasts_busy_and_throttles_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    core = object.__new__(EngineCoreProc)
    core.output_queue = queue.Queue()
    core._ready_progress_seq = 0
    core._last_ready_published_state = EngineCoreReadyState.IDLE
    core._last_ready_published_seq = 0
    core._last_ready_published_at = 0.0
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
    regular_output = EngineCoreOutputs()
    core._maybe_publish_ready_progress({0: regular_output})
    _, output = core.output_queue.get_nowait()
    assert output.ready_progress_seq == 1
    assert regular_output.ready_progress_seq == 1
    assert regular_output.ready_state == EngineCoreReadyState.BUSY

    core.is_sleeping.return_value = True
    core._maybe_publish_ready_progress()
    _, output = core.output_queue.get_nowait()
    assert output.ready_state == EngineCoreReadyState.SLEEPING
