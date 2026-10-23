# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `/health_generate`: the EngineCore decides each rank's
health state, and AsyncLLM aggregates the ranks into one answer."""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import vllm.envs as envs
from vllm.v1.engine import EngineCoreRequestType, EngineHealthState
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import (
    DPEngineCoreProc,
    EngineCoreProc,
    EngineShutdownState,
)
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.engine.exceptions import EngineSleepingError, EngineUnhealthyError

OK = EngineHealthState.OK
BUSY = EngineHealthState.BUSY
STALLED = EngineHealthState.STALLED
SLEEPING = EngineHealthState.SLEEPING


class FakeClock:
    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.now = 1000.0
        monkeypatch.setattr("vllm.v1.engine.core.time.monotonic", lambda: self.now)


def make_core(
    cls: type[EngineCoreProc] = EngineCoreProc,
    *,
    has_requests: bool = False,
    has_unfinished_requests: bool | None = None,
    sleeping: bool = False,
    engines_running: bool = False,
) -> EngineCoreProc:
    core = object.__new__(cls)
    core.scheduler = SimpleNamespace(
        has_requests=Mock(return_value=has_requests),
        has_unfinished_requests=Mock(
            return_value=(
                has_requests
                if has_unfinished_requests is None
                else has_unfinished_requests
            )
        ),
    )
    core.batch_queue = None
    core.engines_running = engines_running
    core.is_sleeping = Mock(return_value=sleeping)  # type: ignore[method-assign]
    core.model_executor = SimpleNamespace(check_health_gpu=Mock())
    core.output_queue = queue.Queue()
    core.shutdown_state = EngineShutdownState.RUNNING
    core.step_fn = Mock(return_value=({}, False))
    core.post_step = Mock()  # type: ignore[method-assign]
    core._was_busy = False
    core._last_progress_at = 0.0
    core._last_gpu_probe_at = 0.0
    core._current_operation = None
    return core


@pytest.mark.parametrize(
    ("cls", "core_kwargs", "since_progress_s", "stall_timeout_s", "expected", "probed"),
    [
        pytest.param(EngineCoreProc, {}, 0, 60, OK, True, id="idle-probes"),
        pytest.param(
            EngineCoreProc, {"sleeping": True}, 0, 60, SLEEPING, False, id="sleeping"
        ),
        pytest.param(
            EngineCoreProc, {"has_requests": True}, 5, 60, BUSY, False, id="busy"
        ),
        pytest.param(
            EngineCoreProc, {"has_requests": True}, 75, 60, STALLED, False, id="stalled"
        ),
        pytest.param(
            EngineCoreProc,
            {"has_requests": True},
            75,
            0,
            BUSY,
            False,
            id="stall-detection-disabled",
        ),
        pytest.param(
            DPEngineCoreProc,
            {"engines_running": True},
            5,
            60,
            BUSY,
            False,
            id="dp-wave-without-local-requests",
        ),
        pytest.param(
            DPEngineCoreProc, {}, 0, 60, OK, False, id="moe-dp-idle-is-liveness-only"
        ),
    ],
)
def test_engine_check_health_gpu(
    monkeypatch: pytest.MonkeyPatch,
    cls,
    core_kwargs,
    since_progress_s,
    stall_timeout_s,
    expected,
    probed,
):
    clock = FakeClock(monkeypatch)
    core = make_core(cls, **core_kwargs)
    core._was_busy = True
    core._last_progress_at = clock.now - since_progress_s

    state, _ = core.check_health_gpu(10, stall_timeout_s)

    assert state == expected
    assert core.model_executor.check_health_gpu.called == probed


def test_engine_idle_probe_cache(monkeypatch: pytest.MonkeyPatch):
    """Passing probes are cached for the TTL; failures and model steps are
    not."""
    clock = FakeClock(monkeypatch)
    core = make_core()
    probe = core.model_executor.check_health_gpu

    core.check_health_gpu(10, 60)
    core.check_health_gpu(10, 60)
    assert probe.call_count == 1

    clock.now += 11
    core.check_health_gpu(10, 60)
    assert probe.call_count == 2

    core._mark_progress()
    core.check_health_gpu(10, 60)
    assert probe.call_count == 3

    clock.now += 11
    probe.side_effect = RuntimeError("CUDA error")
    with pytest.raises(RuntimeError):
        core.check_health_gpu(10, 60)
    probe.side_effect = None
    core.check_health_gpu(10, 60)
    assert probe.call_count == 5


@pytest.mark.parametrize("paused", [False, True])
def test_engine_stall_clock_restarts_when_engine_becomes_busy(
    monkeypatch: pytest.MonkeyPatch, paused: bool
):
    """Time spent idle or paused is not a stall, even when the probe arrives
    in the same input batch as the request that makes the engine busy."""
    clock = FakeClock(monkeypatch)
    core = make_core(has_requests=paused, sleeping=paused)
    core._update_progress_clock()

    clock.now += 3600
    core.scheduler.has_requests.return_value = True
    core.is_sleeping.return_value = False

    assert core.check_health_gpu(10, 60)[0] == BUSY


@pytest.mark.parametrize(
    ("model_executed", "has_unfinished_requests", "expected"),
    [
        pytest.param(True, True, BUSY, id="scheduled-tokens"),
        pytest.param(False, True, STALLED, id="no-tokens-scheduled"),
        pytest.param(False, False, BUSY, id="connector-cleanup-only"),
    ],
)
def test_engine_step_progress(
    monkeypatch: pytest.MonkeyPatch,
    model_executed: bool,
    has_unfinished_requests: bool,
    expected: EngineHealthState,
):
    clock = FakeClock(monkeypatch)
    core = make_core(has_requests=True, has_unfinished_requests=has_unfinished_requests)
    core.step_fn.return_value = ({}, model_executed)

    for _ in range(4):
        core._process_engine_step()
        clock.now += 30

    assert core.check_health_gpu(10, 60)[0] == expected


def test_engine_records_operation_around_utility_call(
    monkeypatch: pytest.MonkeyPatch,
):
    clock = FakeClock(monkeypatch)
    core = make_core()
    seen_during_call = []

    def collective_rpc(method: str):
        clock.now += 5
        seen_during_call.append(core.get_current_operation())

    core.collective_rpc = collective_rpc  # type: ignore[method-assign]

    core._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 1, "collective_rpc", ("update_weights",))
    )

    assert seen_during_call == [("collective_rpc:update_weights", 5.0)]
    assert core.get_current_operation() is None


def make_async_engine(*rank_results) -> AsyncLLM:
    """AsyncLLM whose engine rank ``i`` answers the probe with
    ``rank_results[i]``: a health state, an exception, or ``None`` to hang."""

    async def probe(rank: int, _cache_ttl_s: float, _stall_timeout_s: float):
        result = rank_results[rank]
        if result is None:
            await asyncio.Future()
        if isinstance(result, BaseException):
            raise result
        return result, 0.0

    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        get_engine_ranks=Mock(return_value=list(range(len(rank_results)))),
        check_health_gpu_async=AsyncMock(side_effect=probe),
        get_current_operation_async=AsyncMock(return_value=None),
        resources=SimpleNamespace(engine_dead=False),
        shutdown=Mock(),
    )
    engine.output_handler = None
    engine.check_health = AsyncMock()  # type: ignore[method-assign]
    engine._health_probe_tasks = {}
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rank_results", "reason", "engine_ranks"),
    [
        pytest.param([OK], None, None, id="idle-ok"),
        pytest.param([BUSY, OK], None, None, id="dp-busy-and-idle"),
        pytest.param([SLEEPING, OK], "sleeping", [0], id="sleeping"),
        pytest.param([BUSY, STALLED], "stalled", [1], id="one-dp-rank-stalled"),
        pytest.param([BUSY, None], "probe_timeout", [1], id="one-dp-rank-hung"),
        pytest.param(
            [OK, RuntimeError("CUDA error")], "probe_failed", [1], id="probe-failed"
        ),
    ],
)
async def test_check_health_gpu_aggregates_engine_ranks(
    monkeypatch: pytest.MonkeyPatch, rank_results, reason, engine_ranks
):
    monkeypatch.setattr(envs, "VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.05)
    engine = make_async_engine(*rank_results)
    operation = ("collective_rpc:update_weights", 70.0)
    engine.engine_core.get_current_operation_async.side_effect = lambda rank: (
        operation if rank == 0 else None
    )

    if reason is None:
        await engine.check_health_gpu()
        return
    error = EngineSleepingError if reason == "sleeping" else EngineUnhealthyError
    with pytest.raises(error) as exc_info:
        await engine.check_health_gpu()

    assert exc_info.value.reason == reason
    assert exc_info.value.details["engine_ranks"] == engine_ranks
    assert exc_info.value.details["in_progress"] == [
        {"operation": operation[0], "engine_rank": 0, "elapsed_s": 70.0}
    ]
    assert not engine.errored


@pytest.mark.asyncio
async def test_concurrent_health_checks_share_one_probe(
    monkeypatch: pytest.MonkeyPatch,
):
    """A probe outlives callers that time out, and later callers reuse it
    instead of queueing another one behind a blocked busy loop."""
    monkeypatch.setattr(envs, "VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.05)
    finish = asyncio.Event()

    async def probe(*_args):
        await finish.wait()
        return OK, 0.0

    engine = make_async_engine(OK)
    engine.engine_core.check_health_gpu_async.side_effect = probe

    for _ in range(2):
        with pytest.raises(EngineUnhealthyError, match="timed out"):
            await engine.check_health_gpu()
    finish.set()
    await engine.check_health_gpu()

    assert engine.engine_core.check_health_gpu_async.await_count == 1


@pytest.mark.asyncio
async def test_async_client_health_utilities_target_engine_rank():
    client = object.__new__(AsyncMPClient)
    client._call_utility_async = AsyncMock(return_value=["stalled", 75.0])

    assert await client.check_health_gpu_async(1, 10, 60) == (STALLED, 75.0)
    client._call_utility_async.assert_awaited_once_with(
        "check_health_gpu", 10, 60, engine=b"\x01\x00"
    )

    client._call_utility_async = AsyncMock(return_value=["sleep", 3.0])
    assert await client.get_current_operation_async(1) == ("sleep", 3.0)
    client._call_utility_async.assert_awaited_once_with(
        "get_current_operation", engine=b"\x01\x00"
    )
