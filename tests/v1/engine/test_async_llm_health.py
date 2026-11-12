# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `/health_generate`: the EngineCore decides each rank's
health state, and AsyncLLM aggregates the ranks into one answer."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.v1.engine import EngineHealthState
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.exceptions import EngineSleepingError, EngineUnhealthyError

OK = EngineHealthState.OK
BUSY = EngineHealthState.BUSY
SLEEPING = EngineHealthState.SLEEPING


class FakeClock:
    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.now = 1000.0
        monkeypatch.setattr("vllm.v1.engine.core.time.monotonic", lambda: self.now)


def make_core(*, has_requests: bool = False) -> EngineCoreProc:
    core = object.__new__(EngineCoreProc)
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=has_requests))
    core.batch_queue = None
    core.engines_running = False
    core.is_sleeping = Mock(return_value=False)  # type: ignore[method-assign]
    core.model_executor = SimpleNamespace(check_health_gpu=Mock())
    core.step_fn = Mock(return_value=({}, False))
    core.post_step = Mock()  # type: ignore[method-assign]
    core._last_gpu_probe_at = 0.0
    return core


def test_engine_idle_probe_cache(monkeypatch: pytest.MonkeyPatch):
    """Passing probes are cached for the TTL; failures and model steps are
    not."""
    clock = FakeClock(monkeypatch)
    core = make_core()
    probe = core.model_executor.check_health_gpu

    core.check_health_gpu(10)
    core.check_health_gpu(10)
    assert probe.call_count == 1

    clock.now += 11
    core.check_health_gpu(10)
    assert probe.call_count == 2

    core.step_fn.return_value = ({}, True)
    core._process_engine_step()
    core.check_health_gpu(10)
    assert probe.call_count == 3

    clock.now += 11
    probe.side_effect = RuntimeError("CUDA error")
    with pytest.raises(RuntimeError):
        core.check_health_gpu(10)
    probe.side_effect = None
    core.check_health_gpu(10)
    assert probe.call_count == 5


@pytest.mark.parametrize(
    ("has_requests", "sleeping", "expected"),
    [(True, False, BUSY), (False, True, SLEEPING)],
)
def test_engine_skips_probe_when_busy_or_sleeping(
    has_requests: bool, sleeping: bool, expected: EngineHealthState
):
    """Busy ranks get no extra GPU work, and sleeping ranks are not probed."""
    core = make_core(has_requests=has_requests)
    core.is_sleeping.return_value = sleeping

    assert core.check_health_gpu(10) == expected
    core.model_executor.check_health_gpu.assert_not_called()


def make_async_engine(*rank_results) -> AsyncLLM:
    """AsyncLLM whose engine rank `i` answers the probe with
    `rank_results[i]`: a health state, an exception, or `None` to hang."""

    async def probe(rank: int, *_args):
        result = rank_results[rank]
        if result is None:
            await asyncio.Future()
        if isinstance(result, BaseException):
            raise result
        return result

    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        get_engine_ranks=Mock(return_value=list(range(len(rank_results)))),
        check_health_gpu_async=AsyncMock(side_effect=probe),
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
        pytest.param([SLEEPING, OK], "sleeping", [0], id="sleeping"),
        pytest.param([BUSY, None], "probe_timeout", [1], id="one-dp-rank-hung"),
        pytest.param(
            [OK, RuntimeError("CUDA error")], "probe_failed", [1], id="probe-failed"
        ),
    ],
)
async def test_check_health_gpu_aggregates_engine_ranks(
    rank_results, reason, engine_ranks
):
    engine = make_async_engine(*rank_results)

    error = EngineSleepingError if reason == "sleeping" else EngineUnhealthyError
    with pytest.raises(error) as exc_info:
        await engine.check_health_gpu(0.05)

    assert exc_info.value.reason == reason
    assert exc_info.value.details["engine_ranks"] == engine_ranks
    assert not engine.errored


@pytest.mark.asyncio
async def test_concurrent_health_checks_share_one_probe():
    """A probe outlives callers that time out, and later callers reuse it
    instead of queueing another one behind a blocked busy loop."""
    finish = asyncio.Event()

    async def probe(*_args):
        await finish.wait()
        return OK

    engine = make_async_engine(OK)
    engine.engine_core.check_health_gpu_async.side_effect = probe

    for _ in range(2):
        with pytest.raises(EngineUnhealthyError, match="timed out"):
            await engine.check_health_gpu(0.05)
    finish.set()
    await engine.check_health_gpu(0.05)

    assert engine.engine_core.check_health_gpu_async.await_count == 1
