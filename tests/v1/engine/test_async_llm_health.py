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


def make_core(*, has_requests: bool, sleeping: bool) -> EngineCoreProc:
    core = object.__new__(EngineCoreProc)
    core.scheduler = SimpleNamespace(has_requests=Mock(return_value=has_requests))
    core.batch_queue = None
    core.engines_running = False
    core.is_sleeping = Mock(return_value=sleeping)  # type: ignore[method-assign]
    core.model_executor = SimpleNamespace(check_health_gpu=Mock())
    return core


@pytest.mark.parametrize(
    ("has_requests", "sleeping", "expected", "probed"),
    [
        pytest.param(False, False, OK, True, id="idle"),
        pytest.param(True, False, BUSY, False, id="busy"),
        pytest.param(False, True, SLEEPING, False, id="sleeping"),
    ],
)
def test_engine_probes_only_idle_ranks(
    has_requests: bool, sleeping: bool, expected: EngineHealthState, probed: bool
):
    """Idle ranks run the GPU probe; busy ranks get no extra GPU work, and
    sleeping ranks are not probed."""
    core = make_core(has_requests=has_requests, sleeping=sleeping)

    assert core.check_health_gpu() == expected
    assert core.model_executor.check_health_gpu.called == probed


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
