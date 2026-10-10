# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.exceptions import EngineDeadError, EngineUnhealthyError
from vllm.v1.metrics.stats import SchedulerStats


class FakeScheduler:
    def __init__(self, unfinished: bool):
        self.unfinished = unfinished

    def has_unfinished_requests(self) -> bool:
        return self.unfinished

    def add_request(self, request) -> None:
        self.unfinished = True


def make_request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
    )


def make_core(*, unfinished: bool = True) -> EngineCore:
    core = object.__new__(EngineCore)
    core.scheduler = FakeScheduler(unfinished)
    core._busy_since = time.monotonic()
    core._last_token_at = time.monotonic()
    return core


def make_async_engine(unhealthy_reason: str | None = None) -> AsyncLLM:
    engine = object.__new__(AsyncLLM)
    engine.engine_core = SimpleNamespace(
        check_ready_async=AsyncMock(return_value=unhealthy_reason),
        shutdown=Mock(),
    )
    engine.renderer = None
    engine.output_handler = None
    return engine


@pytest.mark.asyncio
async def test_async_llm_check_ready_delegates_to_engine_core():
    engine = make_async_engine()

    with patch.object(
        type(engine),
        "errored",
        new_callable=lambda: property(lambda self: False),
    ):
        await engine.check_ready()

    engine.engine_core.check_ready_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_llm_check_ready_raises_unhealthy():
    engine = make_async_engine("not ready")

    with (
        patch.object(
            type(engine),
            "errored",
            new_callable=lambda: property(lambda self: False),
        ),
        pytest.raises(EngineUnhealthyError),
    ):
        await engine.check_ready()


@pytest.mark.asyncio
async def test_async_llm_check_ready_dead_engine():
    engine = make_async_engine()

    with (
        patch.object(
            type(engine),
            "errored",
            new_callable=lambda: property(lambda self: True),
        ),
        patch.object(
            type(engine),
            "dead_error",
            new_callable=lambda: property(lambda self: EngineDeadError()),
        ),
        pytest.raises(EngineDeadError),
    ):
        await engine.check_ready()

    engine.engine_core.check_ready_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_llm_check_ready_times_out(monkeypatch: pytest.MonkeyPatch):
    engine = make_async_engine()

    async def never_returns():
        await asyncio.Future()

    engine.engine_core.check_ready_async = never_returns
    monkeypatch.setattr(envs, "VLLM_READY_CHECK_TIMEOUT_S", 0.01)

    with (
        patch.object(
            type(engine),
            "errored",
            new_callable=lambda: property(lambda self: False),
        ),
        pytest.raises(EngineUnhealthyError, match="readiness check"),
    ):
        await engine.check_ready()


def test_engine_core_check_ready_idle():
    core = make_core(unfinished=False)

    assert core.check_ready() is None


def test_engine_core_check_ready_allows_initial_progress_timeout():
    core = make_core()
    core._busy_since = time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S + 1
    core._last_token_at = 0.0

    assert core.check_ready() is None


def test_engine_core_check_ready_stalled_before_first_token():
    core = make_core()
    core._busy_since = time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S - 1
    core._last_token_at = 0.0

    assert core.check_ready() is not None


def test_engine_core_check_ready_stalled_after_token():
    core = make_core()
    core._busy_since = time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S - 1
    core._last_token_at = (
        time.monotonic() - envs.VLLM_READY_CHECK_PROGRESS_TIMEOUT_S - 1
    )

    assert core.check_ready() is not None


def test_engine_core_check_ready_recent_token():
    core = make_core()
    core._busy_since = time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S - 1
    core._last_token_at = time.monotonic()

    assert core.check_ready() is None


def test_engine_core_add_request_marks_busy_on_idle_transition():
    core = make_core(unfinished=False)
    core._busy_since = 0.0
    core._last_token_at = 0.0

    core.add_request(make_request("first"))

    assert core._busy_since > 0.0
    assert core.check_ready() is None


def test_engine_core_new_requests_do_not_mask_stall():
    core = make_core(unfinished=False)
    core.add_request(make_request("first"))
    stalled_since = time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S - 1
    core._busy_since = stalled_since

    core.add_request(make_request("second"))

    assert core._busy_since == stalled_since
    assert core.check_ready() is not None


def test_engine_core_progress_ignores_scheduler_stats():
    core = make_core()
    core._last_token_at = 0.0

    core._track_readiness_progress(
        {0: EngineCoreOutputs(scheduler_stats=SchedulerStats())}
    )

    assert core._last_token_at == 0.0


def test_engine_core_progress_tracks_generated_tokens():
    core = make_core()
    core._last_token_at = 0.0

    core._track_readiness_progress(
        {0: EngineCoreOutputs(outputs=[EngineCoreOutput("req", [1])])}
    )

    assert core._last_token_at > 0.0


def test_engine_core_progress_tracks_pooling_outputs():
    core = make_core()
    core._last_token_at = 0.0

    core._track_readiness_progress(
        {
            0: EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        "req",
                        [],
                        pooling_output=torch.zeros(4),
                        finish_reason=FinishReason.STOP,
                    )
                ]
            )
        }
    )

    assert core._last_token_at > 0.0


def test_engine_core_progress_tracks_finished_requests():
    core = make_core()
    core._last_token_at = 0.0

    core._track_readiness_progress(
        {
            0: EngineCoreOutputs(
                outputs=[EngineCoreOutput("req", [], finish_reason=FinishReason.ABORT)]
            )
        }
    )

    assert core._last_token_at > 0.0


def test_engine_core_reset_readiness_when_idle():
    core = make_core(unfinished=False)

    core._reset_readiness_if_idle()

    assert core._busy_since == 0.0
    assert core._last_token_at == 0.0
