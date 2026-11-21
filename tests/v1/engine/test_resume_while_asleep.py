# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`/resume` must not restart scheduling while the engine is asleep.

Sleeping at level 1+ offloads executor memory, but resuming scheduling and
stepping afterwards makes kernels access offloaded (CPU) tensors, which kills
EngineCore with "ValueError: Pointer argument cannot be accessed from Triton
(cpu tensor?)". `EngineCore.resume_scheduler` must refuse while executor
memory is asleep, and the `/resume` endpoint must surface the refusal as a
client error instead of reporting a successful resume.
"""

import asyncio
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import CheckpointState
from vllm.v1.engine.core import EngineCoreProc

pytestmark = pytest.mark.cpu_test


def _stub_core() -> EngineCoreProc:
    """A bare EngineCoreProc holding only the state resume_scheduler touches."""
    core = object.__new__(EngineCoreProc)
    core._checkpoint_state = CheckpointState.RUNNING
    core._checkpoint_error = None
    core._kv_cache_reclaimable = False
    core.model_executor = MagicMock()
    core.model_executor.is_sleeping = False
    core.scheduler = MagicMock()
    core.scheduler.pause_state = PauseState.UNPAUSED
    core.reject_new_requests = False
    return core


def test_resume_rejected_while_executor_asleep():
    """A resume while executor memory is offloaded must be refused.

    This is the sleep level 1+/resume sequence from #68595: the refusal
    prevents a later step from touching offloaded (CPU) tensors.
    """
    core = _stub_core()
    core.model_executor.is_sleeping = True

    with pytest.raises(ValueError, match="asleep"):
        core.resume_scheduler()

    core.scheduler.set_pause_state.assert_not_called()


def test_resume_succeeds_when_executor_awake():
    """The normal pause/resume cycle, and `wake_up`'s internal resume of a
    fully awake engine, must keep working."""
    core = _stub_core()
    core.resume_scheduler()

    core.scheduler.set_pause_state.assert_called_once_with(PauseState.UNPAUSED)
    assert core.reject_new_requests is False


def _request_with_engine(engine_client: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine_client))
    )


def test_resume_endpoint_ok_when_engine_accepts():
    from vllm.entrypoints.rl.online.api_router import resume_generation

    engine = MagicMock()
    engine.resume_generation = AsyncMock(return_value=None)
    response = asyncio.run(resume_generation(_request_with_engine(engine)))

    assert response.status_code == HTTPStatus.OK


def test_resume_endpoint_maps_refusal_to_client_error():
    from vllm.entrypoints.rl.online.api_router import resume_generation

    engine = MagicMock()
    engine.resume_generation = MagicMock(
        side_effect=ValueError("Cannot resume generation while the engine is asleep")
    )
    response = asyncio.run(resume_generation(_request_with_engine(engine)))

    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_async_llm_resume_refusal_from_engine_process_is_value_error():
    """In the multiprocess engine the refusal crosses the process boundary as
    `Exception("Call to resume_scheduler method failed: ...")`, which the
    endpoint's `except ValueError` would miss and report as a 500."""
    from vllm.v1.engine.async_llm import AsyncLLM

    llm = object.__new__(AsyncLLM)
    llm.engine_core = MagicMock()
    llm.engine_core.resume_scheduler_async = AsyncMock(
        side_effect=Exception(
            "Call to resume_scheduler method failed: Cannot resume generation "
            "while the engine is asleep; wake up the engine first (e.g. via /wake_up)."
        )
    )
    with pytest.raises(ValueError, match="asleep"):
        asyncio.run(llm.resume_generation())


def test_async_llm_resume_other_engine_failures_unchanged():
    from vllm.v1.engine.async_llm import AsyncLLM

    llm = object.__new__(AsyncLLM)
    llm.engine_core = MagicMock()
    llm.engine_core.resume_scheduler_async = AsyncMock(side_effect=Exception("boom"))
    with pytest.raises(Exception, match="boom") as exc_info:
        asyncio.run(llm.resume_generation())
    assert not isinstance(exc_info.value, ValueError)
