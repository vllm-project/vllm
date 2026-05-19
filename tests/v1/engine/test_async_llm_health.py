# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError, EngineUnhealthyError


@pytest.mark.asyncio
async def test_check_ready_ok():
    engine = object.__new__(AsyncLLM)
    engine.output_processor = SimpleNamespace(has_unfinished_requests=lambda: False)
    with patch.object(
        type(engine),
        "errored",
        new_callable=lambda: property(lambda self: False),
    ):
        await engine.check_ready()


@pytest.mark.asyncio
async def test_check_ready_dead_engine():
    engine = object.__new__(AsyncLLM)
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


@pytest.mark.asyncio
async def test_check_ready_stalled_request():
    engine = object.__new__(AsyncLLM)
    engine.output_processor = SimpleNamespace(has_unfinished_requests=lambda: True)
    engine._last_request_at = (
        time.monotonic() - envs.VLLM_READY_CHECK_IDLE_TIMEOUT_S - 1
    )
    engine._last_progress_at_ref = [0.0]
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
async def test_check_ready_stalled_after_progress():
    engine = object.__new__(AsyncLLM)
    engine.output_processor = SimpleNamespace(has_unfinished_requests=lambda: True)
    engine._last_request_at = time.monotonic()
    engine._last_progress_at_ref = [
        time.monotonic() - envs.VLLM_READY_CHECK_PROGRESS_TIMEOUT_S - 1
    ]
    with (
        patch.object(
            type(engine),
            "errored",
            new_callable=lambda: property(lambda self: False),
        ),
        pytest.raises(EngineUnhealthyError),
    ):
        await engine.check_ready()
