# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""How the AsyncLLM output handler reports an engine core that is already gone.

shutdown() stops the engine core before it cancels the output handler, and the
launcher runs shutdown() in an executor thread, so the event loop keeps going in
between. A read in flight during that window finds the core gone and raises
EngineDeadError, which is the shutdown working rather than failing.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

pytestmark = pytest.mark.cpu_test

_ASYNC_LLM_LOGGER = "vllm.v1.engine.async_llm"


def _make_engine(shutting_down: bool) -> SimpleNamespace:
    """The attributes _run_output_handler reads, with a core that is gone."""

    async def get_output_async():
        raise EngineDeadError()

    return SimpleNamespace(
        output_handler=None,
        _shutting_down_ref=[shutting_down],
        engine_core=SimpleNamespace(get_output_async=get_output_async),
        output_processor=MagicMock(),
        log_stats=False,
        logger_manager=None,
        renderer=SimpleNamespace(mm_processor_cache=None),
    )


async def _drain(engine: SimpleNamespace) -> None:
    AsyncLLM._run_output_handler(engine)
    await engine.output_handler


@pytest.mark.asyncio
async def test_requested_stop_is_not_reported_as_a_failure(caplog):
    engine = _make_engine(shutting_down=True)

    with caplog.at_level(logging.INFO, logger=_ASYNC_LLM_LOGGER):
        await _drain(engine)

    assert "Engine core stopped before the output handler" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    engine.output_processor.propagate_error.assert_called_once()


@pytest.mark.asyncio
async def test_death_outside_shutdown_is_still_reported(caplog):
    engine = _make_engine(shutting_down=False)

    with caplog.at_level(logging.INFO, logger=_ASYNC_LLM_LOGGER):
        await _drain(engine)

    assert [r for r in caplog.records if r.levelno >= logging.ERROR]
    engine.output_processor.propagate_error.assert_called_once()
