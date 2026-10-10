# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.launcher import (
    _fatal_engine_error,
    serve_http,
    terminate_if_errored,
)
from vllm.v1.engine.exceptions import EngineDeadError


def make_engine(*, errored: bool, running: bool, error=None):
    return SimpleNamespace(
        errored=errored,
        is_running=running,
        dead_error=error,
        vllm_config=SimpleNamespace(shutdown_timeout=0),
        shutdown=MagicMock(),
    )


def make_server():
    server = MagicMock()
    server.serve = AsyncMock()
    server.shutdown = AsyncMock()
    server.should_exit = False
    server.shutdown_requested = False
    server.fatal_engine_error = None
    return server


@pytest.mark.parametrize(
    (
        "shutdown_requested",
        "keep_alive",
        "engine_errored",
        "engine_running",
        "expect_fatal_exit",
    ),
    [
        (False, False, True, False, True),
        (True, False, True, False, False),
        (False, True, True, False, False),
        (False, False, True, True, False),
        (False, False, False, False, False),
        (False, False, False, True, False),
    ],
    ids=[
        "engine-death-initiates-exit",
        "signal-before-engine-death",
        "keep-alive-on-engine-death",
        "errored-but-running",
        "stopped-without-error",
        "healthy-engine",
    ],
)
def test_terminate_if_errored_latches_only_engine_caused_exit(
    shutdown_requested: bool,
    keep_alive: bool,
    engine_errored: bool,
    engine_running: bool,
    expect_fatal_exit: bool,
):
    error = EngineDeadError()
    engine = make_engine(
        errored=engine_errored,
        running=engine_running,
        error=error,
    )
    server = make_server()
    server.shutdown_requested = shutdown_requested

    with patch(
        "vllm.entrypoints.launchers.launcher.envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        keep_alive,
    ):
        terminate_if_errored(server, engine)

    assert server.should_exit is expect_fatal_exit
    assert (server.fatal_engine_error is error) is expect_fatal_exit


def test_terminate_if_errored_preserves_first_fatal_error():
    first_error = EngineDeadError("first")
    second_error = EngineDeadError("second")
    server = make_server()

    terminate_if_errored(
        server, make_engine(errored=True, running=False, error=first_error)
    )
    terminate_if_errored(
        server, make_engine(errored=True, running=False, error=second_error)
    )

    assert server.fatal_engine_error is first_error


def test_terminate_if_errored_falls_back_when_dead_error_is_none():
    server = make_server()

    terminate_if_errored(server, make_engine(errored=True, running=False, error=None))

    assert server.should_exit is True
    assert isinstance(server.fatal_engine_error, RuntimeError)
    assert "without a recorded error" in str(server.fatal_engine_error)


def test_fatal_engine_error_reads_latched_cause():
    server = make_server()
    assert _fatal_engine_error(server) is None

    error = EngineDeadError()
    server.fatal_engine_error = error
    assert _fatal_engine_error(server) is error


async def run_serve_http(engine_client, server, handlers=None):
    app = FastAPI()
    app.state.engine_client = engine_client

    config = MagicMock()
    config.ssl = None
    handlers = {} if handlers is None else handlers

    loop = asyncio.get_running_loop()

    def add_signal_handler(sig, handler):
        handlers[sig] = handler

    with (
        patch(
            "vllm.entrypoints.launchers.launcher.uvicorn.Config", return_value=config
        ),
        patch(
            "vllm.entrypoints.launchers.launcher.NoSignalServer", return_value=server
        ),
        patch.object(loop, "add_signal_handler", side_effect=add_signal_handler),
    ):
        return await serve_http(app, sock=None, port=8000)


@pytest.mark.asyncio
async def test_serve_http_propagates_engine_caused_exit():
    error = EngineDeadError()
    engine = make_engine(errored=True, running=False, error=error)
    server = make_server()

    async def stop_for_engine_death(*args, **kwargs):
        terminate_if_errored(server, engine)

    server.serve.side_effect = stop_for_engine_death
    shutdown = await run_serve_http(engine, server)

    with pytest.raises(EngineDeadError) as exc_info:
        await shutdown
    assert exc_info.value is error


@pytest.mark.asyncio
async def test_signal_after_engine_death_does_not_clear_fatal_exit():
    error = EngineDeadError()
    engine = make_engine(errored=True, running=False, error=error)
    server = make_server()
    handlers: dict[signal.Signals, Callable[[], None]] = {}

    async def engine_death_then_signal(*args, **kwargs):
        # serve() starts only after serve_http() has registered these handlers.
        terminate_if_errored(server, engine)
        handlers[signal.SIGTERM]()
        await asyncio.Event().wait()

    server.serve.side_effect = engine_death_then_signal
    shutdown = await run_serve_http(engine, server, handlers)

    with pytest.raises(EngineDeadError) as exc_info:
        await shutdown
    assert exc_info.value is error
    server.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_signal_before_engine_death_remains_intentional_shutdown():
    error = EngineDeadError()
    engine = make_engine(errored=True, running=False, error=error)
    server = make_server()
    handlers: dict[signal.Signals, Callable[[], None]] = {}

    async def signal_then_engine_death(*args, **kwargs):
        handlers[signal.SIGTERM]()
        terminate_if_errored(server, engine)
        await asyncio.Event().wait()

    server.serve.side_effect = signal_then_engine_death
    shutdown = await run_serve_http(engine, server, handlers)

    await shutdown
    assert server.fatal_engine_error is None
    server.shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_keep_alive_does_not_latch_fatal_exit():
    engine = make_engine(errored=True, running=False, error=EngineDeadError())
    server = make_server()

    async def observe_engine_death(*args, **kwargs):
        terminate_if_errored(server, engine)

    server.serve.side_effect = observe_engine_death
    with patch(
        "vllm.entrypoints.launchers.launcher.envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
        True,
    ):
        shutdown = await run_serve_http(engine, server)

    await shutdown
    assert server.fatal_engine_error is None
