# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.launcher import serve_http
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intentional_shutdown", "engine_errored", "expect_engine_error"),
    [
        (False, True, True),
        (True, False, False),
        (True, True, False),
        (False, False, False),
    ],
    ids=[
        "fatal-engine-death",
        "intentional-signal",
        "signal-wins-engine-death-race",
        "healthy-server-stop",
    ],
)
async def test_serve_http_preserves_fatal_engine_exit_status(
    intentional_shutdown: bool,
    engine_errored: bool,
    expect_engine_error: bool,
):
    """Only an unexpected EngineCore death must fail the serving process."""
    app = FastAPI()
    app.state.engine_client = SimpleNamespace(
        errored=engine_errored,
        is_running=not engine_errored,
        dead_error=EngineDeadError(),
    )

    server = MagicMock()
    server.serve = AsyncMock()

    config = MagicMock()
    config.ssl = None

    async def wait_for_signal() -> None:
        await asyncio.Event().wait()

    shutdown_event = MagicMock()
    shutdown_event.is_set.return_value = intentional_shutdown
    shutdown_event.wait = AsyncMock(side_effect=wait_for_signal)

    with (
        patch(
            "vllm.entrypoints.launchers.launcher.uvicorn.Config", return_value=config
        ),
        patch(
            "vllm.entrypoints.launchers.launcher.NoSignalServer", return_value=server
        ),
        patch(
            "vllm.entrypoints.launchers.launcher.asyncio.Event",
            return_value=shutdown_event,
        ),
    ):
        shutdown = await serve_http(app, sock=None, port=8000)

    if expect_engine_error:
        with pytest.raises(EngineDeadError):
            await shutdown
    else:
        await shutdown
