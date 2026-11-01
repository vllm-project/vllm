# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection


def _output(text: str = "hi", token_ids: list[int] | None = None):
    return SimpleNamespace(
        prompt_token_ids=[1],
        outputs=[SimpleNamespace(text=text, token_ids=token_ids or [2])],
    )


async def _yield_one_then_hang():
    yield _output()
    await asyncio.Event().wait()
    yield


async def _never_finishes():
    await asyncio.Event().wait()
    yield


def _make_connection(generate_agen):
    engine_client = SimpleNamespace(
        generate=Mock(return_value=generate_agen),
        abort=AsyncMock(),
    )
    serving = SimpleNamespace(
        engine_client=engine_client,
        model_cls=SimpleNamespace(realtime_max_tokens=16),
    )
    websocket = SimpleNamespace(send_text=AsyncMock())
    conn = RealtimeConnection(websocket, serving)
    conn._is_connected = True
    return conn, engine_client


@pytest.mark.asyncio
async def test_disconnect_mid_stream_aborts_engine_request():
    conn, engine_client = _make_connection(_yield_one_then_hang())
    conn._is_connected = False

    await conn._run_generation(
        streaming_input_gen=None,
        input_stream=asyncio.Queue(),
    )

    engine_client.abort.assert_awaited()
    aborted_id = engine_client.abort.await_args.args[0]
    assert aborted_id.startswith(f"rt-{conn.connection_id}-")
    generate_id = engine_client.generate.call_args.kwargs["request_id"]
    assert aborted_id == generate_id


@pytest.mark.asyncio
async def test_cleanup_cancel_aborts_engine_request():
    conn, engine_client = _make_connection(_never_finishes())
    conn.generation_task = asyncio.create_task(
        conn._run_generation(
            streaming_input_gen=None,
            input_stream=asyncio.Queue(),
        )
    )
    await asyncio.sleep(0)

    await conn.cleanup()

    engine_client.abort.assert_awaited()
    aborted_id = engine_client.abort.await_args.args[0]
    generate_id = engine_client.generate.call_args.kwargs["request_id"]
    assert aborted_id == generate_id
