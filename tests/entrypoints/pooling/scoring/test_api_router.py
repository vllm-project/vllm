# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest

from vllm.entrypoints.pooling.scoring.api_router import (
    do_rerank_v1,
    do_rerank_v2,
)


@pytest.mark.asyncio
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("alias", [do_rerank_v1, do_rerank_v2])
async def test_rerank_alias_uses_one_disconnect_listener(alias):
    disconnect = asyncio.Event()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def handler(request, raw_request):
        handler_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            handler_cancelled.set()
            raise

    state = SimpleNamespace(
        enable_server_load_tracking=True,
        server_load_metrics=0,
        serving_scores=handler,
    )
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=state),
        receive=receive,
    )

    route_task = asyncio.create_task(alias(None, raw_request))
    await asyncio.wait_for(handler_started.wait(), timeout=1)
    disconnect.set()
    assert await asyncio.wait_for(route_task, timeout=1) is None
    await asyncio.wait_for(handler_cancelled.wait(), timeout=1)

    assert receive_calls == 1
    assert state.server_load_metrics == 0
