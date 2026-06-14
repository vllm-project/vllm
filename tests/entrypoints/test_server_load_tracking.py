# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server load accounting must stay non-negative when a client disconnects.

Regression for the Anthropic ``/v1/messages/count_tokens`` route, which nested
``@load_aware_call`` and ``@with_cancellation`` in the wrong order. On a
disconnect ``listen_for_disconnect`` decremented ``server_load_metrics`` once
and ``with_cancellation`` returned ``None``, so the outer ``load_aware_call``
took its non-response branch and decremented a second time, driving the metric
negative.
"""

import asyncio
import types

import pytest
from fastapi.responses import JSONResponse

from vllm.entrypoints.serve.utils.api_utils import load_aware_call, with_cancellation


def _disconnecting_request():
    state = types.SimpleNamespace(
        enable_server_load_tracking=True,
        server_load_metrics=0,
    )

    async def receive():
        return {"type": "http.disconnect"}

    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=state), receive=receive
    )


async def _slow_handler(request, raw_request):
    await asyncio.sleep(5)
    return JSONResponse(content={})


@pytest.mark.asyncio
async def test_server_load_non_negative_on_disconnect_correct_order():
    # @with_cancellation (outer) + @load_aware_call (inner): the order used by
    # every other load-aware route, and now by count_tokens.
    wrapped = with_cancellation(load_aware_call(_slow_handler))
    raw_request = _disconnecting_request()
    result = await wrapped(None, raw_request)
    assert result is None  # handler cancelled by the disconnect
    assert raw_request.app.state.server_load_metrics == 0


@pytest.mark.asyncio
async def test_server_load_double_decrement_on_reversed_order():
    # The reversed nesting count_tokens used to have double-counts the
    # disconnect and leaves the metric negative.
    wrapped = load_aware_call(with_cancellation(_slow_handler))
    raw_request = _disconnecting_request()
    result = await wrapped(None, raw_request)
    assert result is None
    assert raw_request.app.state.server_load_metrics == -1
