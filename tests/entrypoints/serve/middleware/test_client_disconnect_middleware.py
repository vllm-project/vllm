# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from prometheus_client import REGISTRY

from vllm.entrypoints.serve.middleware.client_disconnect import (
    ClientDisconnectMetricsMiddleware,
)

pytestmark = pytest.mark.cpu_test

BODY_CHUNK = {"type": "http.response.body", "body": b"x", "more_body": True}
BODY_END = {"type": "http.response.body", "body": b""}


def _num_disconnects() -> float:
    name = "vllm:http_requests_client_disconnected_total"
    return REGISTRY.get_sample_value(name) or 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("responses", "send_fails", "expected"),
    [
        pytest.param([], False, 1, id="before_response"),
        pytest.param([BODY_CHUNK], False, 1, id="mid_stream"),
        pytest.param([BODY_CHUNK], True, 1, id="send_to_closed_connection"),
        pytest.param([BODY_CHUNK, BODY_END], False, 0, id="after_response"),
    ],
)
async def test_counts_disconnects_before_response_completes(
    responses, send_fails, expected
):
    """A disconnect is a client cancellation only while the response is unfinished,
    and is counted once however many times the server reports it."""

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if send_fails:
            raise OSError

    async def app(scope, receive, send):
        try:
            for message in responses:
                await send(message)
        except OSError:
            pass
        await receive()
        await receive()

    before = _num_disconnects()
    middleware = ClientDisconnectMetricsMiddleware(app)
    await middleware({"type": "http"}, receive, send)
    assert _num_disconnects() - before == expected
