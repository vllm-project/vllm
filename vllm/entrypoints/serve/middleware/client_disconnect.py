# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Awaitable

from prometheus_client import Counter
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_client_disconnects = Counter(
    name="vllm:http_requests_client_disconnected_total",
    documentation=(
        "Number of HTTP requests whose client disconnected before the "
        "response was complete."
    ),
)


class ClientDisconnectMetricsMiddleware:
    """Pure ASGI middleware that counts requests abandoned by the client.

    A disconnect is only observed while something is waiting on it, which is
    the case for streaming responses and `with_cancellation` handlers.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        # Servers also report a disconnect once the response is complete.
        finished = False

        def record_disconnect() -> None:
            nonlocal finished
            if not finished:
                finished = True
                _client_disconnects.inc()

        async def receive_wrapper() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                record_disconnect()
            return message

        async def send_wrapper(message: Message) -> None:
            nonlocal finished
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                finished = True
            try:
                await send(message)
            except OSError:
                # ASGI spec >= 2.4 servers raise on send to a closed connection.
                record_disconnect()
                raise

        return self.app(scope, receive_wrapper, send_wrapper)
