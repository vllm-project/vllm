# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from collections.abc import Awaitable

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)


class XRequestIdMiddleware:
    """
    Middleware the set's the X-Request-Id header for each response
    to a random uuid4 (hex) value if the header isn't already
    present in the request, otherwise use the provided request id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        # Extract the request headers.
        request_headers = Headers(scope=scope)

        async def send_with_request_id(message: Message) -> None:
            """
            Custom send function to mutate the response headers
            and append X-Request-Id to it.
            """
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(raw=message["headers"])
                request_id = request_headers.get("X-Request-Id", uuid.uuid4().hex)
                response_headers.append("X-Request-Id", request_id)
            await send(message)

        return self.app(scope, receive, send_with_request_id)
