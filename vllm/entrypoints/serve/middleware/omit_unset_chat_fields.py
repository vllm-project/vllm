# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omit unset vLLM extension fields from non-streaming Chat Completions.

Enable with ``vllm serve MODEL --middleware`` followed by this import path:
``vllm.entrypoints.serve.middleware.omit_unset_chat_fields.OmitUnsetChatFieldsMiddleware``.

Omitted fields are absent from OpenAI SDK objects instead of reading as None.
This buffers matching JSON responses and parses/encodes them again. If using
compression middleware, register it after this middleware so compression runs
after response filtering. Already compressed responses are left unchanged.
"""

import json

from starlette._utils import get_route_path
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_EXTENSION_FIELDS = (
    "prompt_logprobs",
    "prompt_token_ids",
    "prompt_text",
    "kv_transfer_params",
    "ec_transfer_params",
    "metrics",
)
_CHAT_PATHS = {"/v1/chat/completions", "/v1/chat/completions/batch"}


class OmitUnsetChatFieldsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or get_route_path(scope) not in _CHAT_PATHS
        ):
            await self.app(scope, receive, send)
            return

        start: Message | None = None
        body = bytearray()

        async def filter_response(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                headers = Headers(raw=message["headers"])
                if (
                    message["status"] == 200
                    and headers.get("content-type", "").split(";", 1)[0]
                    == "application/json"
                    and "content-encoding" not in headers
                ):
                    start = message
                    return
            elif message["type"] == "http.response.body" and start is not None:
                body.extend(message.get("body", b""))
                if message.get("more_body", False):
                    return
                content = json.loads(body)
                for field in _EXTENSION_FIELDS:
                    if content.get(field) is None:
                        content.pop(field, None)
                encoded = json.dumps(
                    content, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode("utf-8")
                MutableHeaders(scope=start)["content-length"] = str(len(encoded))
                await send(start)
                message = {**message, "body": encoded}
                start = None
            await send(message)

        await self.app(scope, receive, filter_response)
