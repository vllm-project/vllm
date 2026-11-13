# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import ClassVar

import regex as re
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.logger import init_logger

logger = init_logger(__name__)


class UnicodeFilterMiddleware:
    """Pure ASGI middleware that strips Unicode tag characters (U+E0020 to
    U+E007F) from request bodies sent to LLM generation endpoints.

    These characters are invisible to humans but are tokenized by language
    models, which can lead to unpredictable responses. The middleware is
    intended to clean up requests from clients that emit unwanted Unicode
    content. Normal Unicode, including emojis, is left untouched. Only POST
    requests whose path matches :attr:`ROUTES_TO_FILTER` are inspected; all
    other traffic passes through unchanged.

    Filtering is performed directly on the UTF-8 bytes of the body, so no
    decode/encode round trip is needed, and both encodings a JSON client can
    use for these characters are handled: the raw UTF-8 sequence (``F3 A0 80
    A0`` to ``F3 A0 81 BF``) and the JSON ``\\u`` escape of the equivalent
    surrogate pair (``\\uDB40\\uDC20`` to ``\\uDB40\\uDC7F``). The common
    (clean) case is ruled out with cheap substring checks that skip the
    regexes entirely.
    """

    ROUTES_TO_FILTER: ClassVar[frozenset[str]] = frozenset(
        {
            "/v1/chat/completions",
            "/v1/chat/completions/batch",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
            "/cohere/v2/chat",
        }
    )

    # UTF-8 encodings of U+E0020..U+E007F:
    #   U+E0020..U+E003F -> F3 A0 80 [A0-BF]
    #   U+E0040..U+E007F -> F3 A0 81 [80-BF]
    TAGS_BLOCK_PATTERN: ClassVar[re.Pattern[bytes]] = re.compile(
        b"\xf3\xa0(?:\x80[\xa0-\xbf]|\x81[\x80-\xbf])"
    )
    # JSON-escaped UTF-16 surrogate pairs for the same range, in either hex
    # case. Only an odd run of backslashes starts an escape (``\\uDB40`` is a
    # literal backslash followed by text), so group 1 captures and keeps any
    # preceding escaped backslashes.
    TAGS_BLOCK_ESCAPED_PATTERN: ClassVar[re.Pattern[bytes]] = re.compile(
        rb"(?<!\\)((?:\\\\)*)\\u[dD][bB]40\\u[dD][cC][2-7][0-9a-fA-F]"
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @classmethod
    def strip_tag_characters(cls, body: bytes) -> bytes:
        if b"\xf3\xa0\x80" in body or b"\xf3\xa0\x81" in body:
            body = cls.TAGS_BLOCK_PATTERN.sub(b"", body)
        if b"\\u" in body:
            body = cls.TAGS_BLOCK_ESCAPED_PATTERN.sub(rb"\1", body)
        return body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        path = scope["path"].removeprefix(root_path)
        if path not in self.ROUTES_TO_FILTER:
            await self.app(scope, receive, send)
            return

        # Buffer the whole body so it can be rewritten before the downstream
        # handlers see it.
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":

                async def disconnect_receive() -> Message:
                    return {"type": "http.disconnect"}

                await self.app(scope, disconnect_receive, send)
                return
            body.extend(message.get("body", b""))
            more_body = message.get("more_body", False)

        body_bytes = self.strip_tag_characters(bytes(body))
        if len(body_bytes) != len(body):
            logger.debug(
                "UnicodeFilterMiddleware stripped tag characters from %s", path
            )

        # Keep Content-Length consistent with the (possibly rewritten) body.
        new_headers = [
            (k, v)
            for k, v in scope.get("headers", [])
            if k.lower() != b"content-length"
        ]
        new_headers.append((b"content-length", str(len(body_bytes)).encode()))
        scope = {**scope, "headers": new_headers}

        body_sent = False

        async def replay_receive() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": body_bytes,
                    "more_body": False,
                }
            # The body has been handed over; defer to the server's receive()
            # so later calls block until the client really disconnects, as
            # ``with_cancellation`` and ``StreamingResponse`` expect.
            return await receive()

        await self.app(scope, replay_receive, send)
