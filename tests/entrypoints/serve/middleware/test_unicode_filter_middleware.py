# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``UnicodeFilterMiddleware``.

These tests exercise the middleware in isolation against a tiny FastAPI echo
app, so they do not require a vLLM engine.
"""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from vllm.entrypoints.serve.middleware.unicode_filter import (
    UnicodeFilterMiddleware,
)
from vllm.entrypoints.serve.utils.api_utils import with_cancellation

# A few characters from the Unicode "Tags" block (U+E0020 - U+E007F)
# that the middleware is supposed to strip.
TAG_CHARS = "".join(chr(c) for c in (0xE0020, 0xE0024, 0xE0041, 0xE007F))
# The same characters as JSON "\\u" escapes, which is how json.dumps emits
# them by default (ensure_ascii=True).
TAG_CHARS_ESCAPED = json.dumps(TAG_CHARS)[1:-1]
ROUTES = sorted(UnicodeFilterMiddleware.ROUTES_TO_FILTER)


def _build_app() -> FastAPI:
    """A FastAPI app whose routes echo the bytes they received."""
    app = FastAPI()

    async def echo(request: Request) -> dict:
        body = await request.body()
        return {"body": body.decode("utf-8")}

    for route in ROUTES:
        app.post(route)(echo)
    app.post("/health")(echo)

    @app.get("/v1/chat/completions")
    async def chat_completions_get() -> dict:
        return {"ok": True}

    app.add_middleware(UnicodeFilterMiddleware)
    return app


async def _post(app: FastAPI, path: str, payload: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        return await client.post(
            path,
            content=payload.encode("utf-8"),
            headers={"content-type": "application/json"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ROUTES)
async def test_strips_tag_block_characters(path: str) -> None:
    app = _build_app()
    payload = f"Hello{TAG_CHARS}World"

    response = await _post(app, path, payload)

    assert response.status_code == 200
    assert response.json()["body"] == "HelloWorld"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "escaped",
    [TAG_CHARS_ESCAPED, TAG_CHARS_ESCAPED.upper().replace("\\U", "\\u")],
    ids=["lowercase-hex", "uppercase-hex"],
)
async def test_strips_json_escaped_tag_characters(escaped: str) -> None:
    app = _build_app()
    payload = f'{{"prompt": "Hello{escaped}World"}}'

    response = await _post(app, "/v1/completions", payload)

    assert response.status_code == 200
    assert json.loads(response.json()["body"]) == {"prompt": "HelloWorld"}


@pytest.mark.asyncio
async def test_mixed_raw_and_escaped_tag_characters_are_stripped() -> None:
    app = _build_app()
    payload = f'{{"prompt": "a{TAG_CHARS}b{TAG_CHARS_ESCAPED}c"}}'

    response = await _post(app, "/v1/completions", payload)

    assert json.loads(response.json()["body"]) == {"prompt": "abc"}


@pytest.mark.asyncio
async def test_escaped_backslash_is_not_treated_as_an_escape() -> None:
    """``\\\\udb40\\\\udc41`` in JSON text is a literal backslash followed by
    plain text, not a tag character, and must be left untouched."""
    app = _build_app()
    payload = '{"prompt": "a\\\\udb40\\\\udc41b"}'

    response = await _post(app, "/v1/completions", payload)

    assert response.json()["body"] == payload
    assert json.loads(response.json()["body"]) == {"prompt": "a\\udb40\\udc41b"}


@pytest.mark.asyncio
async def test_emojis_preserved_on_filtered_routes() -> None:
    app = _build_app()
    payload = f"Hello! 😀🌟🎉 {TAG_CHARS} World"

    response = await _post(app, "/v1/chat/completions", payload)

    assert response.status_code == 200
    assert response.json()["body"] == "Hello! 😀🌟🎉  World"


@pytest.mark.asyncio
async def test_other_routes_pass_through_unchanged() -> None:
    app = _build_app()
    payload = f"Hello{TAG_CHARS}World"

    response = await _post(app, "/health", payload)

    assert response.status_code == 200
    # /health is not in ROUTES_TO_FILTER, so the body is preserved verbatim.
    assert response.json()["body"] == payload


@pytest.mark.asyncio
async def test_non_post_methods_pass_through() -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/v1/chat/completions")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_clean_payload_unchanged() -> None:
    app = _build_app()
    payload = '{"model": "test", "prompt": "Hello, world! \\u00e9 \\ud83d\\ude00"}'

    response = await _post(app, "/v1/completions", payload)

    assert response.status_code == 200
    assert response.json()["body"] == payload


@pytest.mark.asyncio
async def test_content_length_updated_when_body_shrinks() -> None:
    """The downstream handler's ``await request.body()`` would hang or
    truncate if Content-Length were left stale after filtering."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def echo(request: Request) -> dict:
        body = await request.body()
        return {
            "body": body.decode("utf-8"),
            "content_length": request.headers.get("content-length"),
        }

    app.add_middleware(UnicodeFilterMiddleware)

    payload = f"abc{TAG_CHARS}xyz"
    response = await _post(app, "/v1/chat/completions", payload)
    data = response.json()
    assert data["body"] == "abcxyz"
    assert data["content_length"] == "6"


class _CompletionRequest(BaseModel):
    prompt: str


@pytest.mark.asyncio
async def test_handler_with_cancellation_still_completes() -> None:
    """VLLM's generation handlers are wrapped in ``with_cancellation``, which
    polls ``receive()`` for ``http.disconnect`` while the handler runs. The
    middleware's replayed ``receive()`` must not report a disconnect before the
    client actually goes away, or every filtered request would be cancelled."""
    app = FastAPI()

    @app.post("/v1/completions")
    @with_cancellation
    async def completions(request: _CompletionRequest, raw_request: Request) -> dict:
        await asyncio.sleep(0.05)
        return {"prompt": request.prompt}

    app.add_middleware(UnicodeFilterMiddleware)

    response = await _post(
        app, "/v1/completions", json.dumps({"prompt": f"Hello{TAG_CHARS}World"})
    )

    assert response.status_code == 200
    assert response.json() == {"prompt": "HelloWorld"}


@pytest.mark.asyncio
async def test_streaming_response_is_not_cut_short() -> None:
    """``StreamingResponse`` listens on ``receive()`` for a disconnect while
    streaming, so a premature disconnect would truncate the stream."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def stream(request: Request) -> StreamingResponse:
        await request.body()

        async def chunks():
            for i in range(3):
                await asyncio.sleep(0.02)
                yield f"chunk{i}\n".encode()

        return StreamingResponse(chunks(), media_type="text/plain")

    app.add_middleware(UnicodeFilterMiddleware)

    response = await _post(app, "/v1/chat/completions", f"Hello{TAG_CHARS}")

    assert response.status_code == 200
    assert response.text == "chunk0\nchunk1\nchunk2\n"


def test_tags_block_patterns_cover_full_block() -> None:
    """Both patterns must match every codepoint in the Tags block and
    nothing immediately outside it."""
    raw = UnicodeFilterMiddleware.TAGS_BLOCK_PATTERN
    escaped = UnicodeFilterMiddleware.TAGS_BLOCK_ESCAPED_PATTERN
    for cp in range(0xE0020, 0xE0080):
        assert raw.fullmatch(chr(cp).encode("utf-8")) is not None, hex(cp)
        assert escaped.search(json.dumps(chr(cp)).encode()) is not None, hex(cp)
    # Just outside the range, plus ASCII and an emoji: must not match.
    for cp in (0xE001F, 0xE0080, ord("a"), ord("😀")):
        assert raw.match(chr(cp).encode("utf-8")) is None, hex(cp)
        assert escaped.search(json.dumps(chr(cp)).encode()) is None, hex(cp)
