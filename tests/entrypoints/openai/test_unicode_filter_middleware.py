# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``UnicodeFilterMiddleware``.

These tests exercise the middleware in isolation against a tiny FastAPI
echo app, so they do not require a vLLM engine.
"""

import httpx
import pytest
from fastapi import FastAPI, Request
from vllm.entrypoints.serve.utils.server_utils import UnicodeFilterMiddleware

# A few characters from the Unicode "Tags" block (U+E0020 - U+E007F)
# that the middleware is supposed to strip.
TAG_CHARS = "".join(chr(c) for c in (0xE0020, 0xE0024, 0xE0041, 0xE007F))


def _build_app() -> FastAPI:
    """A FastAPI app whose routes echo the bytes they received."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict:
        body = await request.body()
        return {"body": body.decode("utf-8")}

    @app.post("/v1/completions")
    async def completions(request: Request) -> dict:
        body = await request.body()
        return {"body": body.decode("utf-8")}

    @app.post("/health")
    async def health(request: Request) -> dict:
        body = await request.body()
        return {"body": body.decode("utf-8")}

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
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
async def test_strips_tag_block_characters(path: str) -> None:
    app = _build_app()
    payload = f"Hello{TAG_CHARS}World"

    response = await _post(app, path, payload)

    assert response.status_code == 200
    assert response.json()["body"] == "HelloWorld"


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
    payload = '{"model": "test", "prompt": "Hello, world!"}'

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


def test_tags_block_pattern_covers_full_block() -> None:
    """The bytes pattern must match every codepoint in the Tags block and
    nothing immediately outside it."""
    pattern = UnicodeFilterMiddleware.TAGS_BLOCK_PATTERN
    # Every codepoint in U+E0020..U+E007F encodes to a 4-byte UTF-8
    # sequence that the pattern should fully match.
    for cp in range(0xE0020, 0xE0080):
        encoded = chr(cp).encode("utf-8")
        match = pattern.fullmatch(encoded)
        assert match is not None, hex(cp)
    # Just outside the range, plus ASCII and an emoji: must not match.
    for cp in (0xE001F, 0xE0080, ord("a"), ord("😀")):
        encoded = chr(cp).encode("utf-8")
        assert pattern.match(encoded) is None, hex(cp)
