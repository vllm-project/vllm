# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import pytest

import vllm.envs as envs
from vllm.connections import HTTPConnection, HTTPResponseSizeExceededError

_ONE_MIB = 1024 * 1024
_SMALL_BODY = b"a" * (_ONE_MIB // 2)
_LARGE_BODY = b"b" * (2 * _ONE_MIB)
_CHUNK_SIZE = 64 * 1024


class _MediaHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = _LARGE_BODY if "large" in self.path else _SMALL_BODY
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        if self.path != "/chunked-large":
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        try:
            for offset in range(0, len(body), _CHUNK_SIZE):
                self.wfile.write(body[offset : offset + _CHUNK_SIZE])
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def local_media_server() -> Generator[str, None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MediaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class _SyncResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.headers: dict[str, str] = {}
        self._chunks = chunks
        self.content_accessed = False
        self.iterated = 0

    @property
    def content(self) -> bytes:
        self.content_accessed = True
        raise AssertionError("bounded reads must not access response.content")

    def iter_content(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            self.iterated += 1
            yield chunk

    def raise_for_status(self) -> None:
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _AsyncContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.iterated = 0

    async def iter_chunked(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            self.iterated += 1
            yield chunk


class _AsyncResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _AsyncContent(chunks)
        self.content_length = None
        self.read = AsyncMock(
            side_effect=AssertionError("bounded reads must not call response.read()")
        )

    def raise_for_status(self) -> None:
        return

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


def test_get_bytes_with_limit_streams_instead_of_materializing() -> None:
    response = _SyncResponse([b"abcd", b"efgh", b"ijkl"])
    connection = HTTPConnection()

    with (
        patch.object(
            connection,
            "get_response",
            return_value=response,
        ) as get_response,
        pytest.raises(HTTPResponseSizeExceededError, match="maximum size"),
    ):
        connection.get_bytes("http://example.com/media", max_bytes=4)

    get_response.assert_called_once_with(
        "http://example.com/media",
        stream=True,
        timeout=None,
        allow_redirects=True,
    )
    assert response.content_accessed is False
    assert response.iterated == 2


@pytest.mark.asyncio
async def test_async_get_bytes_with_limit_streams_instead_of_materializing() -> None:
    response = _AsyncResponse([b"abcd", b"efgh", b"ijkl"])
    connection = HTTPConnection()

    with (
        patch.object(
            connection,
            "get_async_response",
            new=AsyncMock(return_value=response),
        ),
        pytest.raises(HTTPResponseSizeExceededError, match="maximum size"),
    ):
        await connection.async_get_bytes("http://example.com/media", max_bytes=4)

    response.read.assert_not_awaited()
    assert response.content.iterated == 2


def test_get_bytes_rejects_content_length_over_limit(
    monkeypatch: pytest.MonkeyPatch, local_media_server: str
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    connection = HTTPConnection(reuse_client=False)

    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB"):
        connection.get_bytes(f"{local_media_server}/large")


@pytest.mark.asyncio
async def test_async_get_bytes_rejects_stream_over_limit_without_content_length(
    monkeypatch: pytest.MonkeyPatch, local_media_server: str
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    connection = HTTPConnection(reuse_client=False)

    try:
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB"):
            await connection.async_get_bytes(f"{local_media_server}/chunked-large")
    finally:
        if connection._async_client is not None:
            await connection._async_client.close()


@pytest.mark.asyncio
async def test_get_bytes_allows_body_within_limit(
    monkeypatch: pytest.MonkeyPatch, local_media_server: str
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    connection = HTTPConnection(reuse_client=False)

    assert connection.get_bytes(f"{local_media_server}/small") == _SMALL_BODY

    try:
        actual = await connection.async_get_bytes(f"{local_media_server}/small")
        assert actual == _SMALL_BODY
    finally:
        if connection._async_client is not None:
            await connection._async_client.close()
