# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import requests

import vllm.envs as envs
from vllm.connections import (
    HTTPConnection,
    HTTPResponseSizeExceededError,
    MediaRequestDownloadBudget,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

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
    def __init__(self, chunks: list[bytes | Exception]) -> None:
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
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def raise_for_status(self) -> None:
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _AsyncContent:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
        self._chunks = chunks
        self.iterated = 0

    async def iter_chunked(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            self.iterated += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class _AsyncResponse:
    def __init__(self, chunks: list[bytes | Exception]) -> None:
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


def test_request_download_budget_counts_across_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 1)
    budget = MediaRequestDownloadBudget()
    connection = HTTPConnection()
    first = _SyncResponse([b"a" * (3 * _ONE_MIB // 4)])
    second = _SyncResponse([b"b" * (_ONE_MIB // 2)])

    with patch.object(connection, "get_response", side_effect=[first, second]):
        connection.get_bytes("http://example.com/one", request_budget=budget)
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB"):
            connection.get_bytes("http://example.com/two", request_budget=budget)

    assert budget.used == 3 * _ONE_MIB // 4
    assert second.iterated == 1


@pytest.mark.asyncio
async def test_async_request_download_budget_counts_across_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 1)
    budget = MediaRequestDownloadBudget()
    connection = HTTPConnection()
    first = _AsyncResponse([b"a" * (3 * _ONE_MIB // 4)])
    second = _AsyncResponse([b"b" * (_ONE_MIB // 2)])

    with patch.object(
        connection, "get_async_response", new=AsyncMock(side_effect=[first, second])
    ):
        await connection.async_get_bytes(
            "http://example.com/one", request_budget=budget
        )
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB"):
            await connection.async_get_bytes(
                "http://example.com/two", request_budget=budget
            )

    assert budget.used == 3 * _ONE_MIB // 4
    assert second.content.iterated == 1


def test_request_download_budget_counts_retried_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 1)
    monkeypatch.setattr("vllm.connections.time.sleep", lambda _: None)
    budget = MediaRequestDownloadBudget()
    connection = HTTPConnection()
    body = b"a" * (3 * _ONE_MIB // 4)
    interrupted = _SyncResponse([body[: _ONE_MIB // 2], requests.ConnectionError()])

    with (
        patch.object(
            connection,
            "get_response",
            side_effect=[interrupted, _SyncResponse([body])],
        ),
        pytest.raises(ValueError, match="including retried attempts"),
    ):
        connection.get_bytes("http://example.com/media", request_budget=budget)


@pytest.mark.asyncio
async def test_async_request_download_budget_counts_retried_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 1)
    monkeypatch.setattr("vllm.connections.asyncio.sleep", AsyncMock())
    budget = MediaRequestDownloadBudget()
    connection = HTTPConnection()
    body = b"a" * (3 * _ONE_MIB // 4)
    interrupted = _AsyncResponse(
        [body[: _ONE_MIB // 2], aiohttp.ClientConnectionError()]
    )

    with (
        patch.object(
            connection,
            "get_async_response",
            new=AsyncMock(side_effect=[interrupted, _AsyncResponse([body])]),
        ),
        pytest.raises(ValueError, match="including retried attempts"),
    ):
        await connection.async_get_bytes(
            "http://example.com/media", request_budget=budget
        )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("always_incomplete", [False, True], ids=["recover", "exhaust"])
async def test_get_bytes_retries_incomplete_response(
    monkeypatch: pytest.MonkeyPatch,
    local_media_server: str,
    use_async: bool,
    always_incomplete: bool,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MEDIA_FETCH_MAX_RETRIES", 2)
    monkeypatch.setattr("vllm.connections.time.sleep", lambda _: None)
    monkeypatch.setattr("vllm.connections.asyncio.sleep", AsyncMock())
    attempts = 0
    serve_complete_response = _MediaHandler.do_GET

    def serve_response(handler: _MediaHandler) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1 or always_incomplete:
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(_SMALL_BODY)))
            handler.end_headers()
            handler.wfile.write(_SMALL_BODY[:_CHUNK_SIZE])
            handler.close_connection = True
        else:
            serve_complete_response(handler)

    monkeypatch.setattr(_MediaHandler, "do_GET", serve_response)
    connection = HTTPConnection()
    url = f"{local_media_server}/small"

    async def download() -> bytes:
        if use_async:
            return await connection.async_get_bytes(url, timeout=2)
        return connection.get_bytes(url, timeout=2)

    try:
        if always_incomplete:
            error_type = (
                aiohttp.ClientPayloadError
                if use_async
                else requests.exceptions.ChunkedEncodingError
            )
            with pytest.raises(error_type):
                await download()
        else:
            assert await download() == _SMALL_BODY
        assert attempts == 2
    finally:
        if connection._async_client is not None:
            await connection._async_client.close()
        if connection._sync_client is not None:
            connection._sync_client.close()
