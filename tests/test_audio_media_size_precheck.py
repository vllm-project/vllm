# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.connections import HTTPConnection
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import AudioMediaIO, MediaConnector
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _SyncResponse:
    def __init__(self, chunks: list[bytes], content_length: int | None = None):
        self.headers = (
            {} if content_length is None else {"Content-Length": str(content_length)}
        )
        self._chunks = chunks
        self.iterated = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert chunk_size == 64 * KiB_bytes
        for chunk in self._chunks:
            self.iterated += 1
            yield chunk


class _AsyncContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.iterated = 0

    async def iter_chunked(self, chunk_size: int) -> AsyncIterator[bytes]:
        assert chunk_size == 64 * KiB_bytes
        for chunk in self._chunks:
            self.iterated += 1
            yield chunk


class _AsyncResponse:
    def __init__(self, chunks: list[bytes], content_length: int | None = None):
        self.content_length = content_length
        self.content = _AsyncContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None


def test_audio_base64_rejects_before_decode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)
    max_encoded_chars = 4 * ((MiB_bytes + 2) // 3)

    with (
        patch("vllm.multimodal.media.audio.pybase64.b64decode") as decode,
        pytest.raises(VLLMValidationError, match="Maximum file size exceeded"),
    ):
        AudioMediaIO().load_base64("audio/wav", "A" * (max_encoded_chars + 1))

    decode.assert_not_called()


def test_audio_load_bytes_rejects_oversized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        AudioMediaIO().load_bytes(b"\x00" * (MiB_bytes + 1))


def test_audio_load_file_rejects_oversized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)

    oversized = tmp_path / "big.wav"
    oversized.write_bytes(b"\x00" * (MiB_bytes + 1))

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        AudioMediaIO().load_file(oversized)


def test_sync_http_reader_rejects_from_content_length_before_body_read():
    response = _SyncResponse([b"A" * 10], content_length=MiB_bytes + 1)
    connection = HTTPConnection()

    with (
        patch.object(connection, "get_response", return_value=response) as get_response,
        pytest.raises(VLLMValidationError, match="Maximum file size exceeded"),
    ):
        connection.get_bytes("https://example.test/audio", max_bytes=MiB_bytes)

    assert get_response.call_args.kwargs["stream"] is True
    assert response.iterated == 0


@pytest.mark.asyncio
async def test_async_http_reader_stops_after_first_over_limit_chunk():
    response = _AsyncResponse([b"A" * (64 * KiB_bytes), b"B", b"C" * 10])
    connection = HTTPConnection()

    async def get_async_response(*_args, **_kwargs):
        return response

    with (
        patch.object(connection, "get_async_response", side_effect=get_async_response),
        pytest.raises(VLLMValidationError, match="Maximum file size exceeded"),
    ):
        await connection.async_get_bytes(
            "https://example.test/audio",
            max_bytes=64 * KiB_bytes,
        )

    assert response.content.iterated == 2


@pytest.mark.asyncio
async def test_audio_connector_threads_byte_limit_to_http_reader(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)

    class _Connection:
        async def async_get_bytes(self, _url: str, **kwargs):
            assert kwargs["max_bytes"] == MiB_bytes
            raise VLLMValidationError("Maximum file size exceeded")

    connector = MediaConnector(connection=_Connection())  # type: ignore[arg-type]
    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await connector.fetch_audio_async("https://example.test/audio")
