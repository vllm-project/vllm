# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncIterator, Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import vllm.envs as envs
from vllm.connections import HTTPConnection, MediaRequestDownloadBudget
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import AudioMediaIO, ImageMediaIO, MediaConnector
from vllm.multimodal.media.base import MediaDecodedBudget
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


def test_audio_filesize_limit_zero_disables_size_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=0 disables the encoded-size checks.

    Per docs/usage/security.md, setting the limit to 0 disables it, so
    bytes, base64, and file inputs must all reach the decoder.
    """
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 0)

    assert AudioMediaIO().get_max_bytes() is None

    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"\x00" * 1024)

    with patch(
        "vllm.multimodal.media.audio.load_audio", return_value=(None, 0.0)
    ) as decode:
        AudioMediaIO().load_bytes(b"x")
        AudioMediaIO().load_base64("audio/wav", "eA==")
        AudioMediaIO().load_file(wav)

    assert decode.call_count == 3


def test_audio_filesize_limit_positive_still_enforced(
    monkeypatch: pytest.MonkeyPatch,
):
    """A positive limit still reports through get_max_bytes()."""
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)

    assert AudioMediaIO().get_max_bytes() == MiB_bytes


def test_audio_load_bytes_rejects_oversized(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        AudioMediaIO().load_bytes(b"\x00" * (MiB_bytes + 1))


def test_request_audio_budget_limits_multiple_clips_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 24)
    budget = MediaDecodedBudget()
    clip = np.zeros(4, dtype=np.float32)

    def decode(_source, **kwargs):
        if kwargs["max_decode_bytes"] < clip.nbytes:
            raise ValueError("VLLM_MAX_AUDIO_DECODE_BYTES")
        return clip, 16000

    with patch("vllm.multimodal.media.audio.load_audio", side_effect=decode) as load:
        audio_io = AudioMediaIO(decoded_budget=budget)
        audio_io.load_bytes(b"first")
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
            audio_io.load_bytes(b"second")

    assert budget.used == clip.nbytes
    assert load.call_count == 2


def test_request_decoded_budget_is_shared_by_image_and_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 24)
    budget = MediaDecodedBudget()
    image_bytes = BytesIO()
    Image.new("RGB", (2, 3)).save(image_bytes, format="PNG")
    ImageMediaIO(decoded_budget=budget).load_bytes(image_bytes.getvalue())

    with (
        patch("vllm.multimodal.media.audio.load_audio") as decode,
        pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"),
    ):
        AudioMediaIO(decoded_budget=budget).load_bytes(b"audio")

    decode.assert_not_called()


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
    budget = MediaRequestDownloadBudget()

    class _Connection:
        async def async_get_bytes(self, _url: str, **kwargs):
            assert kwargs["max_bytes"] == MiB_bytes
            assert kwargs["request_budget"] is budget
            raise VLLMValidationError("Maximum file size exceeded")

    connector = MediaConnector(  # type: ignore[arg-type]
        connection=_Connection(), download_budget=budget
    )
    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await connector.fetch_audio_async("https://example.test/audio")


@pytest.mark.asyncio
async def test_audio_connector_threads_none_limit_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """A disabled limit reaches the HTTP reader as max_bytes=None.

    vllm.connections treats None as "no audio-specific limit" while the
    global media download cap still applies.
    """
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 0)

    class _Connection:
        async def async_get_bytes(self, _url: str, **kwargs):
            assert kwargs["max_bytes"] is None
            raise VLLMValidationError("Maximum file size exceeded")

    connector = MediaConnector(connection=_Connection())
    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await connector.fetch_audio_async("https://example.test/audio")
