# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the speech-to-text upload size pre-check.

These tests verify that over-limit audio uploads are rejected *before*
the full file is materialized into memory, closing the vulnerability
where vLLM would allocate memory proportional to an oversized upload
before enforcing the VLLM_MAX_AUDIO_CLIP_FILESIZE_MB limit.
"""

from unittest.mock import AsyncMock, patch

import pytest

from vllm.entrypoints.speech_to_text.base.utils import read_upload_with_limit
from vllm.exceptions import VLLMValidationError


def _make_upload_file(data: bytes, *, size: int | None = None) -> AsyncMock:
    """Create a mock UploadFile that yields data in chunks."""
    mock = AsyncMock()
    mock.size = size

    offset = 0

    async def _read(n: int = -1):
        nonlocal offset
        if n <= 0:
            chunk = data[offset:]
            offset = len(data)
            return chunk
        chunk = data[offset : offset + n]
        offset += len(chunk)
        return chunk

    mock.read = AsyncMock(side_effect=_read)
    return mock


@pytest.mark.asyncio
async def test_rejects_oversized_upload_via_content_length():
    """File is rejected early when file.size exceeds the limit."""
    max_mb = 1
    oversized_bytes = max_mb * 1024 * 1024 + 1

    upload = _make_upload_file(b"", size=oversized_bytes)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await read_upload_with_limit(upload, max_size_mb=max_mb)

    upload.read.assert_not_called()


@pytest.mark.asyncio
async def test_rejects_oversized_upload_via_chunked_read():
    """File is rejected mid-read without materializing the full content."""
    max_mb = 1
    max_bytes = max_mb * 1024 * 1024
    oversized_data = b"\x00" * (max_bytes + 1024)

    upload = _make_upload_file(oversized_data, size=None)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await read_upload_with_limit(upload, max_size_mb=max_mb)


@pytest.mark.asyncio
async def test_accepts_file_within_limit():
    """File within the limit is read successfully."""
    max_mb = 1
    data = b"\x00" * (512 * 1024)  # 512 KiB, well under 1 MB

    upload = _make_upload_file(data, size=len(data))
    result = await read_upload_with_limit(upload, max_size_mb=max_mb)

    assert result == data


@pytest.mark.asyncio
async def test_accepts_file_at_exact_limit():
    """File exactly at the limit boundary is accepted."""
    max_mb = 1
    max_bytes = max_mb * 1024 * 1024
    data = b"\x00" * max_bytes

    upload = _make_upload_file(data, size=len(data))
    result = await read_upload_with_limit(upload, max_size_mb=max_mb)

    assert result == data


@pytest.mark.asyncio
async def test_rejects_at_one_byte_over_limit():
    """File one byte over the limit is rejected."""
    max_mb = 1
    max_bytes = max_mb * 1024 * 1024
    data = b"\x00" * (max_bytes + 1)

    upload = _make_upload_file(data, size=None)

    with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
        await read_upload_with_limit(upload, max_size_mb=max_mb)


@pytest.mark.asyncio
async def test_uses_env_default_when_no_limit_specified():
    """Uses VLLM_MAX_AUDIO_CLIP_FILESIZE_MB when max_size_mb is not given."""
    with patch("vllm.entrypoints.speech_to_text.base.utils.envs") as mock_envs:
        mock_envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB = 2
        max_bytes = 2 * 1024 * 1024
        oversized_data = b"\x00" * (max_bytes + 1)

        upload = _make_upload_file(oversized_data, size=None)

        with pytest.raises(VLLMValidationError, match="Maximum file size exceeded"):
            await read_upload_with_limit(upload)


@pytest.mark.asyncio
async def test_chunked_read_does_not_fully_materialize():
    """Verify that for large oversized files, we stop reading early.

    The function reads in 64 KiB chunks and aborts once the accumulated
    size exceeds the limit. We confirm that far fewer read calls were made
    than would be required to fully materialize the file.
    """
    max_mb = 1
    max_bytes = max_mb * 1024 * 1024
    large_size = max_bytes * 10  # 10x the limit
    data = b"\x00" * large_size

    upload = _make_upload_file(data, size=None)

    with pytest.raises(VLLMValidationError):
        await read_upload_with_limit(upload, max_size_mb=max_mb)

    chunk_size = 64 * 1024
    calls_for_full_read = large_size // chunk_size + 1
    calls_to_exceed_limit = max_bytes // chunk_size + 1
    actual_calls = upload.read.call_count
    assert actual_calls <= calls_to_exceed_limit + 1
    assert actual_calls < calls_for_full_read
