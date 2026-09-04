# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for speech-to-text API routes."""

from fastapi import UploadFile

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.utils.mem_constants import KiB_bytes, MiB_bytes

_READ_CHUNK_SIZE = 64 * KiB_bytes


async def read_upload_with_limit(
    file: UploadFile,
    max_size_mb: float | None = None,
) -> bytes:
    """Read an uploaded file enforcing a size limit *before* full
    materialization.

    The function first checks the Content-Length header (``file.size``) when
    available.  Regardless, it then performs a chunked read that stops as soon
    as the accumulated bytes exceed the limit, ensuring that an oversized
    upload never fully materializes in memory.

    Args:
        file: The FastAPI/Starlette ``UploadFile`` object.
        max_size_mb: Maximum allowed compressed file size in megabytes.
            Defaults to ``envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB``.

    Returns:
        The file content as ``bytes``.

    Raises:
        VLLMValidationError: If the file exceeds the configured size limit.
    """
    if max_size_mb is None:
        max_size_mb = envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB

    max_bytes = int(max_size_mb * MiB_bytes)

    if file.size is not None and file.size > max_bytes:
        raise VLLMValidationError(
            "Maximum file size exceeded",
            parameter="audio_filesize_mb",
            value=file.size / MiB_bytes,
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise VLLMValidationError(
                "Maximum file size exceeded",
                parameter="audio_filesize_mb",
                value=total / MiB_bytes,
            )
        chunks.append(chunk)

    return b"".join(chunks)
