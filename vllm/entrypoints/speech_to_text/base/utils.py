# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared utilities for speech-to-text API routes."""

from collections.abc import Sequence
from typing import Protocol

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


class SubtitleSegment(Protocol):
    """Structural type for a timestamped speech-to-text segment."""

    start: float
    end: float
    text: str


def _format_timestamp(seconds: float, decimal_separator: str) -> str:
    """Format a time in seconds as an ``HH:MM:SS<sep>mmm`` subtitle stamp."""
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{milliseconds:03d}"


def format_segments_as_srt(segments: Sequence[SubtitleSegment]) -> str:
    """Render timestamped segments as an SRT subtitle track.

    Follows the shape returned by the OpenAI transcription API for
    ``response_format="srt"``: one numbered cue per segment with
    ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` timestamps and a blank line between
    cues. Segments whose text is empty after stripping are skipped and the
    remaining cues are renumbered accordingly.
    """
    cues = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        cues.append(
            f"{len(cues) + 1}\n"
            f"{_format_timestamp(seg.start, ',')} --> "
            f"{_format_timestamp(seg.end, ',')}\n"
            f"{text}\n"
        )
    return "\n".join(cues)


def format_segments_as_vtt(segments: Sequence[SubtitleSegment]) -> str:
    """Render timestamped segments as a WebVTT subtitle track.

    Follows the shape returned by the OpenAI transcription API for
    ``response_format="vtt"``: a ``WEBVTT`` header followed by unnumbered
    cues with ``HH:MM:SS.mmm --> HH:MM:SS.mmm`` timestamps and a blank
    line between cues. Segments whose text is empty after stripping are
    skipped.
    """
    cues = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        cues.append(
            f"{_format_timestamp(seg.start, '.')} --> "
            f"{_format_timestamp(seg.end, '.')}\n"
            f"{text}\n"
        )
    if not cues:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + "\n".join(cues)
