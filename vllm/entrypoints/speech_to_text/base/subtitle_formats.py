# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SRT / WebVTT subtitle formatting for speech-to-text segment output."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..transcription.protocol import TranscriptionSegment
    from ..translation.protocol import TranslationSegment

    Segment = TranscriptionSegment | TranslationSegment


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: "list[Segment]") -> str:
    """Convert segments to SRT subtitle format."""
    parts = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        parts.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}")
    return "\n\n".join(parts) + "\n" if parts else ""


def segments_to_vtt(segments: "list[Segment]") -> str:
    """Convert segments to WebVTT subtitle format."""
    parts = ["WEBVTT"]
    for seg in segments:
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        parts.append(f"{start} --> {end}\n{seg.text.strip()}")
    return "\n\n".join(parts) + "\n"
