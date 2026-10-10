# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SRT / WebVTT subtitle formatting for speech-to-text segment output."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..transcription.protocol import TranscriptionSegment
    from ..translation.protocol import TranslationSegment

    Segment = TranscriptionSegment | TranslationSegment


def _format_timestamp(seconds: float, ms_sep: str) -> str:
    """Format seconds as HH:MM:SS<ms_sep>mmm (SRT uses ',', VTT uses '.')."""
    # Round in integer milliseconds so 11.54 -> 540 (not 539 via float
    # truncation) and 0.9995 -> 001,000 -> 00:00:01,000 (never ",1000").
    hours, rem = divmod(round(seconds * 1000), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_sep}{millis:03d}"


def _cue_text(text: str, vtt: bool) -> str:
    # "-->" inside cue text would be parsed as a timing line.
    text = text.strip().replace("-->", "->")
    if vtt:
        # WebVTT cue payloads must escape '&' and '<' (tag / entity starts).
        text = text.replace("&", "&amp;").replace("<", "&lt;")
    return text


def segments_to_srt(segments: "list[Segment]") -> str:
    """Convert segments to SRT subtitle format."""
    parts = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg.start, ",")
        end = _format_timestamp(seg.end, ",")
        parts.append(f"{i}\n{start} --> {end}\n{_cue_text(seg.text, vtt=False)}")
    return "\n\n".join(parts) + "\n" if parts else ""


def segments_to_vtt(segments: "list[Segment]") -> str:
    """Convert segments to WebVTT subtitle format."""
    parts = ["WEBVTT"]
    for seg in segments:
        start = _format_timestamp(seg.start, ".")
        end = _format_timestamp(seg.end, ".")
        parts.append(f"{start} --> {end}\n{_cue_text(seg.text, vtt=True)}")
    return "\n\n".join(parts) + "\n"
