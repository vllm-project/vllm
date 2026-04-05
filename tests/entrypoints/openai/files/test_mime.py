# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.files import mime
from vllm.entrypoints.openai.files.mime import (
    SNIFF_HEAD_BYTES,
    _sniff_inline,
    is_allowed_mime,
    sniff_mime,
)

# Short valid signatures for each supported format. These are real magic
# bytes truncated to the minimum required; the sniffer never reads past
# SNIFF_HEAD_BYTES so padding isn't necessary.
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
_GIF87 = b"GIF87a" + b"\x00" * 10
_GIF89 = b"GIF89a" + b"\x00" * 10
_WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "
_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "
_WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 12
_MP3_ID3 = b"ID3\x04\x00" + b"\x00" * 11
_MP3_SYNC = b"\xff\xfb\x90\x00" + b"\x00" * 12
_MP4_ISOM = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 4
_MP4_MP42 = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 4
_MP4_M4V = b"\x00\x00\x00\x20ftypM4V " + b"\x00" * 4
_M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4
_MOV = b"\x00\x00\x00\x20ftypqt  " + b"\x00" * 4
_MP4_UNKNOWN_BRAND = b"\x00\x00\x00\x20ftypXXXX" + b"\x00" * 4
_HEIC = b"\x00\x00\x00\x20ftypheic" + b"\x00" * 4
_HEIC_MIF1 = b"\x00\x00\x00\x20ftypmif1" + b"\x00" * 4
_AVIF = b"\x00\x00\x00\x20ftypavif" + b"\x00" * 4
# MP3 frame-sync with a RESERVED bitrate index (0xF) in byte 2 — byte 2
# bits EEEE.FFGH, 0xFC = 1111_1100, so bitrate=0xF (reserved), must reject.
_MP3_SYNC_BAD_BITRATE = b"\xff\xfb\xfc\x00" + b"\x00" * 12
# MP3 frame-sync with a RESERVED sample-rate index (0x3) in byte 2 — 0x0C = 0000_1100
_MP3_SYNC_BAD_SAMPLERATE = b"\xff\xfb\x0c\x00" + b"\x00" * 12


@pytest.mark.parametrize(
    "head,expected",
    [
        (_PNG, "image/png"),
        (_JPEG, "image/jpeg"),
        (_GIF87, "image/gif"),
        (_GIF89, "image/gif"),
        (_WEBP, "image/webp"),
        (_WAV, "audio/wav"),
        (_WEBM, "video/webm"),
        (_MP3_ID3, "audio/mpeg"),
        (_MP3_SYNC, "audio/mpeg"),
        (_MP4_ISOM, "video/mp4"),
        (_MP4_MP42, "video/mp4"),
        (_MP4_M4V, "video/mp4"),
        (_M4A, "audio/mp4"),
        (_MOV, "video/quicktime"),
        (_MP4_UNKNOWN_BRAND, "video/mp4"),
        (_HEIC, "image/heic"),
        (_HEIC_MIF1, "image/heic"),
        (_AVIF, "image/avif"),
    ],
)
def test_inline_sniffer_recognises_supported_formats(head, expected):
    assert _sniff_inline(head) == expected


@pytest.mark.parametrize(
    "head",
    [
        _MP3_SYNC_BAD_BITRATE,
        _MP3_SYNC_BAD_SAMPLERATE,
    ],
)
def test_mp3_frame_sync_validates_header_bits(head):
    """Random binary blobs that start with \\xff\\xfb (or similar) but
    have a reserved bitrate/sample-rate index must NOT be classified as
    audio/mpeg — the frame-sync prefix alone is only 12 bits and
    collides ~1/2048. The byte-2 bitrate and sample-rate bits guard
    against false positives on random data."""
    assert _sniff_inline(head) is None


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"\x00" * 64,
        b"plain text payload, not media at all, nope",
        b"<!DOCTYPE html><html><body></body></html>",
        b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 16,  # ELF executable
        b"MZ\x90\x00\x03\x00" + b"\x00" * 16,  # Windows PE/DOS
        b"#!/bin/bash\necho 'nope'\n",
        b"RIFF" + b"\x00" * 4 + b"NOPE" + b"\x00" * 4,  # RIFF, wrong subtype
    ],
)
def test_inline_sniffer_rejects_non_media(head):
    assert _sniff_inline(head) is None


def test_sniff_mime_truncated_inputs_do_not_crash():
    """Tiny heads are possible (empty upload, single byte) — must not
    raise; must return None."""
    assert sniff_mime(b"") is None
    assert sniff_mime(b"\x89") is None
    assert sniff_mime(b"\x89P") is None


def test_sniff_head_bytes_matches_longest_signature():
    """The SNIFF_HEAD_BYTES constant should be enough for every signature
    the inline sniffer checks. If someone adds a longer signature they
    must bump SNIFF_HEAD_BYTES."""
    # 12 bytes covers RIFF + subtype and ftyp + brand, the longest we check.
    assert SNIFF_HEAD_BYTES >= 12


@pytest.mark.parametrize(
    "mime_type,allowed",
    [
        ("video/mp4", True),
        ("video/webm", True),
        ("video/quicktime", True),
        ("image/png", True),
        ("image/jpeg", True),
        ("image/gif", True),
        ("image/webp", True),
        ("audio/mpeg", True),
        ("audio/wav", True),
        ("audio/mp4", True),
        ("application/octet-stream", False),
        ("text/plain", False),
        ("application/pdf", False),
        ("application/x-executable", False),
        ("", False),
        (None, False),
    ],
)
def test_is_allowed_mime(mime_type, allowed):
    assert is_allowed_mime(mime_type) is allowed


def test_inline_path_used_when_pymagic_absent(monkeypatch):
    """When python-magic is not installed, sniff_mime must fall back to
    the inline table. Exercises the always-available path."""
    monkeypatch.setattr(mime, "_PYMAGIC", None)
    assert sniff_mime(_PNG) == "image/png"
    assert sniff_mime(b"plain text") is None


def test_pymagic_path_is_tried_first(monkeypatch):
    """When python-magic IS available, sniff_mime should try it first and
    only fall back to inline when pymagic returns empty/raises."""

    class FakeMagic:
        def __init__(self, result: str | None):
            self._result = result
            self.calls = 0

        def from_buffer(self, head: bytes) -> str | None:
            self.calls += 1
            return self._result

    # Case 1: pymagic returns a valid MIME — inline table not consulted.
    fake = FakeMagic("video/x-matroska")
    monkeypatch.setattr(mime, "_PYMAGIC", fake)
    assert sniff_mime(_WEBM) == "video/x-matroska"
    assert fake.calls == 1

    # Case 2: pymagic raises — fall back to inline.
    class RaisingMagic:
        def from_buffer(self, head: bytes) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(mime, "_PYMAGIC", RaisingMagic())
    assert sniff_mime(_PNG) == "image/png"  # inline path

    # Case 3: pymagic returns empty string — fall back to inline.
    fake_empty = FakeMagic("")
    monkeypatch.setattr(mime, "_PYMAGIC", fake_empty)
    assert sniff_mime(_JPEG) == "image/jpeg"
    assert fake_empty.calls == 1
