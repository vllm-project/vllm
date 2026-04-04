# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Magic-byte MIME sniffing for uploaded media files.

By default, uses an inline table of signatures for the supported formats
(mp4/mov, webm/matroska, png, jpeg, gif, webp, wav, mp3). Operators who
want broader/stricter detection can install `python-magic` — it will be
used automatically if importable.

This module intentionally has no required dependencies. The allowlist is
narrow (`video/*`, `image/*`, `audio/*`) because the file store exists
only to serve multimodal inputs.
"""

from __future__ import annotations

# Number of head bytes required for the sniffer. All signatures below fit
# comfortably in the first 16 bytes; 64 gives headroom for future formats.
SNIFF_HEAD_BYTES = 64

# (offset, expected_bytes, mime_type)
_FIXED_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    # PNG
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    # GIF87a / GIF89a
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    # JPEG (SOI marker + any APPn marker)
    (0, b"\xff\xd8\xff", "image/jpeg"),
    # Matroska / WebM (EBML header)
    (0, b"\x1a\x45\xdf\xa3", "video/webm"),
    # MP3 with ID3v2 tag
    (0, b"ID3", "audio/mpeg"),
)

# MPEG-1 Audio Layer III frame sync (11 bits = 0xFFE, followed by layer bits).
# Accept the common layer-III patterns seen in wild files without ID3 tags.
_MP3_FRAME_SYNC_PREFIXES: tuple[bytes, ...] = (
    b"\xff\xfb",
    b"\xff\xf3",
    b"\xff\xf2",
    b"\xff\xfa",
)

# ISO Base Media Format brands. The atom layout is:
#   [4 bytes atom size][b"ftyp"][4 bytes major brand]...
# We check bytes [4:8] == "ftyp" then classify by major brand at [8:12].
_ISO_VIDEO_BRANDS: frozenset[bytes] = frozenset(
    {
        b"isom",
        b"iso2",
        b"iso3",
        b"iso4",
        b"iso5",
        b"iso6",
        b"avc1",
        b"mp41",
        b"mp42",
        b"dash",
        b"msnv",
        b"MSNV",
        b"M4V ",
        b"M4VH",
        b"M4VP",
        b"f4v ",
        b"3gp4",
        b"3gp5",
        b"3gp6",
        b"3g2a",
    }
)
_ISO_AUDIO_BRANDS: frozenset[bytes] = frozenset(
    {
        b"M4A ",
        b"M4B ",
        b"M4P ",
    }
)
_ISO_QUICKTIME_BRANDS: frozenset[bytes] = frozenset(
    {
        b"qt  ",
    }
)


def _sniff_riff(head: bytes) -> str | None:
    """RIFF container — format is determined by bytes [8:12].

    Returns:
        The MIME type for recognised RIFF subtypes (WEBP/WAVE/AVI),
        or None when the head isn't a RIFF or the subtype is unknown.
    """
    if len(head) < 12 or head[0:4] != b"RIFF":
        return None
    subtype = head[8:12]
    if subtype == b"WEBP":
        return "image/webp"
    if subtype == b"WAVE":
        return "audio/wav"
    if subtype == b"AVI ":
        return "video/x-msvideo"
    return None


def _sniff_iso_base_media(head: bytes) -> str | None:
    """ISO Base Media (mp4, mov, m4a, 3gp). Brand at bytes [8:12].

    Returns:
        `video/mp4`, `audio/mp4`, or `video/quicktime` for known brands;
        `video/mp4` as a permissive fallback for unrecognised brands
        that still have the `ftyp` atom shape; or None when the head
        isn't an ISO Base Media container at all.
    """
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    brand = head[8:12]
    if brand in _ISO_VIDEO_BRANDS:
        return "video/mp4"
    if brand in _ISO_AUDIO_BRANDS:
        return "audio/mp4"
    if brand in _ISO_QUICKTIME_BRANDS:
        return "video/quicktime"
    # Unknown brand but still ftyp-shaped — treat as generic mp4 to avoid
    # rejecting legitimate but rare brands. The allowlist gate below will
    # still accept it (video/mp4 is in the allowed prefixes).
    return "video/mp4"


def _sniff_inline(head: bytes) -> str | None:
    for offset, signature, mime in _FIXED_SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return mime
    for prefix in _MP3_FRAME_SYNC_PREFIXES:
        if head.startswith(prefix):
            return "audio/mpeg"
    if (m := _sniff_riff(head)) is not None:
        return m
    if (m := _sniff_iso_base_media(head)) is not None:
        return m
    return None


# Optional python-magic upgrade path. If the module is importable we use it
# for a broader detection range; otherwise we fall back to the inline table.
try:
    import magic as _magic  # type: ignore[import-not-found]

    _PYMAGIC: object | None = _magic.Magic(mime=True)
except Exception:  # pragma: no cover — import side-effect path
    _PYMAGIC = None


def sniff_mime(head: bytes) -> str | None:
    """Return the sniffed MIME type for the file head bytes.

    Caller passes the first `SNIFF_HEAD_BYTES` bytes (or fewer for tiny
    files). We do not mutate or retain the input.

    Returns:
        The sniffed MIME type as a string, or None when the head does
        not match any supported signature.
    """
    # python-magic path (opt-in via pip install)
    if _PYMAGIC is not None:
        try:
            mime = _PYMAGIC.from_buffer(head)  # type: ignore[attr-defined]
            if isinstance(mime, str) and mime:
                return mime
        except Exception:
            # Fall through to inline detection on any pymagic error.
            pass
    return _sniff_inline(head)


# MIME-type prefixes that the upload store accepts. The store exists only
# to serve multimodal inputs to the model, so the allowlist is narrow by
# design.
_ALLOWED_PREFIXES: tuple[str, ...] = ("video/", "image/", "audio/")


def is_allowed_mime(mime: str | None) -> bool:
    """Check whether a sniffed MIME type is in the media allowlist.

    Returns:
        True when `mime` starts with `video/`, `image/`, or `audio/`;
        False for None, empty string, or any other prefix.
    """
    if not mime:
        return False
    return any(mime.startswith(p) for p in _ALLOWED_PREFIXES)
