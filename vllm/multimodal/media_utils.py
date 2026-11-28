# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import regex as re

_UTF8_BOM = b"\xef\xbb\xbf"
_REFERENCE_HEADER = re.compile(
    rb"\s*(?:\xef\xbb\xbf\s*)?(?:ffconcat version|#EXTM3U|v=0|<|\xff\xfe|\xfe\xff"
    rb"|(?P<sdp>c=IN IP[46])(?=[ \t]|$))"
)
_SDP_CONNECTION = re.compile(rb"(?:^|[\r\n])c=IN IP[46](?:[ \t]|$)")
_SDP_OVERLAP = len(b"\nc=IN IP4")
_HEADER_READ_BYTES = 64 * 1024
_REFERENCE_ERROR = "Media playlists and stream descriptions are not supported."


def validate_media_bytes(data: bytes) -> None:
    """Reject media descriptors that can open additional files or sockets."""
    nul = data.find(b"\x00")
    header = data[:nul] if nul >= 0 else data
    if _REFERENCE_HEADER.match(header) or _SDP_CONNECTION.search(header):
        raise ValueError(_REFERENCE_ERROR)


def _validate_media_file(stream: BinaryIO) -> None:
    prefix: bytes | None = b""
    overlap = b""
    while True:
        raw = stream.readline(_HEADER_READ_BYTES)
        data, nul, _ = raw.partition(b"\x00")
        end_of_line = raw.endswith(b"\n")
        complete_line = bool(nul) or end_of_line or len(raw) < _HEADER_READ_BYTES

        if prefix is not None:
            candidate = prefix + data
            match = _REFERENCE_HEADER.match(candidate, partial=True)
            if match is None:
                prefix = None
            elif not match.partial and (
                match.lastgroup != "sdp"
                or complete_line
                or match.end() < len(candidate)
            ):
                raise ValueError(_REFERENCE_ERROR)
            else:
                # Retain an incomplete signature without retaining whitespace.
                prefix = candidate.lstrip()
                if prefix.startswith(_UTF8_BOM):
                    prefix = _UTF8_BOM + prefix[len(_UTF8_BOM) :].lstrip()

        # Pad partial lines so a chunk edge never acts as a line boundary.
        window = b"\x01" + overlap + data if overlap else data
        if not complete_line:
            window += b"\x01"
        if _SDP_CONNECTION.search(window):
            raise ValueError(_REFERENCE_ERROR)

        if nul or not raw:
            return
        if end_of_line:
            prefix, overlap = b"", b""
        else:
            overlap = (overlap + data)[-_SDP_OVERLAP:]


def validate_media_source(source: BytesIO | Path | str) -> None:
    if isinstance(source, BytesIO):
        validate_media_bytes(source.getvalue())
    else:
        with open(source, "rb") as stream:
            _validate_media_file(stream)
