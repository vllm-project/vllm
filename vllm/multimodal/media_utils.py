# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path

import regex as re


def validate_media_bytes(data: bytes) -> None:
    """Reject media descriptors that can open additional files or sockets."""
    nul = data.find(b"\x00")
    header = data[:nul] if nul >= 0 else data
    header = header.lstrip().removeprefix(b"\xef\xbb\xbf").lstrip()
    if header.startswith(
        (b"ffconcat version", b"#EXTM3U", b"v=0", b"<", b"\xff\xfe", b"\xfe\xff")
    ) or re.search(rb"(?:^|[\r\n])c=IN IP[46](?:[ \t]|$)", header):
        raise ValueError("Media playlists and stream descriptions are not supported.")


def validate_media_source(source: BytesIO | Path | str) -> None:
    if isinstance(source, BytesIO):
        validate_media_bytes(source.getvalue())
    else:
        with open(source, "rb") as stream:
            for line in stream:
                validate_media_bytes(line)
                if b"\x00" in line:
                    break
