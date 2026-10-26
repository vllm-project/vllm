# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.media_utils import validate_media_bytes, validate_media_source

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


@pytest.mark.parametrize(
    "data",
    [
        b"ffconcat version 1.0\nfile 'private.avi'\n",
        b"\xef\xbb\xbf#EXTM3U\n#EXT-X-TARGETDURATION:1\nprivate.ts\n",
        b" \xef\xbb\xbf#EXTM3U\nprivate.ts\n",
        b"v=0\r\nc=IN IP4 127.0.0.1\r\nm=video 1234 RTP/AVP 96\r\n",
        b"s=session\nc=IN IP6 ::1\n",
        b"<?xml version='1.0'?><MPD></MPD>",
        b"<MPD></MPD>",
        "<MPD></MPD>".encode("utf-16"),
    ],
)
@pytest.mark.parametrize("source_type", ["bytes", "buffer", "file"])
def test_media_reference_descriptions_rejected(tmp_path, data, source_type):
    """Reference formats must fail before a decoder can open another resource."""
    with pytest.raises(ValueError, match="playlists and stream descriptions"):
        if source_type == "bytes":
            validate_media_bytes(data)
        elif source_type == "buffer":
            validate_media_source(BytesIO(data))
        else:
            path = tmp_path / "media"
            path.write_bytes(data)
            validate_media_source(path)


def test_binary_media_metadata_is_not_a_reference_description(tmp_path):
    data = b"RIFF\x00\x00\x00\x00WAVE\nc=IN IP4 127.0.0.1\n"
    path = tmp_path / "media.wav"
    path.write_bytes(data)
    validate_media_bytes(data)
    validate_media_source(path)
    source = BytesIO(data)
    source.seek(8)
    validate_media_source(source)
    assert source.tell() == 8


def test_media_with_bytes_pickle_roundtrip():
    """Regression test for pickle/unpickle of MediaWithBytes.

    Verifies that MediaWithBytes can be pickled and unpickled without
    RecursionError. See: https://github.com/vllm-project/vllm/issues/30818
    """
    original_image = Image.open(ASSETS_DIR / "image1.png").convert("RGB")
    original_bytes = b"test_bytes_data"

    wrapper = MediaWithBytes(media=original_image, original_bytes=original_bytes)

    # Verify attribute delegation works before pickling
    assert wrapper.width == original_image.width
    assert wrapper.height == original_image.height
    assert wrapper.mode == original_image.mode

    # Pickle and unpickle (this would cause RecursionError before the fix)
    pickled = pickle.dumps(wrapper)
    unpickled = pickle.loads(pickled)

    # Verify the unpickled object works correctly
    assert unpickled.original_bytes == original_bytes
    assert unpickled.media.width == original_image.width
    assert unpickled.media.height == original_image.height

    # Verify attribute delegation works after unpickling
    assert unpickled.width == original_image.width
    assert unpickled.height == original_image.height
    assert unpickled.mode == original_image.mode
