# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.media import LazyMedia, MediaWithBytes

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


class _CountingDecoder:
    """Decoder that records how many times it ran."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


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


def test_lazy_media_decodes_on_first_access():
    """LazyMedia must not decode on construction, repr, or bytes access."""
    decoder = _CountingDecoder("decoded")
    item = LazyMedia(decoder, b"raw-bytes")

    assert not item.is_decoded
    assert repr(item) == "<LazyMedia undecoded, 9 bytes>"
    assert item.original_bytes == b"raw-bytes"
    assert decoder.calls == 0

    assert item.media == "decoded"
    assert item.is_decoded
    assert repr(item) == "<LazyMedia decoded, 9 bytes>"

    # Decoding is idempotent: the result is cached.
    assert item.media == "decoded"
    assert decoder.calls == 1


def test_lazy_media_concurrent_decode_runs_once():
    """Concurrent access from multiple threads decodes exactly once."""
    start = threading.Barrier(8)
    decoder_calls = 0
    lock = threading.Lock()

    def decode():
        nonlocal decoder_calls
        # Keep the decode window open so the other threads contend on it.
        time.sleep(0.05)
        with lock:
            decoder_calls += 1
        return "decoded"

    item = LazyMedia(decode, b"raw")

    def concurrent_decode():
        start.wait(timeout=10)
        return item.decode()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: concurrent_decode(), range(8)))

    assert results == ["decoded"] * 8
    assert decoder_calls == 1


def test_lazy_media_map_stacks_transform_and_shares_bytes():
    """map() defers the transform until decode and shares original_bytes."""
    decoder = _CountingDecoder(2)
    item = LazyMedia(decoder, b"raw")
    mapped = item.map(lambda x: x * 10)

    assert mapped.original_bytes is item.original_bytes
    assert decoder.calls == 0

    assert mapped.media == 20
    # The mapped item decodes the underlying item exactly once.
    assert decoder.calls == 1
    assert item.is_decoded
    assert item.media == 2


def test_lazy_media_release_bytes():
    item = LazyMedia(_CountingDecoder("decoded"), b"raw-bytes")

    item.release_bytes()
    assert item.original_bytes == b""

    # Decode still works after the bytes are released.
    assert item.media == "decoded"


def test_lazy_media_delegates_attributes():
    """Attribute access and __array__ delegate to the decoded media."""
    image = Image.new("RGB", (4, 2))
    item = LazyMedia(lambda: image, b"raw")

    assert item.size == (4, 2)
    assert np.array(item).shape == (2, 4, 3)


def test_lazy_media_pickle_becomes_eager():
    """Pickling materializes a LazyMedia into an eager MediaWithBytes."""
    image = Image.open(ASSETS_DIR / "image1.png").convert("RGB")
    item = LazyMedia(lambda: image, b"raw-bytes")

    unpickled = pickle.loads(pickle.dumps(item))

    assert type(unpickled) is MediaWithBytes
    assert np.array_equal(np.asarray(unpickled.media), np.asarray(image))
    assert unpickled.original_bytes == b"raw-bytes"


def test_lazy_media_pickle_flattens_media_with_bytes():
    """A decoder returning MediaWithBytes must not nest after pickling."""
    image = Image.new("RGB", (4, 2))
    inner = MediaWithBytes(image, b"raw-bytes", {"image_mode": "RGB"})
    item = LazyMedia(lambda: inner, b"")

    unpickled = pickle.loads(pickle.dumps(item))

    assert type(unpickled) is MediaWithBytes
    assert unpickled.media.size == image.size
    assert unpickled.media.mode == image.mode
    assert unpickled.original_bytes == b"raw-bytes"
    assert unpickled.io_config == {"image_mode": "RGB"}
