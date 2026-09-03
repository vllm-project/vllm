# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from pathlib import Path

import pytest
from PIL import Image

from vllm.multimodal.media import MediaWithBytes

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


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
