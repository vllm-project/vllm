# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from vllm.multimodal.hasher import MultiModalHasher

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


# NOTE: Images that are the same visually are allowed to have the same hash
@pytest.mark.parametrize("mode_pair", [("1", "L"), ("RGBA", "CMYK")])
def test_hash_collision_image_mode(mode_pair):
    mode1, mode2 = mode_pair
    image1 = Image.new(mode1, size=(10, 10), color=1)
    image2 = Image.new(mode2, size=(10, 10), color=1)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


def test_hash_collision_image_palette():
    # These images differ only in Image.palette._palette
    image1 = Image.open(ASSETS_DIR / "image1.png")
    image2 = Image.open(ASSETS_DIR / "image2.png")

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


def test_hash_collision_image_transpose():
    image1 = Image.new("1", size=(10, 20))
    ImageDraw.Draw(image1).line([(0, 0), (10, 0)])

    image2 = Image.new("1", size=(20, 10))
    ImageDraw.Draw(image2).line([(0, 0), (0, 10)])

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hash_collision_tensor_shape(dtype):
    # The hash should be different though the data is the same when flattened
    arr1 = torch.zeros((5, 10, 20, 3), dtype=dtype)
    arr2 = torch.zeros((10, 20, 5, 3), dtype=dtype)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(data=arr1) != hasher.hash_kwargs(data=arr2)


def test_hash_collision_array_shape():
    # The hash should be different though the data is the same when flattened
    arr1 = np.zeros((5, 10, 20, 3))
    arr2 = np.zeros((10, 20, 5, 3))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(data=arr1) != hasher.hash_kwargs(data=arr2)


def test_hash_non_contiguous_array():
    arr = np.arange(24).reshape(4, 6).T
    assert not arr.flags.c_contiguous

    arr_c = np.ascontiguousarray(arr)
    assert arr_c.flags.c_contiguous

    hasher = MultiModalHasher
    # Both should be hashable and produce the same hashes
    assert hasher.hash_kwargs(data=arr) == hasher.hash_kwargs(data=arr_c)


def test_hash_image_exif_id():
    # Test that EXIF ImageId tag can be used to store UUID
    # and the hasher will use that instead of the image data.
    image1 = image2 = Image.new("1", size=(10, 20))
    id = uuid.uuid4()
    image1.getexif()[Image.ExifTags.Base.ImageID] = id
    image2 = Image.open(ASSETS_DIR / "image1.png")
    image2.getexif()[Image.ExifTags.Base.ImageID] = "Not a UUID"
    image2a = Image.open(ASSETS_DIR / "image1.png")

    hasher = MultiModalHasher
    # first image has UUID in ImageID, so it should hash to that UUID
    assert hasher.hash_kwargs(image=image1) == hasher.hash_kwargs(image=id.bytes)
    # second image has non-UUID in ImageID, so it should hash to the image data
    assert hasher.hash_kwargs(image=image2) == hasher.hash_kwargs(image=image2a)


# Tests for FIPS 140-3 compliant hashing support
class TestFIPSHashing:
    """Test FIPS-compliant SHA-256 hashing functionality."""

    def test_sha256_hasher_basic(self):
        """Test that _Sha256Hasher produces valid hashes."""
        from vllm.multimodal.hasher import _Sha256Hasher

        hasher = _Sha256Hasher()
        hasher.update(b"test data")
        result = hasher.hexdigest()

        # SHA-256 produces 64-character hex digest
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_sha256_hasher_memoryview(self):
        """Test that _Sha256Hasher handles memoryview correctly."""
        from vllm.multimodal.hasher import _Sha256Hasher

        data = b"test data"
        mv = memoryview(data)

        hasher1 = _Sha256Hasher()
        hasher1.update(data)

        hasher2 = _Sha256Hasher()
        hasher2.update(mv)

        assert hasher1.hexdigest() == hasher2.hexdigest()

    def test_blake3_hasher_basic(self):
        """Test that _Blake3Hasher produces valid hashes when available."""
        from vllm.multimodal.hasher import _HAS_BLAKE3, _Blake3Hasher

        if not _HAS_BLAKE3:
            pytest.skip("blake3 not available")

        hasher = _Blake3Hasher()
        hasher.update(b"test data")
        result = hasher.hexdigest()

        # blake3 also produces 64-character hex digest by default
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_blake3_and_sha256_produce_different_hashes(self):
        """Test that blake3 and SHA-256 produce different hashes for same input."""
        from vllm.multimodal.hasher import _HAS_BLAKE3, _Blake3Hasher, _Sha256Hasher

        if not _HAS_BLAKE3:
            pytest.skip("blake3 not available")

        data = b"test data for hashing"

        blake3_hasher = _Blake3Hasher()
        blake3_hasher.update(data)

        sha256_hasher = _Sha256Hasher()
        sha256_hasher.update(data)

        # Different algorithms should produce different hashes
        assert blake3_hasher.hexdigest() != sha256_hasher.hexdigest()

    def test_create_hasher_returns_correct_type(self):
        """Test that _create_hasher returns appropriate hasher type."""
        from vllm.multimodal.hasher import (
            _USE_FIPS_HASHING,
            _Blake3Hasher,
            _create_hasher,
            _Sha256Hasher,
        )

        hasher = _create_hasher()

        if _USE_FIPS_HASHING:
            assert isinstance(hasher, _Sha256Hasher)
        else:
            assert isinstance(hasher, _Blake3Hasher)

    def test_hash_kwargs_consistency_with_fips(self):
        """Test that hash_kwargs produces consistent results."""
        data = {"key1": "value1", "key2": 42, "key3": b"bytes"}

        hash1 = MultiModalHasher.hash_kwargs(**data)
        hash2 = MultiModalHasher.hash_kwargs(**data)

        assert hash1 == hash2

    def test_hash_kwargs_with_image_fips(self):
        """Test that image hashing works in FIPS mode."""
        image = Image.new("RGB", size=(10, 10), color=(255, 0, 0))

        # Should not raise an exception
        result = MultiModalHasher.hash_kwargs(image=image)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_kwargs_with_tensor_fips(self):
        """Test that tensor hashing works in FIPS mode."""
        tensor = torch.zeros((5, 10, 20), dtype=torch.float32)

        # Should not raise an exception
        result = MultiModalHasher.hash_kwargs(data=tensor)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_kwargs_with_numpy_array_fips(self):
        """Test that numpy array hashing works in FIPS mode."""
        arr = np.zeros((5, 10, 20))

        # Should not raise an exception
        result = MultiModalHasher.hash_kwargs(data=arr)
        assert isinstance(result, str)
        assert len(result) == 64
