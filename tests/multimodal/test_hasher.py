# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from vllm.multimodal.hasher import MultiModalHasher

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


def test_hash_collision_tensor_shape():
    # The hash should be different though the data is the same when flattened
    arr1 = torch.zeros((5, 10, 20, 3))
    arr2 = torch.zeros((10, 20, 5, 3))

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
