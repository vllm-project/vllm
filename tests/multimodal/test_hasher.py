# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from PIL import Image

from vllm.multimodal.hasher import MultiModalHasher

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


def test_hash_collision_regression():
    # These images differ only in Image.palette._palette
    image1 = Image.open(ASSETS_DIR / "image1.png")
    image2 = Image.open(ASSETS_DIR / "image2.png")

    hasher = MultiModalHasher
    assert hasher.hash_kwargs(image=image1) != hasher.hash_kwargs(image=image2)
