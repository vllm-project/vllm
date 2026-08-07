# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from vllm.config.multimodal import MMHasherAlgorithm
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.parse import MultiModalDataParser

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent / "assets"
assert ASSETS_DIR.exists()


@pytest.mark.parametrize("algorithm", ["sha256", "sha512"])
def test_hash_algorithm(algorithm: MMHasherAlgorithm):
    hasher = getattr(hashlib, algorithm)()
    for bytes_ in MultiModalHasher.iter_item_to_bytes("value", "test"):
        hasher.update(bytes_)

    assert MultiModalHasher.hash_kwargs(algorithm, value="test") == hasher.hexdigest()


def test_hash_algorithm_required():
    with pytest.raises(TypeError, match="algorithm"):
        MultiModalHasher.hash_kwargs(value="test")  # type: ignore[call-arg]


def test_hash_single_item_different_shape():
    x1 = torch.zeros(())
    x2 = torch.zeros((1,))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", x=x1) != hasher.hash_kwargs("blake3", x=x2)


def test_hash_key_order_invariant():
    x = torch.zeros((5, 10))
    y = torch.ones((5, 10))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", x=x, y=y) == hasher.hash_kwargs(
        "blake3", y=y, x=x
    )


# NOTE: Images that are the same visually are allowed to have the same hash
@pytest.mark.parametrize("mode_pair", [("1", "L"), ("RGBA", "CMYK")])
def test_hash_collision_image_mode(mode_pair):
    mode1, mode2 = mode_pair
    image1 = Image.new(mode1, size=(10, 10), color=1)
    image2 = Image.new(mode2, size=(10, 10), color=1)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=image1) != hasher.hash_kwargs(
        "blake3", image=image2
    )


def test_hash_collision_image_palette():
    # These images differ only in Image.palette._palette
    image1 = Image.open(ASSETS_DIR / "image1.png")
    image2 = Image.open(ASSETS_DIR / "image2.png")

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=image1) != hasher.hash_kwargs(
        "blake3", image=image2
    )


def test_hash_collision_image_transpose():
    image1 = Image.new("1", size=(10, 20))
    ImageDraw.Draw(image1).line([(0, 0), (10, 0)])

    image2 = Image.new("1", size=(20, 10))
    ImageDraw.Draw(image2).line([(0, 0), (0, 10)])

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=image1) != hasher.hash_kwargs(
        "blake3", image=image2
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hash_collision_tensor_shape(dtype):
    # The hash should be different though the data is the same when flattened
    arr1 = torch.zeros((5, 10, 20, 3), dtype=dtype)
    arr2 = torch.zeros((10, 20, 5, 3), dtype=dtype)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", data=arr1) != hasher.hash_kwargs(
        "blake3", data=arr2
    )


def test_hash_collision_array_shape():
    # The hash should be different though the data is the same when flattened
    arr1 = np.zeros((5, 10, 20, 3))
    arr2 = np.zeros((10, 20, 5, 3))

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", data=arr1) != hasher.hash_kwargs(
        "blake3", data=arr2
    )


def test_hash_collision_video_num_frames():
    source = b"x" * 100

    def item_for_hash(num_frames: int):
        frames: np.ndarray = np.zeros((num_frames, 8, 8, 3), dtype=np.uint8)
        metadata = {
            "total_num_frames": 16,
            "fps": 2.0,
            "duration": 8.0,
            "video_backend": "opencv",
            "frames_indices": list(range(num_frames)),
            "do_sample_frames": False,
        }
        video = MediaWithBytes((frames, metadata), source)
        items = MultiModalDataParser()._parse_video_data([video])
        return items.get_all_items_for_hash()[0]

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", video=item_for_hash(2)) != hasher.hash_kwargs(
        "blake3", video=item_for_hash(4)
    )


def test_hash_non_contiguous_array():
    arr = np.arange(24).reshape(4, 6).T
    assert not arr.flags.c_contiguous

    arr_c = np.ascontiguousarray(arr)
    assert arr_c.flags.c_contiguous

    hasher = MultiModalHasher
    # Both should be hashable and produce the same hashes
    assert hasher.hash_kwargs("blake3", data=arr) == hasher.hash_kwargs(
        "blake3", data=arr_c
    )


def test_hash_image_exif_id_does_not_override_content():
    image1 = Image.new("RGB", size=(10, 20), color=(255, 0, 0))
    ImageDraw.Draw(image1).line([(0, 0), (9, 0)], fill=(0, 255, 0))
    image2 = Image.new("RGB", size=(10, 20), color=(0, 0, 255))
    ImageDraw.Draw(image2).line([(0, 0), (0, 19)], fill=(255, 255, 0))
    exif_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
    image1.getexif()[Image.ExifTags.Base.ImageID] = exif_id
    image2.getexif()[Image.ExifTags.Base.ImageID] = exif_id

    image1_bytes = BytesIO()
    image1.save(image1_bytes, format="PNG")
    image2_bytes = BytesIO()
    image2.save(image2_bytes, format="PNG")

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=image1) != hasher.hash_kwargs(
        "blake3", image=image2
    )
    assert hasher.hash_kwargs(
        "blake3", image=MediaWithBytes(image1, image1_bytes.getvalue())
    ) != hasher.hash_kwargs(
        "blake3", image=MediaWithBytes(image2, image2_bytes.getvalue())
    )


def _rgba_png_bytes() -> bytes:
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_hash_collision_media_io_config():
    data = _rgba_png_bytes()
    white = ImageMediaIO(rgba_background_color=(255, 255, 255)).load_bytes(data)
    black = ImageMediaIO(rgba_background_color=(0, 0, 0)).load_bytes(data)
    white2 = ImageMediaIO(rgba_background_color=(255, 255, 255)).load_bytes(data)
    keep = ImageMediaIO(image_mode=None).load_bytes(data)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=white) != hasher.hash_kwargs(
        "blake3", image=black
    )
    assert hasher.hash_kwargs("blake3", image=white) != hasher.hash_kwargs(
        "blake3", image=keep
    )
    assert hasher.hash_kwargs("blake3", image=white) == hasher.hash_kwargs(
        "blake3", image=white2
    )


def test_hash_media_io_noop_config_preserves_hash():
    image = Image.new("RGB", (8, 8), (0, 128, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    loaded = ImageMediaIO().load_bytes(data)
    assert loaded.io_config is None

    plain = MediaWithBytes(loaded.media, data)
    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", image=loaded) == hasher.hash_kwargs(
        "blake3", image=plain
    )
