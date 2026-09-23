# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import hashlib
import struct
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageOps

from vllm.config.multimodal import MMHasherAlgorithm
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.media.base import DecodeSpec, MediaRef
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
        video = MediaRef(
            lambda: (frames, {}), source, DecodeSpec({"num_frames": num_frames})
        )
        items = MultiModalDataParser()._parse_video_data([video])
        item = items.get_all_raw()[0]
        assert not video.is_decoded
        return item

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", video=item_for_hash(2)) != hasher.hash_kwargs(
        "blake3", video=item_for_hash(4)
    )


def test_hash_video_tensor_frames():
    """Videos holding tensor frames (e.g. NVDEC-decoded) hash like
    array-framed ones: the key comes from the encoded bytes and the decode
    spec, so neither a D2H copy nor a decode is needed."""
    source = b"x" * 100
    spec = DecodeSpec({"video_backend": "torchcodec"})

    def item_for_hash(frames):
        video = MediaRef(lambda: (frames, {}), source, spec)
        items = MultiModalDataParser()._parse_video_data([video])
        item = items.get_all_raw()[0]
        assert not video.is_decoded
        return item

    np_frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    torch_frames = torch.zeros((2, 8, 8, 3), dtype=torch.uint8)

    hasher = MultiModalHasher
    assert hasher.hash_kwargs("blake3", video=item_for_hash(np_frames)) == (
        hasher.hash_kwargs("blake3", video=item_for_hash(torch_frames))
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
    assert hasher.hash_kwargs("blake3", image=image1) == hasher.hash_kwargs(
        "blake3", image=id.bytes
    )
    # second image has non-UUID in ImageID, so it should hash to the image data
    assert hasher.hash_kwargs("blake3", image=image2) == hasher.hash_kwargs(
        "blake3", image=image2a
    )


def test_hash_image_malformed_exif():
    # Test that images with malformed EXIF headers (e.g. invalid TIFF header)
    # do not raise an unhandled exception during hashing and fall back to image data.
    buf = BytesIO()
    Image.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    rest = jpg[2:]
    rest = rest[2 + struct.unpack(">H", rest[2:4])[0] :]
    payload = b"Exif\x00\x00XXXX\x00\x00\x00\x08" + bytes(32)
    data = b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + rest

    image = Image.open(BytesIO(data))
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    image.load()

    hasher = MultiModalHasher
    # Should hash without raising SyntaxError or any other exception
    hash_val = hasher.hash_kwargs("blake3", image=image)
    assert isinstance(hash_val, str) and len(hash_val) > 0

    # Also verify a ref built from those bytes: the EXIF probe tolerates the
    # malformed header instead of raising at fetch time.
    media_item = ImageMediaIO().load_bytes_ref(data)
    hash_media = hasher.hash_kwargs("blake3", image=media_item)
    assert isinstance(hash_media, str) and len(hash_media) > 0
    assert not media_item.is_decoded


def _rgba_png_bytes() -> bytes:
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_hash_collision_media_io_config():
    """Decode settings that mutate the media are part of its cache identity."""
    data = _rgba_png_bytes()
    white = ImageMediaIO(rgba_background_color=(255, 255, 255)).load_bytes_ref(data)
    black = ImageMediaIO(rgba_background_color=(0, 0, 0)).load_bytes_ref(data)
    white2 = ImageMediaIO(rgba_background_color=(255, 255, 255)).load_bytes_ref(data)
    keep = ImageMediaIO(image_mode=None).load_bytes_ref(data)

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


def test_hash_media_io_noop_config_still_scopes_key():
    """A setting that happens not to change this payload still scopes the key.

    `image_mode="RGB"` is a no-op for an already-RGB PNG, but the ref cannot
    know that without decoding, so the two configurations stay distinct.
    """
    image = Image.new("RGB", (8, 8), (0, 128, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    convert = ImageMediaIO().load_bytes_ref(data)
    keep = ImageMediaIO(image_mode=None).load_bytes_ref(data)

    assert convert.key != keep.key
    assert np.array_equal(np.array(convert.decode()), np.array(keep.decode()))


# The digest input is a concatenation of byte chunks, so it has to be uniquely
# decodable. Each case below is a pair of distinct processor kwargs that used to
# serialize to the same bytes, which made two requests share an mm hash -- and
# therefore share both the processor cache entry and the prefix-cache block key.
IMAGE = b"\x89PNG\r\n\x1a\n"


def _hash(**mm_processor_kwargs: object) -> str:
    return MultiModalHasher.hash_kwargs(
        "blake3", model_id="m", image=IMAGE, **mm_processor_kwargs
    )


def test_hash_collision_kwargs_key_value_boundary():
    # Both used to flatten to b"ab" + b"c".
    assert _hash(**{"ab": "c"}) != _hash(**{"a": "bc"})


def test_hash_collision_nested_vs_flattened_key():
    nested = _hash(size={"shortest_edge": 224})
    flattened = _hash(**{"size.shortest_edge": 224})
    assert nested != flattened


def test_hash_collision_sequence_vs_mapping():
    assert _hash(fps=[2, 4]) != _hash(fps={"0": 2, "1": 4})


@pytest.mark.parametrize("empty", ["", b"", []])
def test_hash_collision_none_vs_empty(empty):
    assert _hash(video_pruning_rate=None) != _hash(video_pruning_rate=empty)


def test_hash_collision_empty_container_vs_omitted():
    omitted = _hash()
    assert _hash(size={}) != omitted
    assert _hash(size=[]) != omitted
    assert _hash(size={}) != _hash(size=[])


class _CountingDecoder:
    """Decoder that records how many times it ran."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def test_hash_media_ref_does_not_decode():
    """Hashing a ref uses its precomputed key and never decodes it."""
    decoder = _CountingDecoder(np.zeros((4,), dtype=np.float32))
    item = MediaRef(decoder, b"audio-bytes")

    hasher = MultiModalHasher
    hash_ref = hasher.hash_kwargs("blake3", audio=item)

    assert decoder.calls == 0
    assert hash_ref == hasher.hash_kwargs("blake3", audio=item.key)


def test_hash_media_ref_survives_release():
    """The key is derived up front, so hashing still works after release()."""
    item = MediaRef(lambda: np.zeros((4,), dtype=np.float32), b"raw")
    hasher = MultiModalHasher
    before = hasher.hash_kwargs("blake3", video=item)

    item.release()

    assert item.data == b""
    assert hasher.hash_kwargs("blake3", video=item) == before
