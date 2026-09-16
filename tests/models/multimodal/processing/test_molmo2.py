# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image as PILImage
from PIL import ImageOps

from vllm.config.multimodal import VideoDummyOptions
from vllm.model_executor.models.molmo2 import (
    Molmo2DummyInputsBuilder,
    build_flat_image_bool_length,
    exif_transpose,
)
from vllm.multimodal.media.image import ImageMediaIO


def test_build_flat_image_bool_length_matches_molmoweb_processor_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[14, 14, 14, 23]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=True,
        use_single_crop_col_tokens=None,
        use_single_crop_start_token=False,
    )

    assert num_image_tokens.tolist() == [550]
    assert len(image_tokens) == 550
    assert image_tokens[0].item() == hf_config.image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 28


def test_build_flat_image_bool_length_respects_disabled_col_tokens():
    hf_config = SimpleNamespace(
        image_patch_id=151938,
        low_res_image_start_token_id=151940,
        image_start_token_id=151936,
        image_col_id=151939,
        image_end_token_id=151937,
    )
    image_grids = torch.tensor([[2, 3, 5, 7]], dtype=torch.long)

    image_tokens, num_image_tokens = build_flat_image_bool_length(
        image_grids,
        hf_config,
        image_use_col_tokens=False,
        use_single_crop_col_tokens=False,
        use_single_crop_start_token=True,
    )

    assert num_image_tokens.tolist() == [45]
    assert len(image_tokens) == 45
    assert image_tokens[0].item() == hf_config.low_res_image_start_token_id
    assert (image_tokens == hf_config.image_col_id).sum().item() == 0


@pytest.mark.parametrize(
    ("num_frames_override", "expected_frames"),
    [(1, 2), (2, 2), (3, 3)],
)
def test_dummy_video_num_frames_override_honors_min_of_two(
    num_frames_override, expected_frames
):
    """A ``num_frames`` override below 2 must be ignored (the model needs at
    least 2 frames), matching the "cannot be less than 2, will be ignored"
    warning."""
    builder = object.__new__(Molmo2DummyInputsBuilder)
    builder.info = SimpleNamespace(
        get_hf_processor=lambda: SimpleNamespace(
            video_processor=SimpleNamespace(size={"width": 64, "height": 64})
        ),
        get_num_frames_with_most_features=lambda seq_len, mm_counts: 16,
        get_image_size_with_most_features=lambda: (64, 64),
    )

    data = builder.get_dummy_mm_data(
        seq_len=128,
        mm_counts={"image": 0, "video": 1},
        mm_options={"video": VideoDummyOptions(num_frames=num_frames_override)},
    )

    video, _metadata = data["video"][0]
    assert video.shape[0] == expected_frames


def _malformed_exif_jpeg() -> bytes:
    """A JPEG whose APP1/Exif segment carries an invalid TIFF header."""
    buf = BytesIO()
    PILImage.new("RGB", (64, 48)).save(buf, "JPEG")
    jpg = buf.getvalue()
    rest = jpg[2:]
    rest = rest[2 + struct.unpack(">H", rest[2:4])[0] :]
    payload = b"Exif\x00\x00XXXX\x00\x00\x00\x08" + bytes(32)
    return b"\xff\xd8\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + rest


def _served_image() -> PILImage.Image:
    """An image with malformed EXIF, loaded the way a served request loads it.

    ``ImageMediaIO.load_bytes`` runs a suppressed ``exif_transpose`` and then
    ``load()``, which is what primes Pillow's lazy EXIF state so the *next*
    read of it raises.
    """
    return ImageMediaIO(image_mode="RGB").load_bytes(_malformed_exif_jpeg()).media


def test_exif_transpose_tolerates_malformed_exif():
    """A malformed EXIF header must not fail the request in the prompt update.

    ``get_image_replacement_molmo2`` transposes the served image again, so a
    bare ``ImageOps.exif_transpose`` turns it into a ``SyntaxError`` escaping the
    multi-modal processor. This is the same payload #56576 made
    ``MultiModalHasher`` tolerate; hashing runs first, so that fix moved the
    failure here rather than removing it.
    """
    # Precondition, on its own image: Pillow raises only on the first read of
    # the poisoned state, so every case below needs a freshly loaded image or it
    # would pass by coincidence.
    with pytest.raises(Exception):  # noqa: B017 - Pillow raises SyntaxError
        ImageOps.exif_transpose(_served_image())

    assert exif_transpose(_served_image()) is not None
    assert exif_transpose([_served_image(), None])[0] is not None
    assert exif_transpose(None) is None
