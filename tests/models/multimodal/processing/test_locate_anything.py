# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LocateAnything's multimodal processing.

These tests guard the most load-bearing assumption of the vLLM
integration: the HF ``LocateAnythingProcessor`` only emits
``pixel_values`` / ``image_grid_hws`` for images referenced by a
*numbered* ``<image-N>`` placeholder in the text. An image passed
without a matching placeholder is silently dropped, so the vision
encoder never runs and the model "works" with no image entering it.

``get_placeholder_str`` / ``get_dummy_text`` therefore MUST produce
``<image-N>`` placeholders (not ``<img><IMG_CONTEXT></img>`` nor a bare
``<IMG_CONTEXT>``).

The processor download needs network access + ``trust_remote_code``
(handled by ``build_model_context`` via the model registry); no model
weights or GPU required.
"""

import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "nvidia/LocateAnything-3B"
IMG_CONTEXT = "<IMG_CONTEXT>"


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"image": 1})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


@pytest.fixture(scope="module")
def hf_processor(processor):
    return processor.info.get_hf_processor()


def _dummy_image(width: int = 448, height: int = 448) -> Image.Image:
    return Image.new("RGB", (width, height), color=(127, 127, 127))


# ---------------------------------------------------------------------------
# Token-count math (real ProcessingInfo backed by the real HF image processor)
# ---------------------------------------------------------------------------


def test_image_processor_geometry_assumptions(hf_processor):
    # The expected token counts below are derived from these values; if
    # the checkpoint's image processor config changes, re-derive them.
    image_processor = hf_processor.image_processor
    assert image_processor.patch_size == 14
    assert tuple(image_processor.merge_kernel_size) == (2, 2)
    assert image_processor.in_token_limit == 25600


def test_num_image_tokens_basic(processor):
    # 448x448 -> 32x32 patches -> /(2*2) -> 16x16 = 256 tokens
    n = processor.info.get_num_image_tokens(image_width=448, image_height=448)
    assert n == 256


def test_num_image_tokens_pads_to_merge_multiple(processor):
    # 430 not a multiple of 28 -> pad up to 448 -> 16x16 = 256
    n = processor.info.get_num_image_tokens(image_width=430, image_height=430)
    assert n == 256


def test_num_image_tokens_exact_multiple_no_pad(processor):
    # 420 already a multiple of 28 -> 15x15 = 225 (no padding)
    n = processor.info.get_num_image_tokens(image_width=420, image_height=420)
    assert n == 225


def test_num_image_tokens_large_image_pads_only(processor):
    # 1024x1024 -> 73*73 = 5329 patches, still under in_token_limit
    # (25600): no downscale, just pad 1022 -> 1036 -> 37x37 = 1369.
    n = processor.info.get_num_image_tokens(image_width=1024, image_height=1024)
    assert n == 1369


def test_num_image_tokens_downscales_above_limit(processor):
    # 3000x3000 -> 214*214 = 45796 patches > in_token_limit (25600), so
    # the image is downscaled (math.sqrt branch) to exactly the limit
    # (scale = 160/214 -> 2242px -> 160x160 patches), then padded to a
    # merge multiple: 2268 -> 81x81 = 6561 tokens. The smaller cases
    # never reach this branch, which is the most error-prone part of the
    # token-count math (a mismatch with the HF image processor's resize
    # would desync placeholder count vs. vision-tower output length).
    n = processor.info.get_num_image_tokens(image_width=3000, image_height=3000)
    assert n == 6561


# ---------------------------------------------------------------------------
# HF processor placeholder contract
# ---------------------------------------------------------------------------


def test_numbered_placeholder_produces_pixel_values(hf_processor):
    """The fixed placeholder form must make the processor emit vision
    inputs."""
    out = hf_processor(text=["<image-1>"], images=[_dummy_image()], return_tensors="pt")

    # The core regression: vision inputs must be present and non-empty.
    assert "pixel_values" in out, "processor dropped the image"
    assert out["pixel_values"].shape[0] > 0
    assert "image_grid_hws" in out
    assert out["image_grid_hws"].shape[0] == 1


def test_wrong_placeholder_drops_the_image(hf_processor):
    """Documents the failure mode: a non-numbered placeholder yields no
    vision inputs.

    If a future processor revision changes this behaviour, this test
    flags the ``get_placeholder_str`` contract for re-evaluation.
    """
    out = hf_processor(
        text=["<img><IMG_CONTEXT></img>"],
        images=[_dummy_image()],
        return_tensors="pt",
    )

    has_pixels = "pixel_values" in out and out["pixel_values"].shape[0] > 0
    assert not has_pixels, (
        "processor unexpectedly produced pixel_values for the "
        "non-numbered placeholder; the vLLM get_placeholder_str "
        "contract may need revisiting"
    )


def test_placeholder_token_count_matches_grid(hf_processor):
    """Expanded <IMG_CONTEXT> count must equal (h*w)//merge**2.

    This is the count vLLM's ``_get_prompt_updates`` replaces, so it
    must match what the projector ultimately emits, or
    placeholder/feature counts diverge.
    """
    out = hf_processor(text=["<image-1>"], images=[_dummy_image()], return_tensors="pt")

    grid_h, grid_w = (int(v) for v in out["image_grid_hws"][0])
    merge = hf_processor.image_processor.merge_kernel_size
    expected_tokens = (grid_h * grid_w) // (merge[0] * merge[1])

    # Decode the expanded text and count the IMG_CONTEXT placeholders.
    decoded = hf_processor.tokenizer.decode(out["input_ids"][0])
    assert decoded.count(IMG_CONTEXT) == expected_tokens
    assert expected_tokens > 0
