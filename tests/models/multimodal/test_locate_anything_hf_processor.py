# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processor-level contract test for LocateAnything.

These tests guard the single most load-bearing assumption of the vLLM
integration: the HF ``LocateAnythingProcessor`` only emits ``pixel_values`` /
``image_grid_hws`` for images that are referenced by a *numbered* ``<image-N>``
placeholder in the text. An image passed without a matching placeholder is
silently dropped, so the vision encoder never runs and the model "works" with
no image actually entering it.

`get_placeholder_str` / `get_dummy_text` therefore MUST produce ``<image-N>``
placeholders (not ``<img><IMG_CONTEXT></img>`` nor a bare ``<IMG_CONTEXT>``).

Requires network access + ``trust_remote_code`` to pull the processor from
``nvidia/LocateAnything-3B``; it does NOT need the model weights or a GPU.
"""

import pytest

transformers = pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

MODEL_ID = "nvidia/LocateAnything-3B"
IMG_CONTEXT = "<IMG_CONTEXT>"


def _load_processor():
    try:
        return transformers.AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
    except Exception as exc:  # network / auth / repo unavailable
        pytest.skip(f"could not load {MODEL_ID} processor: {exc}")


def _dummy_image(width: int = 448, height: int = 448) -> Image.Image:
    return Image.new("RGB", (width, height), color=(127, 127, 127))


def test_numbered_placeholder_produces_pixel_values():
    """The fixed placeholder form must make the processor emit vision inputs."""
    processor = _load_processor()
    image = _dummy_image()

    out = processor(text=["<image-1>"], images=[image], return_tensors="pt")

    # The core regression: vision inputs must be present and non-empty.
    assert "pixel_values" in out, "processor dropped the image (no pixel_values)"
    assert out["pixel_values"].shape[0] > 0
    assert "image_grid_hws" in out
    assert out["image_grid_hws"].shape[0] == 1


def test_wrong_placeholder_drops_the_image():
    """Documents the bug: the old placeholder form yields no vision inputs.

    This is the exact failure mode the fix addresses; if a future processor
    revision changes this behaviour the test will flag it for re-evaluation.
    """
    processor = _load_processor()
    image = _dummy_image()

    out = processor(
        text=["<img><IMG_CONTEXT></img>"], images=[image], return_tensors="pt"
    )

    has_pixels = "pixel_values" in out and out["pixel_values"].shape[0] > 0
    assert not has_pixels, (
        "processor unexpectedly produced pixel_values for the non-numbered "
        "placeholder; the vLLM get_placeholder_str contract may need revisiting"
    )


def test_placeholder_token_count_matches_grid():
    """Number of expanded <IMG_CONTEXT> tokens must equal (h*w)//merge**2.

    This is the count vLLM's `_get_prompt_updates` replaces, so it must match
    what the projector ultimately emits, or placeholder/feature counts diverge.
    """
    processor = _load_processor()
    image = _dummy_image()

    out = processor(text=["<image-1>"], images=[image], return_tensors="pt")

    grid_h, grid_w = (int(v) for v in out["image_grid_hws"][0])
    merge = processor.image_processor.merge_kernel_size
    expected_tokens = (grid_h * grid_w) // (merge[0] * merge[1])

    # Decode the expanded text and count the IMG_CONTEXT placeholders.
    decoded = processor.tokenizer.decode(out["input_ids"][0])
    assert decoded.count(IMG_CONTEXT) == expected_tokens
    assert expected_tokens > 0
