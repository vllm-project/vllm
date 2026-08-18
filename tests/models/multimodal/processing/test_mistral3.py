# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing kwargs."""

import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


def _expected_placeholder_tokens_per_image(
    hf_processor,
    image: Image.Image,
) -> int:
    """Ground-truth placeholder count derived from the HF processor output.

    The HF PixtralImageProcessor emits `pixel_values` whose spatial dims are a
    multiple of `patch_size * spatial_merge_size`. The Mistral3PatchMerger then
    unfolds each (spatial_merge_size x spatial_merge_size) block into a single
    output token, so the number of placeholders per image equals
    (H // effective_patch) * (W // effective_patch).
    """
    hf_out = hf_processor(text="", images=[image], return_tensors="pt")
    pixel_values = hf_out["pixel_values"]
    if isinstance(pixel_values, list):
        # Recent transformers releases return a list for variable-size inputs;
        # `image_sizes` carries the un-padded per-image (H, W).
        image_h, image_w = hf_out["image_sizes"][0].tolist()
    else:
        image_h, image_w = pixel_values.shape[-2:]

    image_processor = hf_processor.image_processor
    patch_size = image_processor.patch_size
    if isinstance(patch_size, dict):
        patch_h = patch_size["height"]
        patch_w = patch_size["width"]
    else:
        patch_h = patch_w = int(patch_size)

    spatial_merge_size = getattr(image_processor, "spatial_merge_size", 1)
    effective_h = patch_h * spatial_merge_size
    effective_w = patch_w * spatial_merge_size
    return (image_h // effective_h) * (image_w // effective_w)


@pytest.mark.parametrize("model_id", ["mistralai/Mistral-Small-3.1-24B-Instruct-2503"])
@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {},
        {"size": {"longest_edge": 1008}},
        {"size": {"longest_edge": 1288}},
    ],
)
@pytest.mark.parametrize(
    "image_size",
    [
        (1540, 1540),
        (1536, 1187),
        (2200, 1700),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_placeholder_matches_processed_pixel_values(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    image_size: tuple[int, int],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Placeholder count must follow the HF-processed pixel_values geometry.

    Regression test for the case where `size={"longest_edge": N}` in
    `mm_processor_kwargs` diverges from `vision_config.image_size`: the old
    path computed ncols/nrows against the pretrained RoPE capacity and produced
    a placeholder count that no longer matched the vision-tower output.
    """
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)

    dummy_image = Image.new("RGB", image_size, color=(127, 127, 127))
    prompt = hf_processor.image_token * num_imgs
    mm_data = {"image": [dummy_image] * num_imgs}

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    hf_config = processor.info.get_hf_config()
    image_token_id = hf_config.image_token_index
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)

    expected_per_image = _expected_placeholder_tokens_per_image(
        hf_processor, dummy_image
    )
    assert img_tok_count == expected_per_image * num_imgs
