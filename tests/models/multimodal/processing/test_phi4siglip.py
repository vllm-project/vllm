# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context

MODEL_ID = "microsoft/Phi-4-reasoning-vision-15B"
IMAGE_PLACEHOLDER = "<image>"

# (patch_size, min_num_patches, max_num_patches) = (16, 256, 3600), so a 960x960
# image is exactly max_num_patches and a 256x256 one is exactly min_num_patches.
MAX_NUM_PATCHES = 3600
PATCH_FEATURES = 768  # patch_size**2 * 3


@pytest.fixture(scope="module")
def processor():
    ctx = build_model_context(MODEL_ID, limit_mm_per_prompt={"image": 2})
    return MULTIMODAL_REGISTRY.create_processor(ctx.model_config)


def _process(processor, images: list[Image.Image]):
    return processor(
        IMAGE_PLACEHOLDER * len(images),
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )


@pytest.mark.parametrize(
    ("size", "expected_shape"),
    [
        ((128, 128), (16, 16)),  # 64 patches -> upscaled to min_num_patches
        ((700, 500), (31, 43)),  # 1333 patches -> kept at native resolution
        ((1024, 1024), (60, 60)),  # 4096 patches -> downscaled to max_num_patches
    ],
)
def test_resize_branches(processor, size, expected_shape):
    """Images below/within/above the patch budget resize to the right grid."""
    processed = _process(processor, [Image.new("RGB", size)])
    mm_data = processed["mm_kwargs"].get_data()

    spatial_shapes = mm_data["spatial_shapes"]
    assert tuple(spatial_shapes[0].tolist()) == expected_shape

    # The vision tower packs patches using spatial_shapes but slices them using
    # pixel_attention_mask, so the two must agree.
    expected_tokens = expected_shape[0] * expected_shape[1]
    assert int(mm_data["pixel_attention_mask"][0].sum()) == expected_tokens
    assert [p.length for p in processed["mm_placeholders"]["image"]] == [
        expected_tokens
    ]


@pytest.mark.parametrize("num_images", [1, 2])
def test_patches_are_padded_to_a_constant(processor, num_images):
    """Differently sized images still stack, as batched fields require."""
    sizes = [(128, 128), (700, 500)][:num_images]
    processed = _process(processor, [Image.new("RGB", size) for size in sizes])

    pixel_values = processed["mm_kwargs"].get_data()["pixel_values"]
    assert pixel_values.shape == (num_images, MAX_NUM_PATCHES, PATCH_FEATURES)
