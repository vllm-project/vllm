# SPDX-License-Identifier: Apache-2.0
"""Tests for H2OVL's multimodal preprocessing kwargs."""
from typing import Optional

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.utils import cached_get_tokenizer

from ....conftest import _ImageAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", [
    "h2oai/h2ovl-mississippi-800m",
    "h2oai/h2ovl-mississippi-2b",
])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("max_dynamic_patch", [1, 2, 4, 8])
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    model_id: str,
    image_assets: _ImageAssets,
    size_factors: list[int],
    max_dynamic_patch: int,
    dynamic_image_size: Optional[bool],
    num_imgs: int,
):
    from vllm.model_executor.models.h2ovl import (calculate_h2ovl_targets,
                                                  get_h2ovl_target_ratios)

    ctx = build_model_context(
        model_name=model_id,
        tokenizer_name=model_id,
        trust_remote_code=True,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        trust_remote_code=ctx.model_config.trust_remote_code,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config,
        tokenizer=tokenizer,
    )

    config = processor.info.get_hf_config()
    use_msac = config.use_msac

    mm_processor_kwargs = {
        "max_dynamic_patch": max_dynamic_patch,
    }
    if dynamic_image_size is not None:
        mm_processor_kwargs["dynamic_image_size"] = dynamic_image_size

    min_num = config.min_dynamic_patch
    max_num = max_dynamic_patch if dynamic_image_size else 1

    # Build the image str / prompt based on the number of images we pass
    prompt = "<image>" * num_imgs

    for asset in image_assets:
        for factor in size_factors:
            image = rescale_image_size(asset.pil_image, factor)
            mm_data = {"image": [image] * num_imgs}

            width, height = image.size

            # Calculate the expected number of blocks
            if num_imgs == 1 and use_msac:
                # First pass
                blocks1, _, _, aspect_ratio = calculate_h2ovl_targets(
                    orig_width=width,
                    orig_height=height,
                    target_ratios=get_h2ovl_target_ratios(
                        min_num,
                        max_num,
                        prior_aspect_ratio=None,
                    ),
                    image_size=config.vision_config.image_size,
                    use_thumbnail=False,  # Thumbnail is handled separately
                )

                # Second pass
                blocks2, _, _, _ = calculate_h2ovl_targets(
                    orig_width=width,
                    orig_height=height,
                    target_ratios=get_h2ovl_target_ratios(
                        min_num,
                        max_num,
                        prior_aspect_ratio=aspect_ratio,
                    ),
                    image_size=config.vision_config.image_size,
                    use_thumbnail=False,
                )

                # Add thumbnail if use_thumbnail is True and total_blocks > 1
                if config.use_thumbnail:
                    blocks1 += 1 if blocks1 > 1 else 0
                    blocks2 += 1 if blocks2 > 1 else 0

                # Total blocks is the sum of blocks from both passes minus
                # overlapping
                total_blocks = blocks1 + blocks2 - 1

                expected_num_patches = total_blocks
            else:
                blocks, _, _, _ = calculate_h2ovl_targets(
                    orig_width=width,
                    orig_height=height,
                    target_ratios=get_h2ovl_target_ratios(
                        min_num,
                        max_num,
                        prior_aspect_ratio=None,
                    ),
                    image_size=config.vision_config.image_size,
                    use_thumbnail=False,
                )
                expected_num_patches = blocks

                if config.use_thumbnail and expected_num_patches != 1:
                    expected_num_patches += 1

            processed_inputs = processor.apply(prompt, mm_data,
                                               mm_processor_kwargs)
            pixel_shape = (
                processed_inputs["mm_kwargs"]["pixel_values_flat"].shape)

            assert pixel_shape[0] == expected_num_patches * num_imgs
