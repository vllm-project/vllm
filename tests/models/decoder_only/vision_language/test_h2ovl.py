from typing import Optional, Tuple

import pytest
import torch
from PIL.Image import Image
from transformers import AutoConfig

# Import the functions to test
from vllm.model_executor.models.h2ovl import (calculate_num_blocks,
                                              image_to_pixel_values_wrapper)
from vllm.multimodal.utils import rescale_image_size

models = [
    "h2oai/h2ovl-mississippi-800m",  # Replace with your actual model names
    "h2oai/h2ovl-mississippi-2b",
]


def run_preprocessing_test(
    image: Image,
    config,
    max_dynamic_patch: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Test the image preprocessing and calculate expected blocks."""

    if max_dynamic_patch is None:
        max_dynamic_patch = config.max_dynamic_patch

    width, height = image.size
    use_MSAC = config.use_msac

    # Create the mapper function with the provided configuration
    mapper = image_to_pixel_values_wrapper(config, max_dynamic_patch, use_MSAC)
    pixel_values = mapper(image)

    # Calculate the expected number of blocks
    if use_MSAC:
        # First pass
        blocks1, _, _, aspect_ratio = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,  # Thumbnail is handled separately
            prior_aspect_ratio=None,
        )

        # Second pass
        blocks2, _, _, _ = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,
            prior_aspect_ratio=aspect_ratio,
        )

        # Add thumbnail if use_thumbnail is True and total_blocks > 1
        if config.use_thumbnail:
            blocks1 += 1 if blocks1 > 1 else 0
            blocks2 += 1 if blocks2 > 1 else 0

        # Total blocks is the sum of blocks from both passes minus overlapping
        total_blocks = blocks1 + blocks2 - 1

        expected_blocks = total_blocks

    else:
        blocks, _, _, _ = calculate_num_blocks(
            width,
            height,
            config.min_dynamic_patch,
            max_dynamic_patch,
            config.vision_config.image_size,
            use_thumbnail=False,
            prior_aspect_ratio=None,
        )
        expected_blocks = blocks

        if config.use_thumbnail and expected_blocks > 1:
            expected_blocks += 1

    return pixel_values, expected_blocks


@pytest.mark.parametrize("model_name", models)
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
@pytest.mark.parametrize("max_dynamic_patch", [None, 2, 4, 8])
def test_image_preprocessing(image_assets, model_name, size_factors,
                             max_dynamic_patch):
    """Test image preprocessing pipeline with different configurations."""
    # Load the configuration from the model
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    for asset in image_assets:
        image = asset.pil_image
        for factor in size_factors:
            scaled_image = rescale_image_size(image, factor)

            # Test preprocessing and get expected number of blocks
            pixel_values, expected_blocks = run_preprocessing_test(
                scaled_image, config, max_dynamic_patch)

            # Verify output shapes and properties
            actual_blocks = pixel_values.shape[0]
            assert actual_blocks == expected_blocks, (
                f"Expected {expected_blocks} blocks, got {actual_blocks}")

            # Check image dimensions
            expected_size = (
                3,  # Number of channels (C, H, W)
                config.vision_config.image_size,
                config.vision_config.image_size,
            )
            for img in pixel_values:
                assert img.shape == expected_size, (
                    f"Expected image size {expected_size}, got {img.shape}")
