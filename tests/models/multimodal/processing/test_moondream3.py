# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Moondream3 multimodal processing."""

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MOONDREAM3_MODEL_ID = "moondream/moondream3-preview"
# Expected tokens: 729 (27x27 patches from 378x378 crop / 14 patch size)
EXPECTED_IMAGE_TOKENS = 729


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_creation(model_id: str):
    """Test that Moondream3 processor can be created."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert processor is not None


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_apply(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that Moondream3 processor can process inputs.

    NOTE: The prompt must have a space after <image> to ensure correct
    tokenization: '<image> ' not '<image>\\n'. This is because the
    tokenizer treats '<image>' differently based on following context.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Use space after <image> to ensure correct tokenization
    prompt = "<image> \n\nQuestion: What is this?\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    assert "prompt_token_ids" in processed_inputs
    # Token count should be close to 729 (image) + text tokens
    assert len(processed_inputs["prompt_token_ids"]) >= EXPECTED_IMAGE_TOKENS


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_pixel_values(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that pixel values are correctly produced."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image> \n\nQuestion: What is this?\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Check mm_kwargs contains pixel_values
    mm_kwargs = processed_inputs.get("mm_kwargs")
    assert mm_kwargs is not None
    mm_data_result = mm_kwargs.get_data()
    assert "pixel_values" in mm_data_result

    # Verify pixel_values shape
    pixel_values = mm_data_result["pixel_values"]
    assert pixel_values.dim() == 5  # [batch, num_crops, C, H, W]
    assert pixel_values.shape[2] == 3  # RGB channels
    assert pixel_values.shape[3] == 378  # crop height
    assert pixel_values.shape[4] == 378  # crop width


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_image_token_expansion(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that <image> placeholder is expanded to correct number of tokens."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.tokenizer

    # The placeholder tokens for <image>
    placeholder_tokens = tokenizer.encode("<image>", add_special_tokens=False)
    # Should be [48, 4737, 50] = ['<', 'image', '>']
    assert len(placeholder_tokens) == 3

    prompt = "<image> \n\nQuestion: Describe.\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})
    token_ids = processed_inputs["prompt_token_ids"]

    # Count occurrences of the first placeholder token (used as replacement)
    # The <image> should be expanded to 729 tokens
    first_placeholder_token = placeholder_tokens[0]  # '<' token
    count = token_ids.count(first_placeholder_token)

    # Should have 729 image tokens
    assert count == EXPECTED_IMAGE_TOKENS, (
        f"Expected {EXPECTED_IMAGE_TOKENS} image tokens, got {count}"
    )
