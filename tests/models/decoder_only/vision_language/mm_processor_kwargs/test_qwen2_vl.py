from typing import Any, Dict, Tuple

import pytest
from transformers import AutoTokenizer

from vllm.inputs import InputProcessingContext

from .....conftest import _ImageAssets
from ....utils import build_model_context

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = "min_pixels"
MAX_PIXELS = "max_pixels"


# Fixtures lazy import to avoid initializing CUDA during test collection
# NOTE: Qwen2VL supports multiple input modalities, so it registers multiple
# input mappers.
@pytest.fixture()
def processor_for_qwen2_vl():
    from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
    return Qwen2VLMultiModalProcessor


@pytest.mark.parametrize(
    "mm_processor_kwargs, expected_toks_per_img, expected_pixels_shape", [
        ({}, 1426, (5704, 1176)),
        ({
            MIN_PIXELS: 64**2,
            MAX_PIXELS: 512**2
        }, 330, (1320, 1176)),
    ])
@pytest.mark.parametrize("model", [MODEL])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    processor_for_qwen2_vl,
    image_assets: _ImageAssets,
    model: str,
    mm_processor_kwargs: Dict[str, Any],
    expected_toks_per_img: int,
    expected_pixels_shape: Tuple[int, int],
    num_imgs: int,
):
    """Ensure Qwen2VLMultiModalProcessor handles min/max pixels properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ctx = InputProcessingContext(ctx.model_config, tokenizer)
    # Build the image str / prompt based on the number of images we pass
    prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
    images = [image_assets[0].pil_image] * num_imgs

    mm_data = {"image": images}

    processor = processor_for_qwen2_vl(ctx)
    processed_inputs = processor.apply(prompt, mm_data, mm_processor_kwargs)

    # Ensure we have the right number of placeholders per num_crops size
    hf_processor = processor._get_hf_processor(**mm_processor_kwargs)
    image_token_id = tokenizer.convert_tokens_to_ids(hf_processor.image_token)
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_shape = processed_inputs["mm_kwargs"]["pixel_values"].shape

    assert img_tok_count == expected_toks_per_img * num_imgs
    assert pixel_shape[0] == expected_pixels_shape[0] * num_imgs
    assert pixel_shape[1] == expected_pixels_shape[1]
