from typing import Any, Dict, Tuple

import pytest
import torch
from PIL.Image import Image
from transformers import AutoTokenizer

from vllm.inputs import InputContext, token_inputs
from vllm.multimodal import MultiModalRegistry

from .....conftest import _ImageAssets
from ....utils import build_model_context

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MIN_PIXELS = "min_pixels"
MAX_PIXELS = "max_pixels"


# Fixtures lazy import to avoid initializing CUDA during test collection
# NOTE: Qwen2VL supports multiple input modalities, so it registers multiple
# input mappers.
@pytest.fixture()
def image_input_mapper_for_qwen2_vl():
    from vllm.model_executor.models.qwen2_vl import (
        image_input_mapper_for_qwen2_vl)
    return image_input_mapper_for_qwen2_vl


@pytest.fixture()
def input_processor_for_qwen2_vl():
    from vllm.model_executor.models.qwen2_vl import (
        input_processor_for_qwen2_vl)
    return input_processor_for_qwen2_vl


@pytest.fixture()
def qwen2_vl_context() -> InputContext:
    return build_model_context(model_name=MODEL)


@pytest.fixture()
def get_max_qwen2_vl_image_tokens():
    from vllm.model_executor.models.qwen2_vl import (
        get_max_qwen2_vl_image_tokens)
    return get_max_qwen2_vl_image_tokens


@pytest.fixture()
def dummy_data_for_qwen2_vl():
    from vllm.model_executor.models.qwen2_vl import dummy_data_for_qwen2_vl
    return dummy_data_for_qwen2_vl


@pytest.mark.parametrize("mm_processor_kwargs,expected_max_tokens", [
    ({}, 1225),
    ({
        MIN_PIXELS: 64**2,
        MAX_PIXELS: 512**2
    }, 324),
])
def test_qwen2_vl_max_image_tokens(get_max_qwen2_vl_image_tokens,
                                   qwen2_vl_context: InputContext,
                                   mm_processor_kwargs: Dict[str, Any],
                                   expected_max_tokens: int):
    """Ensure that the max token calc handles min/max pixels properly."""
    actual_max_tokens = get_max_qwen2_vl_image_tokens(qwen2_vl_context,
                                                      **mm_processor_kwargs)
    assert actual_max_tokens == expected_max_tokens


@pytest.mark.parametrize("mm_processor_kwargs,token_count,img_size", [
    [{}, 1225, (980, 980)],
    [{
        MIN_PIXELS: 64**2,
        MAX_PIXELS: 512**2
    }, 324, (504, 504)],
])
def test_qwen2_vl_dummy_data(dummy_data_for_qwen2_vl,
                             qwen2_vl_context: InputContext,
                             mm_processor_kwargs: Dict[str, Any],
                             token_count: int, img_size: Tuple[int, int]):
    """Ensure that the dummy data handles min/max pixels properly."""
    seq_len = 3000
    hf_config = qwen2_vl_context.get_hf_config()
    image_token_id = hf_config.image_token_id

    # NOTE: video value is required, but isn't actually used
    # when making the dummy data except for error handling currently
    dummy_data = dummy_data_for_qwen2_vl(
        ctx=qwen2_vl_context,
        seq_len=seq_len,
        mm_counts={
            "image": 1,
            "video": 0
        },
        **mm_processor_kwargs,
    )
    seq_data = dummy_data.seq_data
    mm_data = dummy_data.multi_modal_data

    # Ensure we have the right number of placeholders for min/max pixel values
    assert seq_data.get_token_ids().count(image_token_id) == token_count

    # Ensure the images were resized correctly
    image = mm_data["image"]
    assert isinstance(image, Image)
    assert image.size == img_size


@pytest.mark.parametrize("mm_processor_kwargs,num_placeholders", [
    ({}, 1426),
    ({
        MIN_PIXELS: 64**2,
        MAX_PIXELS: 512**2
    }, 330),
])
def test_input_processor(input_processor_for_qwen2_vl,
                         qwen2_vl_context: InputContext,
                         image_assets: _ImageAssets, num_placeholders: int,
                         mm_processor_kwargs: Dict[str, Any]):
    """Ensure that the image processor handles min/max pixels properly."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"

    image = image_assets[0].pil_image
    hf_config = qwen2_vl_context.get_hf_config()
    image_token_id = hf_config.image_token_id

    inputs = token_inputs(prompt_token_ids=tokenizer.encode(prompt),
                          prompt=prompt,
                          multi_modal_data={"image": [image]})

    processed_inputs = input_processor_for_qwen2_vl(qwen2_vl_context, inputs,
                                                    **mm_processor_kwargs)
    assert processed_inputs["prompt_token_ids"].count(
        image_token_id) == num_placeholders
    assert len(processed_inputs["multi_modal_data"]["image"]) == 1


@pytest.mark.parametrize("mm_processor_kwargs,pixels_shape", [
    ({}, [5704, 1176]),
    ({
        MIN_PIXELS: 64**2,
        MAX_PIXELS: 512**2
    }, [1320, 1176]),
])
def test_image_mapper_override(qwen2_vl_context: InputContext,
                               image_assets: _ImageAssets,
                               mm_processor_kwargs: Dict[str, Any],
                               pixels_shape: Tuple[int, int]):
    """Ensure that the image mapper handles min/max pixels properly."""
    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(qwen2_vl_context.model_config)

    image = image_assets[0].pil_image

    mapped_output = mm_registry.map_input(
        qwen2_vl_context.model_config,
        {"image": image},
        mm_processor_kwargs=mm_processor_kwargs,
    )

    # Dimension 0 of pixel values should match the product of image_grid_thw
    actual_pixels_shape = mapped_output["pixel_values"].shape
    assert list(actual_pixels_shape) == pixels_shape
    assert actual_pixels_shape[0] == torch.prod(
        mapped_output["image_grid_thw"])
