"""Tests for Qwen's multimodal preprocessing kwargs."""
from typing import Dict, List, Union

import pytest
import torch
from PIL.Image import Image

from vllm.inputs import InputContext, token_inputs
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.utils import cached_get_tokenizer

from .....conftest import IMAGE_ASSETS
from ....utils import build_model_context

### Multimodal preprocessing tests
SAMPLE_IMAGE = IMAGE_ASSETS[0].pil_image
# These values are specific to Qwen-VL/Chat; we can get these from the model
# config also, but they are hardcoded here to keep the parameterize/fixtures
# easy to read.
IMG_START_ID = 151857
IMG_END_ID = 151858
IMG_PAD_ID = 151859
TOKS_PER_IMG = 256
VIS_ENC_DIM = 4096
IMG_SIZE = 448


@pytest.fixture()
def input_mapper_for_qwen():
    # Lazy import to avoid initializing CUDA during test collection
    from vllm.model_executor.models.qwen import input_mapper_for_qwen
    return input_mapper_for_qwen


@pytest.fixture()
def input_processor_for_qwen():
    # Lazy import to avoid initializing CUDA during test collection
    from vllm.model_executor.models.qwen import input_processor_for_qwen
    return input_processor_for_qwen


@pytest.fixture()
def qwen_vl_context() -> InputContext:
    """Get an InputContext for Qwen-VL."""
    return build_model_context(model_name="Qwen/Qwen-VL",
                               trust_remote_code=True)


# Happy path tests for single/multi-image scenarios for the multimodal
# input processor and mapper, respectively
@pytest.mark.parametrize("num_images", [1, 2])
def test_input_processor_valid_mm_data(input_processor_for_qwen,
                                       qwen_vl_context: InputContext,
                                       num_images: int):
    """Happy cases for image inputs to Qwen's multimodal input processor."""
    prompt = "".join(
        [f"Picture {num}: <img></img>\n" for num in range(1, num_images + 1)])
    inputs = token_inputs(
        prompt=prompt,
        # When processing multimodal data for a multimodal model, the qwen
        # input processor will overwrite the provided prompt_token_ids with
        # the image prompts
        prompt_token_ids=[],
        multi_modal_data={"image": torch.rand(num_images, TOKS_PER_IMG, 4096)},
    )
    proc_inputs = input_processor_for_qwen(qwen_vl_context, inputs)
    assert isinstance(proc_inputs, dict)

    # Each image should have one start / stop and a fixed context of 256
    proc_tokens = proc_inputs["prompt_token_ids"]
    assert proc_tokens.count(IMG_START_ID) == num_images
    assert proc_tokens.count(IMG_END_ID) == num_images
    assert proc_tokens.count(IMG_PAD_ID) == num_images * TOKS_PER_IMG


@pytest.mark.parametrize(
    "img_data,expected_shape",
    [
        # single / multi-image
        (SAMPLE_IMAGE, (1, 3, IMG_SIZE, IMG_SIZE)),
        (2 * [SAMPLE_IMAGE], (2, 3, IMG_SIZE, IMG_SIZE)),
        # single / multi-image embeddings
        (torch.rand(
            (TOKS_PER_IMG, VIS_ENC_DIM)), (1, TOKS_PER_IMG, VIS_ENC_DIM)),
        (torch.rand(
            (1, TOKS_PER_IMG, VIS_ENC_DIM)), (1, TOKS_PER_IMG, VIS_ENC_DIM)),
        (torch.rand(
            (2, TOKS_PER_IMG, VIS_ENC_DIM)), (2, TOKS_PER_IMG, VIS_ENC_DIM)),
    ])
def test_input_mapper_valid_mm_data(input_mapper_for_qwen,
                                    qwen_vl_context: InputContext,
                                    img_data: Union[torch.Tensor, List[Image],
                                                    Image],
                                    expected_shape: List[int]):
    """Happy cases for image inputs to Qwen's multimodal input mapper."""
    mapped_img_data = input_mapper_for_qwen(qwen_vl_context, img_data)
    # Ensure that we get the appropriately shaped pixel_values
    # for images and image embeddings, respectively.
    assert isinstance(mapped_img_data, MultiModalKwargs)
    assert "pixel_values" in mapped_img_data
    assert mapped_img_data["pixel_values"].shape == expected_shape


# Sad path tests for the multimodal input processor and mapper, respectively
@pytest.mark.parametrize("mm_data", [
    {
        "image": torch.rand(5)
    },
    {
        "image": torch.rand((5, 5, 5, 5, 5))
    },
])
def test_input_processor_invalid_mm_data(input_processor_for_qwen,
                                         qwen_vl_context: InputContext,
                                         mm_data: Dict[str, torch.Tensor]):
    """Test sad cases validated in Qwen's multimodal input processor."""
    tokenizer = cached_get_tokenizer(qwen_vl_context.model_config.tokenizer,
                                     trust_remote_code=True)
    prompt = "Picture 1: <img></img>\n"
    prompt_token_ids = tokenizer.encode(prompt)
    inputs = token_inputs(prompt=prompt,
                          prompt_token_ids=prompt_token_ids,
                          multi_modal_data=mm_data)
    # Should fail since we have too many or too few dimensions for embeddings
    with pytest.raises(ValueError):
        input_processor_for_qwen(qwen_vl_context, inputs)


@pytest.mark.parametrize(
    "img_data",
    [
        # Wrong context length
        torch.rand((1, TOKS_PER_IMG + 10, VIS_ENC_DIM)),
        # Wrong visual encoder output size
        torch.rand((1, TOKS_PER_IMG, VIS_ENC_DIM + 10)),
    ])
def test_input_mapper_invalid_mm_data(
    input_mapper_for_qwen,
    qwen_vl_context: InputContext,
    img_data: Union[torch.Tensor, List[Image], Image],
):
    """Sad cases validated in Qwen VL's multimodal input mapper."""
    with pytest.raises(ValueError):
        input_mapper_for_qwen(qwen_vl_context, img_data)
