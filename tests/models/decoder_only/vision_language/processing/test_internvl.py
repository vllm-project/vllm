"""Tests for InternVL's multimodal preprocessing kwargs."""
from typing import Callable, Optional

import pytest
from transformers import AutoTokenizer

from vllm.inputs import InputContext, token_inputs
from vllm.multimodal import MultiModalRegistry

from .....conftest import _ImageAssets
from ....utils import build_model_context

models = ["OpenGVLab/InternVL2-2B"]


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def input_processor_for_internvl():
    from vllm.model_executor.models.internvl import InternVLInputPipeline

    pipeline = InternVLInputPipeline('<img>', '</img>', '<IMG_CONTEXT>')
    return pipeline.input_processor


@pytest.fixture()
def dummy_data_for_internvl():
    from vllm.model_executor.models.internvl import InternVLInputPipeline

    pipeline = InternVLInputPipeline('<img>', '</img>', '<IMG_CONTEXT>')
    return pipeline.dummy_data


@pytest.fixture()
def get_max_internvl_image_tokens():
    from vllm.model_executor.models.internvl import (
        get_max_internvl_image_tokens)
    return get_max_internvl_image_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("max_dynamic_patch", [1, 4])
@pytest.mark.parametrize("dynamic_image_size", [True, False, None])
def test_input_mapper_override(
    model: str,
    image_assets: _ImageAssets,
    max_dynamic_patch: int,
    dynamic_image_size: Optional[bool],
):
    mm_processor_kwargs = {
        "max_dynamic_patch": max_dynamic_patch,
    }
    if dynamic_image_size is not None:
        mm_processor_kwargs["dynamic_image_size"] = dynamic_image_size

    expected_num_patches = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    if dynamic_image_size is False:
        expected_num_patches = 1

    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    image = image_assets[0].pil_image.resize((448 * 2, 448 * 2))
    vllm_result = mm_registry.map_input(
        ctx.model_config,
        {"image": image},
    )
    assert vllm_result["pixel_values"].size(1) == expected_num_patches


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("max_dynamic_patch", [1, 4, None])
@pytest.mark.parametrize("dynamic_image_size", [True, False, None])
def test_max_tokens_override(
    get_max_internvl_image_tokens: Callable,
    model: str,
    max_dynamic_patch: Optional[int],
    dynamic_image_size: Optional[bool],
):
    """Ensure get_max_internvl_image_tokens handles mm_processor_kwargs."""
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    if max_dynamic_patch is None:
        max_dynamic_patch = ctx.get_hf_config().max_dynamic_patch
    expected_num_patches = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    if dynamic_image_size is False:
        expected_num_patches = 1
    expected_max_tokens = 256 * expected_num_patches

    actual_max_tokens = get_max_internvl_image_tokens(
        ctx=InputContext(ctx.model_config),
        max_dynamic_patch=max_dynamic_patch,
        dynamic_image_size=dynamic_image_size,
    )
    assert expected_max_tokens == actual_max_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("max_dynamic_patch", [1, 4, None])
@pytest.mark.parametrize("dynamic_image_size", [True, False, None])
def test_dummy_data_override(
    dummy_data_for_internvl: Callable,
    model: str,
    num_imgs: int,
    max_dynamic_patch: Optional[int],
    dynamic_image_size: Optional[bool],
):
    """Ensure dummy_data_for_internvl handles kwargs properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the dummy data func.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    if max_dynamic_patch is None:
        max_dynamic_patch = ctx.get_hf_config().max_dynamic_patch
    expected_num_patches = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    if dynamic_image_size is False:
        expected_num_patches = 1
    expected_max_tokens = 256 * expected_num_patches

    dummy_data = dummy_data_for_internvl(
        ctx=ctx,
        seq_len=8192,  # Should be bigger than num_imgs * toks_per_img
        mm_counts={"image": num_imgs},
        max_dynamic_patch=max_dynamic_patch,
        dynamic_image_size=dynamic_image_size,
    )
    sequence_data = dummy_data.seq_data

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    image_token_id = tokenizer.encode('<IMG_CONTEXT>',
                                      add_special_tokens=False)[0]

    # Ensure we have the right number of placeholders per size
    img_tok_count = sequence_data.get_token_ids().count(image_token_id)
    assert img_tok_count == expected_max_tokens * num_imgs


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("max_dynamic_patch", [1, 4])
@pytest.mark.parametrize("dynamic_image_size", [True, False, None])
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_input_processor_override(
    input_processor_for_internvl: Callable,
    image_assets: _ImageAssets,
    model: str,
    num_imgs: int,
    max_dynamic_patch: int,
    dynamic_image_size: Optional[bool],
):
    """Ensure input_processor_for_internvl handles kwargs properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    expected_num_patches = max_dynamic_patch + 1 if max_dynamic_patch > 1 else 1
    if dynamic_image_size is False:
        expected_num_patches = 1

    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )
    expected_toks_per_img = 256 * expected_num_patches

    # Build the image str / prompt based on the number of images we pass
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    placeholders = "<image>" if num_imgs == 1 else "\n".join(
        f"Image-{i}: <image>\n" for i in range(1, num_imgs + 1))
    prompt = placeholders
    images = [image_assets[0].pil_image.resize((448 * 2, 448 * 2))] * num_imgs

    inputs = token_inputs(prompt_token_ids=tokenizer.encode(prompt),
                          prompt=prompt,
                          multi_modal_data={"image": images})

    processed_inputs = input_processor_for_internvl(
        ctx,
        inputs,
        max_dynamic_patch=max_dynamic_patch,
        dynamic_image_size=dynamic_image_size,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.encode('<IMG_CONTEXT>',
                                      add_special_tokens=False)[0]
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert img_tok_count == expected_toks_per_img * num_imgs
