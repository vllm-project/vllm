"""Tests for phi3v's multimodal preprocessing kwargs."""
from typing import Optional

import pytest
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from vllm.inputs import InputContext, token_inputs
from vllm.model_executor.models.phi3v import _IMAGE_TOKEN_ID
from vllm.multimodal import MultiModalRegistry

from .....conftest import _ImageAssets
from ....utils import build_model_context

models = ["microsoft/Phi-3.5-vision-instruct"]


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def input_processor_for_phi3v():
    from vllm.model_executor.models.phi3v import input_processor_for_phi3v
    return input_processor_for_phi3v


@pytest.fixture()
def dummy_data_for_phi3v():
    from vllm.model_executor.models.phi3v import dummy_data_for_phi3v
    return dummy_data_for_phi3v


@pytest.fixture()
def get_max_phi3v_image_tokens():
    from vllm.model_executor.models.phi3v import get_max_phi3v_image_tokens
    return get_max_phi3v_image_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops", [4, 16, None])
def test_input_mapper_override(model: str, image_assets: _ImageAssets,
                               num_crops: Optional[int]):
    """Ensure that the [default] input mapper handles num_crops properly."""
    # We pass the processor kwargs here since for this model, we fall back to
    # the default mapper; this will fall back to the HF mapper and forward
    # mm_processor_kwargs to it.
    mm_processor_kwargs = {
        "num_crops": num_crops
    } if num_crops is not None else {}
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    hf_processor = AutoImageProcessor.from_pretrained(model,
                                                      trust_remote_code=True,
                                                      **mm_processor_kwargs)

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    image = image_assets[0].pil_image
    hf_result = hf_processor.preprocess(
        image,
        return_tensors="pt",
    )

    vllm_result = mm_registry.map_input(
        ctx.model_config,
        {"image": image},
    )

    assert torch.all(hf_result["image_sizes"] == vllm_result["image_sizes"])
    assert torch.all(
        hf_result["num_img_tokens"] == vllm_result["num_img_tokens"])

    # For pixel values, the second axis should be the num_crops + 1
    # for the rescaled original image. The default value in VLLM falls
    # back to the HF config, which is why we compare to the processor num_crops
    assert torch.all(hf_result["pixel_values"] == vllm_result["pixel_values"])
    assert vllm_result["pixel_values"].shape[1] == hf_processor.num_crops + 1


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,expected_max_tokens", [
    (4, 781),
    (16, 2653),
])
def test_max_tokens_override(get_max_phi3v_image_tokens, model: str,
                             num_crops: int, expected_max_tokens: int):
    """Ensure get_max_phi3v_image_tokens handles num_crops properly."""
    # NOTE: mm_processor_kwargs on the context in this test is unused, since
    # this is testing the mapper directly. In practice, the processor kwargs
    # are wrapped in a closure when calling the max tokens func. We explicitly
    # do NOT use the mm_processor_kwargs in the model context here to ensure
    # that the max image tokens implementation is referencing a mix of the
    # kwargs to the function and the original mm_processor_kwargs in case
    # values are somehow updated and end up in a bad state.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    actual_max_tokens = get_max_phi3v_image_tokens(
        InputContext(ctx.model_config),
        num_crops=num_crops,
    )

    assert expected_max_tokens == actual_max_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,toks_per_img,num_imgs", [
    (4, 781, 1),
    (4, 781, 2),
    (16, 2653, 1),
    (16, 2653, 2),
])
def test_dummy_data_override(dummy_data_for_phi3v, model: str, num_crops: int,
                             toks_per_img: int, num_imgs: int):
    """Ensure dummy_data_for_phi3v handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the dummy data func.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    dummy_data = dummy_data_for_phi3v(
        ctx=ctx,
        seq_len=8192,  # Should be bigger than num_imgs * toks_per_img
        mm_counts={"image": num_imgs},
        num_crops=num_crops,
    )
    sequence_data = dummy_data.seq_data
    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = sequence_data.get_token_ids().count(_IMAGE_TOKEN_ID)
    assert img_tok_count == toks_per_img * num_imgs


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("num_crops,expected_toks_per_img,num_imgs", [
    (4, 757, 1),
    (4, 757, 2),
    (16, 1921, 1),
    (16, 1921, 2),
])
def test_input_processor_override(input_processor_for_phi3v,
                                  image_assets: _ImageAssets, model: str,
                                  num_crops: int, expected_toks_per_img: int,
                                  num_imgs: int):
    """Ensure input_processor_for_phi3v handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Build the image str / prompt based on the number of images we pass
    img_str = "".join([f"<|image_{idx}|>\n" for idx in range(1, num_imgs + 1)])
    prompt = f"<|user|>\n{img_str}<|end|>\n<|assistant|>\n"
    images = [image_assets[0].pil_image] * num_imgs

    inputs = token_inputs(prompt_token_ids=tokenizer.encode(prompt),
                          prompt=prompt,
                          multi_modal_data={"image": images})

    processed_inputs = input_processor_for_phi3v(ctx,
                                                 inputs,
                                                 num_crops=num_crops)

    # Ensure we have the right number of placeholders per num_crops size
    img_tok_count = processed_inputs["prompt_token_ids"].count(_IMAGE_TOKEN_ID)
    assert img_tok_count == expected_toks_per_img * num_imgs
