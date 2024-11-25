"""Tests for Idefics3's multimodal preprocessing kwargs."""
from typing import Optional

import pytest
import torch
import transformers
from transformers import AutoImageProcessor, AutoTokenizer

from vllm.inputs import InputContext, token_inputs
from vllm.multimodal import MultiModalRegistry

from .....conftest import _ImageAssets
from ....utils import build_model_context

models = ["HuggingFaceM4/Idefics3-8B-Llama3"]


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def input_processor_for_idefics3():
    from vllm.model_executor.models.idefics3 import (
        input_processor_for_idefics3)
    return input_processor_for_idefics3


@pytest.fixture()
def dummy_data_for_idefics3():
    from vllm.model_executor.models.idefics3 import dummy_data_for_idefics3
    return dummy_data_for_idefics3


@pytest.fixture()
def get_max_idefics3_image_tokens():
    from vllm.model_executor.models.idefics3 import (
        get_max_idefics3_image_tokens)
    return get_max_idefics3_image_tokens


@pytest.mark.skipif(transformers.__version__ < "4.46.0",
                    reason="Model introduced in HF >= 4.46.0")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("longest_edge", [None, 168, 336, 400, 2 * 336])
def test_input_mapper_override(model: str, image_assets: _ImageAssets,
                               longest_edge: Optional[int]):
    """Ensure that the [default] input mapper handles size properly."""

    mm_processor_kwargs = {
        "size": {
            "longest_edge": longest_edge
        }
    } if longest_edge is not None else {}
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

    assert torch.all(hf_result["pixel_values"] == vllm_result["pixel_values"])


@pytest.mark.skipif(transformers.__version__ < "4.46.0",
                    reason="Model introduced in HF >= 4.46.0")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("longest_edge, expected_max_tokens", [
    (None, 2873),
    (168, 169),
    (336, 169),
    (400, 338),
    (672, 338),
])
def test_max_tokens_override(get_max_idefics3_image_tokens, model: str,
                             longest_edge: Optional[int],
                             expected_max_tokens: int):
    """Ensure get_max_idefics3_image_tokens handles mm_processor_kwargs."""
    size = {"longest_edge": longest_edge} if longest_edge is not None else None
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    actual_max_tokens = get_max_idefics3_image_tokens(
        ctx=InputContext(ctx.model_config),
        size=size,
    )

    assert expected_max_tokens == actual_max_tokens


@pytest.mark.skipif(transformers.__version__ < "4.46.0",
                    reason="Model introduced in HF >= 4.46.0")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("longest_edge, toks_per_img, num_imgs", [
    (168, 169, 1),
    (168, 169, 2),
    (400, 338, 1),
    (400, 338, 2),
])
def test_dummy_data_override(dummy_data_for_idefics3, model: str,
                             longest_edge: int, toks_per_img: int,
                             num_imgs: int):
    """Ensure dummy_data_for_idefics3 handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the dummy data func.
    size = {"longest_edge": longest_edge} if longest_edge is not None else None
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    dummy_data = dummy_data_for_idefics3(
        ctx=ctx,
        seq_len=8192,  # Should be bigger than num_imgs * toks_per_img
        mm_counts={"image": num_imgs},
        size=size)
    sequence_data = dummy_data.seq_data
    # Ensure we have the right number of placeholders per size
    image_token_id = ctx.get_hf_config().image_token_id
    img_tok_count = sequence_data.get_token_ids().count(image_token_id)
    assert img_tok_count == toks_per_img * num_imgs


@pytest.mark.skipif(transformers.__version__ < "4.46.0",
                    reason="Model introduced in HF >= 4.46.0")
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("longest_edge,expected_toks_per_img,num_imgs", [
    (336, 169 * (1**2 + 1), 1),
    (336, 169 * (1**2 + 1), 2),
    (400, 169 * (2**2 + 1), 1),
    (400, 169 * (2**2 + 1), 2),
])
def test_input_processor_override(input_processor_for_idefics3,
                                  image_assets: _ImageAssets, model: str,
                                  longest_edge: int,
                                  expected_toks_per_img: int, num_imgs: int):
    """Ensure input_processor_for_idefics3 handles num_crops properly."""
    # Same as the previous test - don't initialize mm_processor_kwargs
    # in this test and assume that the kwargs will be correctly expanded by
    # the partial when calling the custom input processor.
    size = {"longest_edge": longest_edge} if longest_edge is not None else None
    ctx = build_model_context(
        model_name=model,
        tokenizer_name=model,
        trust_remote_code=True,
        mm_processor_kwargs=None,
    )

    # Build the image str / prompt based on the number of images we pass
    tokenizer = AutoTokenizer.from_pretrained(model)
    placeholders = "<image>" if num_imgs == 1 else "\n".join(
        f"Image-{i}: <image>\n" for i in range(1, num_imgs + 1))
    prompt = f"<|begin_of_text|>User:{placeholders}\n<end_of_utterance>\nAssistant:"  # noqa: E501
    images = [image_assets[0].pil_image.resize((336 * 4, 336 * 4))] * num_imgs

    inputs = token_inputs(prompt_token_ids=tokenizer.encode(prompt),
                          prompt=prompt,
                          multi_modal_data={"image": images})

    processed_inputs = input_processor_for_idefics3(ctx, inputs, size=size)

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = ctx.get_hf_config().image_token_id
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert img_tok_count == expected_toks_per_img * num_imgs
