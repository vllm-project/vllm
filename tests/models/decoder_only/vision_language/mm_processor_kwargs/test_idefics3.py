"""Tests for phi3v's multimodal preprocessing kwargs."""
from typing import Optional

import pytest
import torch
from transformers import AutoImageProcessor, AutoTokenizer

from vllm.inputs import InputContext, token_inputs
from vllm.multimodal import MultiModalRegistry

from .....conftest import _ImageAssets
from ....utils import build_model_context

models = ["HuggingFaceM4/Idefics3-8B-Llama3"]


# Wrap lazy imports to avoid initializing CUDA during test collection
@pytest.fixture()
def input_processor_for_idefics3():
    from vllm.model_executor.models.idefics3 import input_processor_for_idefics3
    return input_processor_for_idefics3


@pytest.fixture()
def dummy_data_for_idefics3():
    from vllm.model_executor.models.idefics3 import dummy_data_for_idefics3
    return dummy_data_for_idefics3


@pytest.fixture()
def get_max_idefics3_image_tokens():
    from vllm.model_executor.models.idefics3 import get_max_idefics3_image_tokens
    return get_max_idefics3_image_tokens


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("longest_edge", [None, 336, 2*336, 1000])
def test_input_mapper_override(model: str, image_assets: _ImageAssets,
                               longest_edge: Optional[int]):
    """Ensure that the [default] input mapper handles size properly."""
    # 
    mm_processor_kwargs = {
        "size": {"longest_edge": longest_edge}
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


