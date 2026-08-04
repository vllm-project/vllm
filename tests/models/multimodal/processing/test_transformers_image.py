# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache


@pytest.mark.parametrize("model_id", ["llava-hf/llava-onevision-qwen2-0.5b-ov-hf"])
def test_multimodal_processor(model_id):
    model_config = ModelConfig(
        model=model_id,
        model_impl="transformers",
    )

    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    image_pil = ImageAsset("cherry_blossom").pil_image
    mm_data = {"image": image_pil}
    str_prompt = "<|im_start|>user <image>\nWhat is the content of this image?<|im_end|><|im_start|>assistant\n"  # noqa: E501
    str_processed_inputs = mm_processor(
        prompt=str_prompt,
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    ids_prompt = [
        151644,
        872,
        220,
        151646,
        198,
        3838,
        374,
        279,
        2213,
        315,
        419,
        2168,
        30,
        151645,
        151644,
        77091,
        198,
    ]
    ids_processed_inputs = mm_processor(
        prompt=ids_prompt,
        mm_items=mm_processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={},
    )

    assert (
        str_processed_inputs["prompt_token_ids"]
        == ids_processed_inputs["prompt_token_ids"]
    )


def test_image_multiple_inputs():
    """Multiple images per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers backend."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    image = ImageAsset("cherry_blossom").pil_image
    prompt = (
        "<|im_start|>user <image>\n and <image>\n"
        "What do these images show?<|im_end|><|im_start|>assistant\n"
    )

    result = mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


def test_image_cached_apply_gemma3():
    model_id = "google/gemma-3-4b-it"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    cache = MultiModalProcessorOnlyCache(model_config)
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config, cache=cache)

    image = ImageAsset("cherry_blossom").pil_image
    image_token = mm_processor.info.get_hf_processor().boi_token
    prompt = f"{image_token} What is the content of this image?"

    for _ in range(2):
        result = mm_processor(
            prompt=prompt,
            mm_items=mm_processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={},
        )
        assert len(result["mm_placeholders"]["image"]) == 1
