# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig
from vllm.model_executor.models.transformers.multimodal import (
    MultiModalProcessor,
    OffsetsMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


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


def _process_two_images(separator: str):
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    image = ImageAsset("cherry_blossom").pil_image
    prompt = (
        f"<|im_start|>user <image>{separator}<image>\n"
        "What do these images show?<|im_end|><|im_start|>assistant\n"
    )

    return mm_processor(
        prompt=prompt,
        mm_items=mm_processor.info.parse_mm_data({"image": [image, image]}),
        hf_processor_mm_kwargs={},
    )


def test_image_multiple_inputs():
    """Multiple images per prompt are each detected as a separate placeholder
    and multi-modal item by the Transformers modelling backend."""
    result = _process_two_images(separator="\n and ")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


def test_image_adjacent_inputs():
    """Adjacent images stay separate placeholders rather than merging into one."""
    result = _process_two_images(separator="")

    assert len(result["mm_placeholders"]["image"]) == 2
    assert len(result["mm_kwargs"]["image"]) == 2


def test_batch_padding_removed_from_image_items():
    """Emu3 pads every image up to the largest in the batch, which would leave an
    item's data dependent on what it was processed with and so uncacheable."""
    model_id = "BAAI/Emu3-Chat-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    image_token = mm_processor.info.get_hf_processor().image_token

    images = [
        ImageAsset("cherry_blossom").pil_image,
        ImageAsset("cherry_blossom").pil_image.resize((256, 1024)),
    ]
    result = mm_processor(
        prompt=f"{image_token} and {image_token}",
        mm_items=mm_processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )

    items = result["mm_kwargs"]["image"]
    shapes = set()
    for item in items:
        height, width = item["image_sizes"].data.flatten().tolist()
        pixel_values = item["pixel_values"].data
        assert tuple(pixel_values.shape[-2:]) == (height, width)
        shapes.add(tuple(pixel_values.shape))

    # Both images would have been padded to a common shape had they been kept
    assert len(shapes) == 2


def test_non_embedding_tokens_excluded_from_placeholders():
    """Gemma3 wraps each image in text that carries no embeddings, which must be
    inside the placeholder range but masked out of it."""
    model_id = "google/gemma-3-4b-it"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    hf_processor = mm_processor.info.get_hf_processor()
    result = mm_processor(
        prompt=f"{hf_processor.boi_token} What is this?",
        mm_items=mm_processor.info.parse_mm_data(
            {"image": ImageAsset("cherry_blossom").pil_image}
        ),
        hf_processor_mm_kwargs={},
    )

    (placeholder,) = result["mm_placeholders"]["image"]
    assert placeholder.is_embed is not None
    assert 0 < int(placeholder.is_embed.sum()) < placeholder.length


@pytest.mark.skipif(
    MultiModalProcessor is not OffsetsMultiModalProcessor,
    reason="Replacement offsets are only used from transformers 5.15.0 onwards",
)
def test_missing_replacement_offsets_names_the_processor():
    """A processor that reports no replacement offsets cannot be served, which must
    be said plainly rather than surfacing later as a field config mismatch."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    hf_processor_cls = type(mm_processor.info.get_hf_processor())
    hf_call = hf_processor_cls.__call__

    def without_offsets(self, *args, **kwargs):
        hf_inputs = hf_call(self, *args, **kwargs)
        hf_inputs.pop("text_replacement_offsets", None)
        return hf_inputs

    with (
        patch.object(hf_processor_cls, "__call__", without_offsets),
        pytest.raises(ValueError, match="LlavaOnevisionProcessor returned no"),
    ):
        mm_processor(
            prompt="<image>\nWhat is the content of this image?",
            mm_items=mm_processor.info.parse_mm_data(
                {"image": ImageAsset("cherry_blossom").pil_image}
            ),
            hf_processor_mm_kwargs={},
        )


def test_text_only_prompt():
    """An image model still accepts a prompt with no images."""
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_config = ModelConfig(model=model_id, model_impl="transformers")
    mm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    result = mm_processor(
        prompt="<|im_start|>user Hello!<|im_end|><|im_start|>assistant\n",
        mm_items=mm_processor.info.parse_mm_data({}),
        hf_processor_mm_kwargs={},
    )

    assert len(result["prompt_token_ids"]) > 0
    assert not result["mm_placeholders"]
