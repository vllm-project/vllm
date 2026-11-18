# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen2_vl import Qwen2VLImageProcessor

from vllm.model_executor.models.opencua import OpenCUAProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY

from ...utils import build_model_context


def test_default_pixels_match_hf_processor():
    ctx = build_model_context(
        "xlangai/OpenCUA-7B",
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor = processor.info.get_hf_processor()
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))

    processed_inputs = processor(
        hf_processor.image_token,
        mm_items=processor.info.parse_mm_data({"image": [image]}),
        hf_processor_mm_kwargs={},
    )
    pixel_values = processed_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)

    image_processor = Qwen2VLImageProcessor(**ctx.get_hf_image_processor_config())
    expected = image_processor([image], return_tensors="pt")["pixel_values"]

    torch.testing.assert_close(pixel_values, expected.to(pixel_values.dtype))


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        {"min_pixels": 64**2, "max_pixels": 512**2},
        {"size": {"shortest_edge": 64**2, "longest_edge": 512**2}},
        {"images_kwargs": {"max_pixels": 512**2}},
        {"images_kwargs": {"max_pixels": 512**2, "return_tensors": "pt"}},
    ],
)
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override_matches_image_sizing(
    mm_processor_kwargs: dict[str, object],
    kwargs_on_init: bool,
):
    ctx = build_model_context(
        "xlangai/OpenCUA-7B",
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    info = processor.info
    assert isinstance(info, OpenCUAProcessingInfo)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    hf_processor = info.get_hf_processor(**hf_processor_mm_kwargs)
    image = Image.new("RGB", (1920, 1080))

    processed_inputs = processor(
        hf_processor.image_token,
        mm_items=info.parse_mm_data({"image": [image]}),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )

    merged_kwargs = ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
    expected_tokens = info.get_num_image_tokens(
        image_width=image.width,
        image_height=image.height,
        image_processor=info.get_image_processor(**merged_kwargs),
        mm_kwargs=merged_kwargs,
    )
    image_token_id = info.get_tokenizer().convert_tokens_to_ids(
        hf_processor.image_token
    )
    grid_thw = processed_inputs["mm_kwargs"].get_data()["image_grid_thw"]
    assert isinstance(grid_thw, torch.Tensor)

    assert expected_tokens == 312
    assert processed_inputs["prompt_token_ids"].count(image_token_id) == 312
    assert grid_thw.tolist() == [[1, 26, 48]]
