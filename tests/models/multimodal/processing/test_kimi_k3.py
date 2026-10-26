# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)

from ...utils import build_model_context

_MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture(scope="module")
def image_processors():
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    k25_fused = processor.info.image_processor
    assert isinstance(k25_fused, KimiK25FusedVisionProcessor)
    k3_hf = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return k25_fused, k3_hf


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("width", "height"),
    [(310, 470), (7300, 30), (13, 9)],
)
def test_fused_image_processor_matches_hf(image_processors, width: int, height: int):
    fused, hf = image_processors
    image = _noise_image(width, height)
    expected = hf.preprocess([{"type": "image", "image": image}], return_tensors="pt")
    actual = fused.preprocess([{"type": "image", "image": image}], return_tensors="pt")

    torch.testing.assert_close(actual["grid_thws"], expected["grid_thws"])
    torch.testing.assert_close(actual["pixel_values"], expected["pixel_values"])
    assert fused.media_tokens_calculator(
        {"type": "image", "image": image}
    ) == hf.media_tokens_calculator({"type": "image", "image": image})


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 2})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = processor.info.get_hf_config().image_placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )
