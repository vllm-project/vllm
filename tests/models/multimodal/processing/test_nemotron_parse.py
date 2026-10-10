# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Nemotron-Parse's device-side normalisation path."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform

from ...utils import build_model_context

_MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize(("width", "height"), [(1240, 1754), (310, 470)])
def test_mm_device_do_normalize(width: int, height: int):
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    images = [_noise_image(width, height)]
    prompt = [0]

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    mm_items = processor.info.parse_mm_data({"image": images})
    normalized_values = processor(prompt, mm_items=mm_items)["mm_kwargs"].get_data()[
        "pixel_values"
    ]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    raw_values = processor(prompt, mm_items=mm_items)["mm_kwargs"].get_data()[
        "pixel_values"
    ]
    assert raw_values.dtype == torch.uint8
    assert raw_values.shape == normalized_values.shape

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )
