# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 / V4.1 VL processing helpers."""

import copy

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DeepseekV4Config

from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE_PLACEHOLDER as V4_IMAGE_PLACEHOLDER,
)
from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    build_image_block_pad_free,
)
from vllm.models.deepseek_v4.common.vision import DeepseekV4ViT
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE_PLACEHOLDER as V41_IMAGE_PLACEHOLDER,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform

from ...utils import build_model_context

# Vision fields of deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
VISION_CONFIG = dict(
    vision_n_layers=24,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_patch_size=14,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    vision_max_n_token=384,
    vision_min_pixels=147456,
    vision_max_wh_ratio=8,
)


class _ConfigCtx:
    def __init__(self, config: DeepseekV4Config) -> None:
        self._config = config

    def get_hf_config(self, *args, **kwargs):
        return self._config


def _feature_counts(
    processor: DeepseekV4VLProcessor, width: int, height: int
) -> tuple[int, int]:
    out = processor(images=[Image.new("RGB", (width, height))])
    n_llm_h, n_llm_w = out["llm_grid"][0].tolist()
    types, _ = build_image_block_pad_free(n_llm_h, n_llm_w)
    return len(types), out["patches"].shape[0]


def test_get_image_size_with_most_features_beats_wide_real_image():
    """Dummy size must maximize tokens/patches within the budget (#59271).

    A square under-counts because each aligner row pays an IMAGE_NEW_LINE.
    """
    config = DeepseekV4Config(**VISION_CONFIG)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    processor = DeepseekV4VLProcessor(config)

    dummy = info.get_image_size_with_most_features()
    dummy_features = _feature_counts(processor, dummy.width, dummy.height)
    wide_features = _feature_counts(processor, 1932, 336)

    assert dummy == (1932, 336)
    assert wide_features <= dummy_features


def test_get_image_size_with_most_features_respects_max_wh_ratio():
    config = DeepseekV4Config(**VISION_CONFIG)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    dummy = info.get_image_size_with_most_features()
    assert dummy.width / dummy.height <= config.vision_max_wh_ratio


@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize(
    ("model_id", "placeholder"),
    [
        pytest.param(
            "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            V4_IMAGE_PLACEHOLDER,
            marks=pytest.mark.skipif(
                not (current_platform.is_cuda() or current_platform.is_rocm()),
                reason="DeepSeek-V4 vision is only supported on CUDA/ROCm",
            ),
        ),
        ("deepseek-ai/DeepSeek-V4.1-Flash", V41_IMAGE_PLACEHOLDER),
    ],
)
def test_mm_device_do_normalize(model_id: str, placeholder: str):
    """On-device normalization must match the CPU processor output.

    Uses the ViT's own ``enable_mm_device_normalize`` since DeepSeek-V4 has
    no HF image processor config for ``build_mm_input_norm`` to read.
    """
    device = current_platform.device_type
    ctx = build_model_context(model_id, limit_mm_per_prompt={"image": 2})
    mm_config = ctx.model_config.multimodal_config
    assert mm_config.mm_device_do_normalize

    mm_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (310, 470, 3), dtype=np.uint8)),
        Image.fromarray(rng.integers(0, 256, (480, 320, 3), dtype=np.uint8)),
    ]
    prompt = placeholder * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized = normalized_inputs["mm_kwargs"].get_data()["patches"]

    mm_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw = raw_inputs["mm_kwargs"].get_data()["patches"]
    assert raw.dtype == torch.uint8

    vision_config = copy.deepcopy(ctx.model_config.hf_config)
    vision_config.vision_n_layers = 0
    vision = DeepseekV4ViT(vision_config)
    vision.enable_mm_device_normalize()
    input_norm = vision.patch_embed.input_norm.to(device)

    output = input_norm(raw.flatten(1).to(device), normalized.dtype)
    torch.testing.assert_close(
        output, normalized.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )
