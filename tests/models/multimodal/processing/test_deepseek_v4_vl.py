# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 VL processing helpers."""

from PIL import Image
from transformers import DeepseekV4Config

from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    build_image_block_pad_free,
)

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
