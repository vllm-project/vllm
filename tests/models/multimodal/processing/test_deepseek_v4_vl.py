# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 VL processing helpers."""

from PIL import Image

from vllm.models.deepseek_v4.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo,
    DeepseekV4VLProcessor,
    build_image_block_pad_free,
)
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


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
    config = DeepseekV4Config(vision_n_layers=24)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    processor = DeepseekV4VLProcessor(config)

    dummy = info.get_image_size_with_most_features()
    dummy_features = _feature_counts(processor, dummy.width, dummy.height)
    wide_features = _feature_counts(processor, 1932, 336)

    assert dummy == (1932, 336)
    assert wide_features <= dummy_features


def test_get_image_size_with_most_features_respects_max_wh_ratio():
    config = DeepseekV4Config(vision_n_layers=24)
    info = DeepseekV4VLProcessingInfo(_ConfigCtx(config))
    dummy = info.get_image_size_with_most_features()
    assert dummy.width / dummy.height <= config.vision_max_wh_ratio
