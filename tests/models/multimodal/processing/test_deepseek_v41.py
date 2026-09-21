# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from vllm.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLImageProcessor,
    DeepseekV4VLProcessingInfo,
    num_image_tokens,
)


@pytest.mark.parametrize("budget", [64, 256, 512, 1024])
@pytest.mark.parametrize("max_wh_ratio", [None, 0.5, 1, 4, 8, 3.5])
def test_profile_image_maximizes_patch_count(budget, max_wh_ratio):
    """Profile the largest legal ViT grid, not just a square image."""
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=budget,
        vision_min_pixels=295936,
        vision_max_wh_ratio=max_wh_ratio,
    )
    info = Mock(spec=DeepseekV4VLProcessingInfo)
    info.get_hf_config.return_value = config
    size = DeepseekV4VLProcessingInfo.get_image_size_with_most_features(info)

    patches, vit_h, vit_w, llm_h, llm_w = DeepseekV4VLImageProcessor(config)(
        Image.new("RGB", (size.width, size.height))
    )

    # Exhaustively enumerate legal patch grids, independently of the selector.
    r = config.vision_downsample_ratio
    expected = 0
    for h in range(1, ((budget - 2) // 2) * r + 1):
        llm_rows = (h + r - 1) // r
        for w in range(1, ((budget - 2) // llm_rows - 1) * r + 1):
            # The width clamp is followed by rounding up to a patch boundary.
            if max_wh_ratio is None or w - 1 < h * max_wh_ratio:
                expected = max(expected, h * w)
    assert patches.shape[0] == vit_h * vit_w == expected
    assert num_image_tokens(llm_h, llm_w) <= budget


def test_default_profile_covers_wide_image():
    config = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=None,
    )
    info = Mock(spec=DeepseekV4VLProcessingInfo)
    info.get_hf_config.return_value = config
    size = DeepseekV4VLProcessingInfo.get_image_size_with_most_features(info)
    processor = DeepseekV4VLImageProcessor(config)
    dummy, _, _, h, w = processor(Image.new("RGB", (size.width, size.height)))
    wide, *_ = processor(Image.new("RGB", (2814, 630)))

    assert dummy.shape[0] == 9189
    assert dummy.shape[0] >= wide.shape[0] == 9045
    assert num_image_tokens(h, w) == 1024
