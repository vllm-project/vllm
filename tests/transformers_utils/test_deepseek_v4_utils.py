# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from transformers import DeepseekV4Config

from vllm.transformers_utils.deepseek_v4_utils import (
    get_compress_ratios,
    get_mm_prefix_clamp_sliding_window,
    get_num_hash_layers,
)

# Legacy fields as shipped by deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0, 0, 0]


@pytest.mark.parametrize("vision_n_layers", [0, 32])
def test_legacy_fields_survive_upstream_config(vision_n_layers):
    """Upstream pops the legacy fields; the helpers must recover them."""
    config = DeepseekV4Config(
        num_hidden_layers=43,
        num_nextn_predict_layers=3,
        compress_ratios=COMPRESS_RATIOS,
        num_hash_layers=3,
        vision_n_layers=vision_n_layers,
    )
    assert not hasattr(config, "compress_ratios")
    assert get_compress_ratios(config) == COMPRESS_RATIOS
    assert get_num_hash_layers(config) == 3
    assert get_mm_prefix_clamp_sliding_window(config) == (vision_n_layers > 0)
