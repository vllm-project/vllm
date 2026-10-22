# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import DeepseekV4Config

from vllm.transformers_utils.deepseek_v4_utils import get_compress_ratios

# Legacy field as shipped by deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0, 0, 0]


def test_compress_ratios_survive_upstream_config():
    """Upstream pops `compress_ratios`; the helper must recover it."""
    config = DeepseekV4Config(
        num_hidden_layers=43,
        num_nextn_predict_layers=3,
        compress_ratios=COMPRESS_RATIOS,
    )
    assert not hasattr(config, "compress_ratios")
    assert get_compress_ratios(config) == COMPRESS_RATIOS
