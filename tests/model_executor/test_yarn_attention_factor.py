# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope


def _yarn_mscale(factor, attention_factor=None):
    rope_parameters = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000,
    }
    if attention_factor is not None:
        rope_parameters["attention_factor"] = attention_factor
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=128,
            max_position=int(4096 * factor),
            is_neox_style=True,
            rope_parameters=rope_parameters,
            dtype=torch.float32,
        )
    return float(rope.mscale)


@pytest.mark.parametrize("factor", [8.0, 64.0])
def test_yarn_uses_explicit_attention_factor(factor):
    assert _yarn_mscale(factor, attention_factor=1.0) == pytest.approx(1.0)


def test_yarn_default_mscale_unchanged():
    factor = 64.0
    assert _yarn_mscale(factor) == pytest.approx(0.1 * math.log(factor) + 1.0)
