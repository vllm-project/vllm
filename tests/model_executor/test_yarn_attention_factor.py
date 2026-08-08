# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

pytestmark = pytest.mark.skip_global_cleanup


def _yarn_mscale(factor, **kwargs):
    rope_parameters = {
        "rope_type": "yarn",
        "factor": float(factor),
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000,
    }
    rope_parameters.update(kwargs)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
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


def test_yarn_uses_mscale_ratio():
    factor = 16.0
    expected = (0.2 * math.log(factor) + 1.0) / (0.1 * math.log(factor) + 1.0)
    assert _yarn_mscale(factor, mscale=2.0, mscale_all_dim=1.0) == pytest.approx(
        expected
    )


def test_yarn_equal_mscale_ratio_disables_magnitude_scaling():
    assert _yarn_mscale(16.0, mscale=1.0, mscale_all_dim=1.0) == pytest.approx(1.0)


def test_yarn_attention_factor_takes_precedence_over_mscale_ratio():
    assert _yarn_mscale(
        16.0,
        attention_factor=0.5,
        mscale=2.0,
        mscale_all_dim=1.0,
    ) == pytest.approx(0.5)


def test_yarn_disabled_scaling_takes_precedence_over_mscale_ratio():
    assert _yarn_mscale(
        16.0,
        apply_yarn_scaling=False,
        mscale=2.0,
        mscale_all_dim=1.0,
    ) == pytest.approx(1.0)
