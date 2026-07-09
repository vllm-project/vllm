# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import Olmo2Config, Olmo3Config

from vllm.model_executor.models.olmo2 import _get_rope_parameters


def test_olmo2_sliding_attention_uses_default_rope() -> None:
    config = Olmo2Config(
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 500000,
            "factor": 8.0,
        },
    )

    assert _get_rope_parameters(config, sliding_window=4096) == {
        "rope_type": "default",
        "rope_theta": 500000,
    }


def test_olmo3_sliding_attention_preserves_configured_rope() -> None:
    rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 500000,
        "factor": 8.0,
        "attention_factor": 1.2079,
    }
    config = Olmo3Config(rope_parameters=rope_parameters)

    actual = _get_rope_parameters(config, sliding_window=4096)

    assert actual == config.rope_parameters
    assert actual["rope_type"] == "yarn"
    assert actual["factor"] == 8.0
    assert actual["attention_factor"] == 1.2079
