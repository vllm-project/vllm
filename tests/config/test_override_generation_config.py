# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config.model import ModelConfig


def test_override_generation_config_keeps_presence_and_frequency_penalty():
    """Regression test for #50767: --override-generation-config must not
    silently drop presence_penalty / frequency_penalty."""
    model_config = object.__new__(ModelConfig)
    model_config.generation_config = "vllm"
    model_config.override_generation_config = {
        "presence_penalty": 1.2,
        "frequency_penalty": 0.8,
        "temperature": 0.7,
    }

    assert model_config.get_diff_sampling_param() == {
        "presence_penalty": 1.2,
        "frequency_penalty": 0.8,
        "temperature": 0.7,
    }


def test_generation_config_penalties_survive_diff_sampling_filter():
    """Regression test for #50767: penalties read from the model's
    generation_config.json must survive the diff-sampling whitelist."""
    model_config = object.__new__(ModelConfig)
    model_config.generation_config = "auto"
    model_config.override_generation_config = {}
    model_config.try_get_generation_config = lambda: {
        "presence_penalty": 1.1,
        "frequency_penalty": 0.7,
        "temperature": 0.9,
    }

    assert model_config.get_diff_sampling_param() == {
        "presence_penalty": 1.1,
        "frequency_penalty": 0.7,
        "temperature": 0.9,
    }
