# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.step3p5 import _get_step3p5_layer_types
from vllm.transformers_utils.configs.step3p5 import Step3p5Config


def test_step3p5_config_keeps_mtp_layer_types():
    config = Step3p5Config(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        layer_types=["attention", "moe", "attention", "moe"],
    )

    assert config.layer_types == ["attention", "moe"]
    assert config.vllm_layer_types == ["attention", "moe", "attention"]


def test_get_step3p5_layer_types_defaults_to_empty_list():
    config = SimpleNamespace(vllm_layer_types=None, layer_types=None)

    assert _get_step3p5_layer_types(config) == []


def test_step3p5_config_truncates_non_mtp_layer_types():
    config = Step3p5Config(
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        layer_types=["attention", "moe", "attention"],
    )

    assert config.layer_types == ["attention", "moe"]
