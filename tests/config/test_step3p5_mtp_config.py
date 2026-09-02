# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step3p5Config crops layer_types for HF strict validation and keeps
the MTP tail in mtp_layer_types; SpeculativeConfig.hf_config_override
re-appends the tail for the draft, whose MTP layers index layer_types
at [num_hidden_layers + i] (#40000).
"""

from typing import Any

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.step3p5 import Step3p5Config

# Step-3.5-Flash layout: 45 main layers + 3 MTP layers.
_MAIN = ["sliding_attention", "full_attention"] * 22 + ["sliding_attention"]
_MTP = ["sliding_attention"] * 3


def _step3p5_config(**kwargs: Any) -> Step3p5Config:
    defaults: dict[str, Any] = dict(
        architectures=["Step3p5ForCausalLM"],
        num_hidden_layers=45,
        num_nextn_predict_layers=3,
        layer_types=_MAIN + _MTP,
    )
    defaults.update(kwargs)
    return Step3p5Config(**defaults)


@pytest.mark.cpu_test
def test_step3p5_config_crops_and_keeps_mtp_tail():
    config = _step3p5_config()

    assert config.layer_types == _MAIN
    assert config.mtp_layer_types == _MTP
    # The cropped config must stay HF-strict-valid.
    config.validate()


@pytest.mark.cpu_test
def test_hf_config_override_restores_mtp_layer_types():
    out = SpeculativeConfig.hf_config_override(_step3p5_config())

    assert out.model_type == "step3p5_mtp"
    assert out.architectures == ["Step3p5MTP"]
    # Every MTP index resolves to the checkpoint's real type.
    assert out.layer_types == _MAIN + _MTP


@pytest.mark.cpu_test
def test_hf_config_override_raises_without_mtp_tail():
    # Tail-less checkpoint: actionable error, not an IndexError at
    # draft construction.
    config = _step3p5_config(layer_types=_MAIN)

    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        SpeculativeConfig.hf_config_override(config)


@pytest.mark.cpu_test
@pytest.mark.parametrize("layer_types", [_MAIN + _MTP, _MAIN])
def test_hf_config_override_ignores_foreign_configs(layer_types):
    # Keyed on class, not list shape: a non-Step3p5Config (Step-3.7's hub
    # text config, future models) passes through untouched, whether its
    # layer_types is the full list or cropped.
    config = PretrainedConfig(
        architectures=["Step3p7ForConditionalGeneration"],
        model_type="step3p7",
        num_hidden_layers=45,
        num_nextn_predict_layers=3,
    )
    config.layer_types = layer_types

    out = SpeculativeConfig.hf_config_override(config)

    assert out is config
    assert out.layer_types == layer_types
