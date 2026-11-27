# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PreTrainedConfig

from vllm.config.speculative import SpeculativeConfig


@pytest.mark.parametrize("model_type", ["step5", "step5_vl"])
def test_step5_mtp_hf_config_override(model_type: str):
    text = PreTrainedConfig(
        architectures=["Step5ForCausalLM"],
        num_hidden_layers=92,
        num_nextn_predict_layers=3,
    )
    text.model_type = "step5"
    if model_type == "step5_vl":
        config = PreTrainedConfig(
            architectures=["Step5ForConditionalGeneration"], text_config=text
        )
        config.model_type = "step5_vl"
    else:
        config = text
    config.quantization_config = {"quant_method": "fp8"}

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden.model_type == "step5_mtp"
    assert overridden.architectures == ["Step5MTP"]
    assert overridden.n_predict == 3
    assert overridden.quantization_config == {"quant_method": "fp8"}
