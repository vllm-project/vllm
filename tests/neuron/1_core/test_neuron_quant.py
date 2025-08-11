# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.model_executor.layers.quantization.neuron_quant import (
    NeuronQuantConfig)


def test_get_supported_act_dtypes():
    neuron_quant_config = NeuronQuantConfig()
    supported_act_dtypes = neuron_quant_config.get_supported_act_dtypes()
    target_list = ["any_dtype1", "any_dtype2"]
    for dtype in target_list:
        assert dtype in supported_act_dtypes
