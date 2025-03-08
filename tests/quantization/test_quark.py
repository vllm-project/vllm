# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and weight loading for quark-quantized models.

Run `pytest tests/quantization/test_quark.py`.
"""

import torch

from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
    QuarkLinearMethod, QuarkW8A8Fp8)


def test_quark_fp8(vllm_runner):
    model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"
    with vllm_runner(model_path) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)

            if isinstance(qkv_proj.scheme, QuarkW8A8Fp8):
                assert len(qkv_proj.input_scale.shape) == 0
                assert qkv_proj.weight.dtype is torch.float8_e4m3fn
                #assert qkv_proj.weight.dtype is torch.float8_e4m3fnuz
                assert len(qkv_proj.weight_scale.shape) == 0

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output
