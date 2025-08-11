# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test ModelOpt quantization method setup and weight loading.

Run `pytest tests/quantization/test_modelopt.py`.
"""

import os

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.platforms import current_platform


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module relies on V0 internals, so set VLLM_USE_V1=0.
    """
    if not current_platform.is_cpu():
        monkeypatch.setenv('VLLM_USE_V1', '0')


@pytest.mark.skipif(not is_quant_method_supported("modelopt"),
                    reason="ModelOpt FP8 is not supported on this GPU type.")
def test_modelopt_fp8_checkpoint_setup(vllm_runner):
    """Test ModelOpt FP8 checkpoint loading and structure validation."""
    # TODO: provide a small publically available test checkpoint
    model_path = ("/home/scratch.omniml_data_1/zhiyu/ckpts/test_ckpts/"
                  "TinyLlama-1.1B-Chat-v1.0-fp8-0710")

    # Skip test if checkpoint doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Test checkpoint not found at {model_path}. "
                    "This test requires a local ModelOpt FP8 checkpoint.")

    with vllm_runner(model_path, quantization="modelopt",
                     enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # Check that ModelOpt quantization method is properly applied
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8LinearMethod)
            assert isinstance(qkv_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(gate_up_proj.quant_method,
                              ModelOptFp8LinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8LinearMethod)

            # Check weight dtype is FP8
            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Check scales are present and have correct dtype
            assert hasattr(qkv_proj, 'weight_scale')
            assert hasattr(qkv_proj, 'input_scale')
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.input_scale.dtype == torch.float32

            assert hasattr(o_proj, 'weight_scale')
            assert hasattr(o_proj, 'input_scale')
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.input_scale.dtype == torch.float32

            assert hasattr(gate_up_proj, 'weight_scale')
            assert hasattr(gate_up_proj, 'input_scale')
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.input_scale.dtype == torch.float32

            assert hasattr(down_proj, 'weight_scale')
            assert hasattr(down_proj, 'input_scale')
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.input_scale.dtype == torch.float32

        llm.apply_model(check_model)

        # Run a simple generation test to ensure the model works
        output = llm.generate_greedy(["Hello my name is"], max_tokens=20)
        assert output
        print(f"ModelOpt FP8 output: {output}")
