# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and weight loading for quark-quantized models.

Run `pytest tests/quantization/test_quark.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
    QuarkLinearMethod, QuarkW8A8Fp8, QuarkW8A8Int8)
from vllm.platforms import current_platform


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module relies on V0 internals, so set VLLM_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_USE_V1', '0')


@pytest.mark.parametrize('kv_cache_dtype', ['auto', 'fp8'])
@pytest.mark.parametrize('tp', [1])
def test_quark_fp8_w_per_tensor_a_per_tensor(vllm_runner, kv_cache_dtype, tp):
    model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"
    with vllm_runner(model_path,
                     kv_cache_dtype=kv_cache_dtype,
                     tensor_parallel_size=tp) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)

            if isinstance(qkv_proj.scheme, QuarkW8A8Fp8):
                assert len(qkv_proj.input_scale.shape) == 0
                assert qkv_proj.weight.dtype is current_platform.fp8_dtype()
                assert len(qkv_proj.weight_scale.shape) == 0

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


@pytest.mark.parametrize('tp', [1])
def test_quark_int8_w_per_tensor_a_per_tensor(vllm_runner, tp):
    model_path = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"
    with vllm_runner(model_path, tensor_parallel_size=tp) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_quark_fp8_parity(vllm_runner):
    quark_model_id = "amd-quark/llama-tiny-fp8-quark-quant-method"
    fp8_model_id = "amd-quark/llama-tiny-fp8-quant-method"

    llm_kwargs = {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.1
    }
    with (vllm_runner(quark_model_id, **llm_kwargs) as
          quark_handle, vllm_runner(fp8_model_id, **llm_kwargs) as fp8_handle):
        quark_model = (quark_handle.model.llm_engine.model_executor.
                       driver_worker.model_runner.model)
        quark_state_dict = quark_model.state_dict()

        fp8_model = (fp8_handle.model.llm_engine.model_executor.driver_worker.
                     model_runner.model)
        fp8_state_dict = fp8_model.state_dict()

    assert fp8_state_dict.keys() == quark_state_dict.keys()

    for key in fp8_state_dict:
        assert torch.equal(fp8_state_dict[key], quark_state_dict[key])
