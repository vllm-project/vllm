"""Test model set-up and weight loading for sparseml-quantized models.

Run `pytest tests/quantization/test_compressed_tensors.py`.
"""

import torch

from vllm import SamplingParams
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod, CompressedTensorsW8A8DynamicToken,
    CompressedTensorsW8A8StaticTensor)


def test_compressed_tensors_w8a8_static_setup(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8a8-static-v2"
    with vllm_runner(model_path, enforce_eager=True) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj
        o_proj = layer.self_attn.o_proj
        gate_up_proj = layer.mlp.gate_up_proj
        down_proj = layer.mlp.down_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(o_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(gate_up_proj.quant_method,
                          CompressedTensorsLinearMethod)
        assert isinstance(down_proj.quant_method,
                          CompressedTensorsLinearMethod)

        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8StaticTensor)

        assert qkv_proj.weight.dtype is torch.int8
        assert o_proj.weight.dtype is torch.int8
        assert gate_up_proj.weight.dtype is torch.int8

        assert qkv_proj.weight_scale.shard_splitter is not None
        assert qkv_proj.weight_scale.logical_widths is not None
        assert qkv_proj.input_scale.dtype is torch.float32


def test_compressed_tensors_no_enforce_eager(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8a8-static-v2"
    with vllm_runner(model_path) as llm:
        sampling_params = SamplingParams()
        output = llm.generate("Hello world!", sampling_params=sampling_params)
        assert output


def test_compressed_tensors_w8a8_dynanmic_per_token(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2"
    with vllm_runner(model_path, enforce_eager=True,
                     dtype=torch.float16) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8DynamicToken)
        assert qkv_proj.weight.dtype is torch.int8
