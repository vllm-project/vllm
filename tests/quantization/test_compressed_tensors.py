"""Test model set-up and weight loading for llmcompressor-quantized models.

Run `pytest tests/quantization/test_compressed_tensors.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod, CompressedTensorsW4A16Sparse24,
    CompressedTensorsW8A8Fp8, CompressedTensorsW8A8Int8,
    CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    QuantizationType)


@pytest.mark.parametrize("model_args", [
    ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", "tensor",
     QuantizationType.INT, 2560),
    ("nm-testing/tinyllama-oneshot-w8-channel-a8-tensor", "channel",
     QuantizationType.INT, 2560),
])
def test_compressed_tensors_w8a8_static_setup(vllm_runner, model_args):
    model_path, strategy, quant_type, shape_0 = model_args
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
        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8Int8)

        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.scheme.is_static_input_scheme
        expected_type = torch.int8

        assert qkv_proj.weight.dtype is expected_type
        assert o_proj.weight.dtype is expected_type
        assert gate_up_proj.weight.dtype is expected_type

        if qkv_proj.scheme.strategy == "tensor":
            # Make sure it is a channelwise buffer
            # After running process_weights_after_loading
            assert len(qkv_proj.weight_scale.shape) == 2
            assert qkv_proj.weight_scale.shape[0] == shape_0
            assert qkv_proj.weight_scale.shape[1] == 1
        assert qkv_proj.weight_scale.dtype is torch.float32
        assert qkv_proj.input_scale.dtype is torch.float32

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_no_enforce_eager(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change"
    with vllm_runner(model_path) as llm:
        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


@pytest.mark.parametrize("model_args", [
    ("nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2", "tensor"),
    ("nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2", "channel"),
])
def test_compressed_tensors_w8a8_dynanmic_per_token(vllm_runner, model_args):
    model_path, strategy = model_args
    with vllm_runner(model_path, dtype=torch.float16) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8Int8)
        assert not qkv_proj.scheme.is_static_input_scheme
        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.weight.dtype is torch.int8

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


@pytest.mark.parametrize(
    "wNa16_args",
    [("nm-testing/tinyllama-oneshot-w4a16-channel-v2", "channel", None, 8),
     ("nm-testing/tinyllama-oneshot-w4a16-group128-v2", "group", 128, 8),
     ("nm-testing/tinyllama-oneshot-w8a16-per-channel", "channel", None, 4)])
def test_compressed_tensors_wNa16(vllm_runner, wNa16_args):
    model, strategy, group, pack_factor = wNa16_args
    with vllm_runner(model) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj
        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsWNA16)

        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.scheme.group_size == (-1 if group is None else group)

        assert qkv_proj.weight_packed.dtype is torch.int32
        assert qkv_proj.weight_scale.dtype is torch.float16
        assert qkv_proj.weight_packed.pack_factor == pack_factor

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_w4a16_marlin24(vllm_runner):
    model_path = "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t"
    with vllm_runner(model_path) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW4A16Sparse24)
        assert qkv_proj.weight_packed.dtype is torch.int32

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_fp8(vllm_runner):
    model_path = "nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test"
    with vllm_runner(model_path) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8Fp8)
        assert qkv_proj.weight.dtype is torch.float8_e4m3fn
        assert qkv_proj.input_scale.dtype is torch.float32
        assert qkv_proj.weight_scale.dtype is torch.float32
        # should be scalars after processing
        assert len(qkv_proj.input_scale.shape) == 0
        assert len(qkv_proj.weight_scale.shape) == 0

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_kv_cache(vllm_runner):
    model_path = "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme"
    with vllm_runner(model_path, kv_cache_dtype="fp8") as llm:
        output = llm.generate_greedy("Hello world!", max_tokens=20)
        assert output
