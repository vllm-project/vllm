"""Test model set-up and weight loading for sparseml-quantized models.

Run `pytest tests/quantization/test_compressed_tensors.py`.
"""
from typing import List

import pytest
import torch

from vllm import SamplingParams
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod, CompressedTensorsW4A16,
    CompressedTensorsW4A16Sparse24, CompressedTensorsW8A8DynamicToken,
    CompressedTensorsW8A8StaticTensor)


@pytest.mark.parametrize("model_args", [
    ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", "tensor"),
    ("nm-testing/tinyllama-oneshot-w8-channel-a8-tensor", "channel"),
])
def test_compressed_tensors_w8a8_static_setup(vllm_runner, model_args):
    model_path, strategy = model_args
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

        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.weight.dtype is torch.int8
        assert o_proj.weight.dtype is torch.int8
        assert gate_up_proj.weight.dtype is torch.int8

        if qkv_proj.scheme.strategy == "tensor":
            assert qkv_proj.weight_scale.shard_splitter is not None
            assert qkv_proj.weight_scale.logical_widths is not None
        assert qkv_proj.input_scale.dtype is torch.float32


def test_compressed_tensors_no_enforce_eager(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change"
    with vllm_runner(model_path) as llm:
        sampling_params = SamplingParams()
        output = llm.generate("Hello world!", sampling_params=sampling_params)
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
        assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8DynamicToken)
        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.weight.dtype is torch.int8


@pytest.mark.parametrize("w4a16_args", [
    ("nm-testing/tinyllama-oneshot-w4a16-channel-v2", "channel", None),
    ("nm-testing/tinyllama-oneshot-w4a16-group128-v2", "group", 128),
])
def test_compressed_tensors_w4a16(vllm_runner, w4a16_args):
    model, strategy, group = w4a16_args
    with vllm_runner(model) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj
        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW4A16)

        assert qkv_proj.scheme.strategy == strategy
        assert qkv_proj.scheme.group_size == group

        assert qkv_proj.weight_packed.dtype is torch.int32
        assert qkv_proj.weight_scale.dtype is torch.float16
        assert qkv_proj.weight_packed.pack_factor == 8


def test_compressed_tensors_w4a16_marlin24(vllm_runner):
    model_path = "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t"
    with vllm_runner(model_path) as llm:
        model = llm.model.llm_engine.model_executor.driver_worker.model_runner.model  # noqa: E501
        layer = model.model.layers[0]

        qkv_proj = layer.self_attn.qkv_proj

        assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
        assert isinstance(qkv_proj.scheme, CompressedTensorsW4A16Sparse24)
        assert qkv_proj.weight_packed.dtype is torch.int32

        sampling_params = SamplingParams()
        output = llm.generate("Hello world!", sampling_params=sampling_params)
        assert output


@pytest.mark.parametrize("model_path", [
    "nm-testing/tinyllama-oneshot-w8-channel-a8-tensor",
    "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2",
    "nm-testing/tinyllama-oneshot-w4a16-channel-v2",
    "nm-testing/tinyllama-oneshot-w4a16-group128-v2"
])
def test_accuracy_compressed_tensors(vllm_runner, model_path):
    prompts = ["The capital of France is"]
    n_tokens = 50
    n_log_probs = 5

    with vllm_runner(model_path, dtype=torch.bfloat16) as llm:
        vllm_outputs = llm.generate_greedy_logprobs(prompts, n_tokens,
                                                    n_log_probs)
    sparseml_outputs = _run_sparseml(model_path, prompts, n_tokens)
    _compare_ids_top_logprobs(vllm_outputs, sparseml_outputs)


def _run_sparseml(model_path: str, prompts: List[str], n_tokens: int):
    """
    Run sparseml generation using a given model, prompt, and number 
    of tokens to generate. Generations are modified to remove prompt_ids
    from the generations.

    :param model_path: model path
    :param prompts: list of prompts
    :param n_tokens: number of tokens to generate
    :returns: a list of generations. Number of generations 
        should be equal to number of prompts.
    """
    from sparseml.transformers import (  # noqa: E501
        SparseAutoModelForCausalLM, SparseAutoTokenizer)
    model = SparseAutoModelForCausalLM.from_pretrained(model_path,
                                                       device_map="auto")
    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    num_input_ids = inputs.get("input_ids").shape[-1]
    generations = model.generate(**inputs,
                                 min_new_tokens=n_tokens,
                                 max_new_tokens=n_tokens)
    generations_concat = []
    for i in range(len(generations)):
        generations_concat.append(generations[i, num_input_ids:])
    return generations_concat


def _compare_ids_top_logprobs(vllm_outputs, sparseml_outputs):
    """
    Compare and verify if the sparseml generated token ids 
    are in the top_n logprobs generated through vllm.

    :param vllm_outputs: vllm generations
    :param sparseml_outputs: sparseml generations
    """
    for vllm_output, sparseml_output in zip(vllm_outputs, sparseml_outputs):
        logprobs = vllm_output[2]
        for i in range(len(logprobs)):
            assert sparseml_output[i].item() in logprobs[i]
