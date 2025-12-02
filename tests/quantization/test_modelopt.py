# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test ModelOpt quantization method setup and weight loading.

Run `pytest tests/quantization/test_modelopt.py`.
"""

import os
from dataclasses import dataclass

import lm_eval
import pytest
import torch

from tests.quantization.utils import is_quant_method_supported


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@dataclass
class AccuracyTestConfig:
    model_name: str
    expected_value: float
    rtol: float = 0.03

    def get_model_args(
        self,
        tp_size: int,
        model_max_len: int | None = None,
        kwargs: dict | None = None,
    ) -> dict:
        if kwargs is None:
            kwargs = {}

        model_args = {
            "pretrained": self.model_name,
            "quantization": "modelopt",
            "dtype": "auto",
            "add_bos_token": True,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": 0.7,
            **kwargs,
        }
        if model_max_len is not None:
            model_args["max_model_len"] = model_max_len

        return model_args


GSM8K_FP8_ACCURACY_CONFIGS = [
    # Llama 3.1 8B Instruct FP8 quantized with ModelOpt
    AccuracyTestConfig(
        model_name="nvidia/Llama-3.1-8B-Instruct-FP8",
        expected_value=0.70,
        rtol=0.05,
    ),
    # Qwen 3 8B FP8 quantized with ModelOpt
    AccuracyTestConfig(
        model_name="nvidia/Qwen3-8B-FP8",
        expected_value=0.90,
        rtol=0.05,
    ),
]

GSM8K_FP4_ACCURACY_CONFIGS = [
    # Llama 3.1 8B Instruct FP4 quantized with ModelOpt
    AccuracyTestConfig(
        model_name="nvidia/Llama-3.1-8B-Instruct-FP4",
        expected_value=0.69,
        rtol=0.8,
    ),
    # Qwen 3 8B FP4 quantized with ModelOpt
    AccuracyTestConfig(
        model_name="nvidia/Qwen3-8B-FP4",
        expected_value=0.90,
        rtol=0.8,
    ),
]


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
def test_modelopt_fp8_checkpoint_setup(vllm_runner):
    """Test ModelOpt FP8 checkpoint loading and structure validation."""
    # TODO: provide a small publicly available test checkpoint
    model_path = (
        "/home/scratch.omniml_data_1/zhiyu/ckpts/test_ckpts/"
        "TinyLlama-1.1B-Chat-v1.0-fp8-0710"
    )

    # Skip test if checkpoint doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(
            f"Test checkpoint not found at {model_path}. "
            "This test requires a local ModelOpt FP8 checkpoint."
        )

    with vllm_runner(model_path, quantization="modelopt", enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # Check that ModelOpt quantization method is properly applied
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8LinearMethod,
            )

            assert isinstance(qkv_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(o_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(gate_up_proj.quant_method, ModelOptFp8LinearMethod)
            assert isinstance(down_proj.quant_method, ModelOptFp8LinearMethod)

            # Check weight dtype is FP8
            assert qkv_proj.weight.dtype == torch.float8_e4m3fn
            assert o_proj.weight.dtype == torch.float8_e4m3fn
            assert gate_up_proj.weight.dtype == torch.float8_e4m3fn
            assert down_proj.weight.dtype == torch.float8_e4m3fn

            # Check scales are present and have correct dtype
            assert hasattr(qkv_proj, "weight_scale")
            assert hasattr(qkv_proj, "input_scale")
            assert qkv_proj.weight_scale.dtype == torch.float32
            assert qkv_proj.input_scale.dtype == torch.float32

            assert hasattr(o_proj, "weight_scale")
            assert hasattr(o_proj, "input_scale")
            assert o_proj.weight_scale.dtype == torch.float32
            assert o_proj.input_scale.dtype == torch.float32

            assert hasattr(gate_up_proj, "weight_scale")
            assert hasattr(gate_up_proj, "input_scale")
            assert gate_up_proj.weight_scale.dtype == torch.float32
            assert gate_up_proj.input_scale.dtype == torch.float32

            assert hasattr(down_proj, "weight_scale")
            assert hasattr(down_proj, "input_scale")
            assert down_proj.weight_scale.dtype == torch.float32
            assert down_proj.input_scale.dtype == torch.float32

        llm.apply_model(check_model)

        # Run a simple generation test to ensure the model works
        output = llm.generate_greedy(["Hello my name is"], max_tokens=4)
        assert output
        print(f"ModelOpt FP8 output: {output}")


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt"),
    reason="ModelOpt FP8 is not supported on this GPU type.",
)
@pytest.mark.parametrize("config", GSM8K_FP8_ACCURACY_CONFIGS)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_modelopt_fp8_gsm8k_accuracy(config: AccuracyTestConfig, tp_size: int):
    """Test ModelOpt FP8 quantization accuracy on GSM8K benchmark."""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(
            f"This test requires >={tp_size} GPUs, got only {torch.cuda.device_count()}"
        )

    task = "gsm8k"

    # Run GSM8K evaluation using lm_eval
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(tp_size=tp_size),
        tasks=task,
        batch_size=64,
        num_fewshot=8,
        limit=200,
    )

    EXPECTED_VALUE = config.expected_value
    measured_value = results["results"][task]["exact_match,strict-match"]

    assert (
        measured_value - config.rtol < EXPECTED_VALUE
        and measured_value + config.rtol > EXPECTED_VALUE
    ), f"Expected: {EXPECTED_VALUE} ± {config.rtol} | Measured: {measured_value}"


@pytest.mark.skipif(
    not is_quant_method_supported("modelopt_fp4"),
    reason="ModelOpt FP4 is not supported on this GPU type.",
)
@pytest.mark.parametrize("config", GSM8K_FP4_ACCURACY_CONFIGS)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_modelopt_fp4_gsm8k_accuracy(config: AccuracyTestConfig, tp_size: int):
    """Test ModelOpt FP4 quantization accuracy on GSM8K benchmark."""
    if torch.cuda.device_count() < tp_size:
        pytest.skip(
            f"This test requires >={tp_size} GPUs, got only {torch.cuda.device_count()}"
        )

    task = "gsm8k"

    # Run GSM8K evaluation using lm_eval
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(tp_size=tp_size),
        tasks=task,
        batch_size=64,
        num_fewshot=8,
        limit=200,
    )

    EXPECTED_VALUE = config.expected_value
    measured_value = results["results"][task]["exact_match,strict-match"]

    assert (
        measured_value - config.rtol < EXPECTED_VALUE
        and measured_value + config.rtol > EXPECTED_VALUE
    ), f"Expected: {EXPECTED_VALUE} ± {config.rtol} | Measured: {measured_value}"
