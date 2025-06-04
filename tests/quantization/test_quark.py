# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and weight loading for quark-quantized models.

Run `pytest tests/quantization/test_quark.py`.
"""

import importlib
import importlib.metadata
import os
from dataclasses import dataclass

import huggingface_hub
import lm_eval
import pytest
import torch
from packaging import version

from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
    QuarkLinearMethod, QuarkW4A4MXFP4, QuarkW8A8Fp8, QuarkW8A8Int8)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW4A4MXFp4MoEMethod)
from vllm.platforms import current_platform

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec(
    "quark") is not None and version.parse(
        importlib.metadata.version("amd-quark")) >= version.parse('0.9')

try:
    huggingface_hub.list_repo_refs(
        "amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-SQ")
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


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


@dataclass
class ModelCase:
    model_id: str
    tp: int


@pytest.mark.parametrize('model_case', [
    ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp=1),
    ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp=8),
    ModelCase("fxmarty/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp=1)
])
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
def test_mxfp4_loading_and_execution(vllm_runner, model_case: ModelCase):
    if torch.cuda.device_count() < model_case.tp:
        pytest.skip(f"This test requires >={model_case.tp} gpus, got only "
                    f"{torch.cuda.device_count()}")

    with vllm_runner(model_case.model_id,
                     tensor_parallel_size=model_case.tp,
                     load_format="dummy") as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
            assert isinstance(qkv_proj.scheme, QuarkW4A4MXFP4)

            assert isinstance(layer.mlp.experts.quant_method,
                              QuarkW4A4MXFp4MoEMethod)

        if model_case.model_id == "fxmarty/qwen_1.5-moe-a2.7b-mxfp4":
            llm.apply_model(check_model)

        output = llm.generate_greedy("Today I am in the French Alps and",
                                     max_tokens=20)
        assert output


@dataclass
class GSM8KAccuracyTestConfig:
    model_name: str
    excepted_value: float

    def get_model_args(self) -> str:
        return (
            f"pretrained={self.model_name},"
            "dtype=auto,add_bos_token=True,tensor_parallel_size=8,gpu_memory_utilization=0.7,max_model_len=38768"
        )


ACCURACY_CONFIGS = [
    # Private model.
    GSM8KAccuracyTestConfig(
        model_name="amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant",
        excepted_value=0.96),
]


@pytest.mark.parametrize("config", ACCURACY_CONFIGS)
@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE,
                    reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.")
def test_mxfp4_gsm8k_correctness(config: GSM8KAccuracyTestConfig):
    if torch.cuda.device_count() < 8:
        pytest.skip(
            f"This test requires >=8 gpus, got only {torch.cuda.device_count()}"
        )

    task = "gsm8k"
    rtol = 0.03

    os.environ["VLLM_QUARK_EMU_MEM_OPT"] = "1"
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(),
        tasks=task,
        batch_size=64,
        num_fewshot=8,
    )

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["exact_match,strict-match"]
    assert (measured_value - rtol < EXPECTED_VALUE
            and measured_value + rtol > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"

    del os.environ["VLLM_QUARK_EMU_MEM_OPT"]
    del os.environ["VLLM_USE_TRITON_FLASH_ATTN"]
