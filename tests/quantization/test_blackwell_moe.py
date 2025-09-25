# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.platforms import current_platform
from tests.utils import create_new_process_for_each_test

if not current_platform.is_device_capability(100):
    pytest.skip("This test only runs on Blackwell GPUs (SM100).",
                allow_module_level=True)

def dummy_hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
    """
    Dummy HF overrides function used to create dummy model
    with only minimum nums of layer.
    """
    text_config = hf_config.get_text_config()

    # Do 4 backbone layers to include dense and MoE layers
    text_config.update({
        "num_layers": 4,
        "num_hidden_layers": 4,
    })

    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
        })
    # e.g.: ibm-granite/granite-speech-3.3-2b
    if hasattr(hf_config, "encoder_config"):
        hf_config.encoder_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
        })
    # e.g.: Qwen/Qwen2-Audio-7B-Instruct
    if hasattr(hf_config, "audio_config"):
        hf_config.audio_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
            "encoder_layers": 1,
        })

    return hf_config


# @create_new_process_for_each_test()
def can_initialize(vllm_runner, model_name: str, **model_kwargs):

    default_model_kwargs = {
        "enforce_eager": True,
        "trust_remote_code": True,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.8,
        "load_format": "dummy",
        # "hf_overrides": dummy_hf_overrides,
    }
    default_model_kwargs.update(model_kwargs)

    with vllm_runner(model_name, **default_model_kwargs) as llm:
        sp = SamplingParams(temperature=0, max_tokens=2)
        llm.generate("Hello, world!", sampling_params=sp)

def test_blackwell_fp8_tensor_moe_flashinfer_trtllm(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    can_initialize(vllm_runner, "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", tensor_parallel_size=1)

def test_blackwell_fp8_block_moe_deep_gemm(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    can_initialize(vllm_runner, "deepseek-ai/DeepSeek-V3.1", tensor_parallel_size=1)

def test_blackwell_nvfp4_moe_flashinfer_cutlass(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize(vllm_runner, "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4", tensor_parallel_size=1)

def test_blackwell_nvfp4_moe_flashinfer_trtllm(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    can_initialize(vllm_runner, "nvidia/Llama-4-Scout-17B-16E-Instruct-FP4", tensor_parallel_size=1)
    can_initialize(vllm_runner, "nvidia/DeepSeek-R1-0528-FP4-v2", tensor_parallel_size=1)

