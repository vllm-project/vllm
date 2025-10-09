# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Optional

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

if not current_platform.is_device_capability(100):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM100).", allow_module_level=True
    )


@pytest.fixture(scope="module", autouse=True)
def set_test_environment():
    """Sets environment variables required for this test module."""
    # Make sure TRTLLM attention is available
    os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
    # Set compilation threads to 16 to speed up startup
    os.environ["FLASHINFER_NVCC_THREADS"] = "16"


# dummy_hf_overrides = {"num_layers": 4, "num_hidden_layers": 4,
# "text_config": {"num_layers": 4, "num_hidden_layers": 4}}
dummy_hf_overrides = {"num_layers": 4, "num_hidden_layers": 4}


def can_initialize(model: str, extra_args: Optional[list[str]] = None):
    # Server arguments
    extra_args = extra_args if extra_args is not None else []
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0}),
        *extra_args,
    ]

    # Launch server and make a simple request
    with RemoteOpenAIServer(
        model,
        server_args,
        max_wait_seconds=1000,  # Due to FlashInfer compile
        override_hf_configs=dummy_hf_overrides,
    ) as server:
        client = server.get_client()
        # Make a simple request to verify the server works
        completion = client.completions.create(
            model=model,
            prompt=["Hello, World!"],
            temperature=0,
            max_tokens=2,
        )
        print(completion)
        assert completion.choices[0].text is not None


## Llama4 ##


@pytest.mark.skip(
    reason=(
        "RuntimeError: run_moe() Expected a value of type "
        "'Optional[List[Tensor]]' for argument '_9' but instead found type "
        "'list'."
    )
)
def test_llama4_fp8_tensor_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("nvidia/Llama-4-Scout-17B-16E-Instruct-FP8")


@pytest.mark.skip(reason="Works, but takes too long to run")
def test_llama4_fp8_tensor_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    can_initialize("nvidia/Llama-4-Scout-17B-16E-Instruct-FP8")


@pytest.mark.skip(reason="Works, but takes too long to run")
def test_llama4_nvfp4_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("nvidia/Llama-4-Scout-17B-16E-Instruct-FP4")


@pytest.mark.skip(reason="RuntimeError: No kernel found for the given options")
def test_llama4_nvfp4_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    can_initialize("nvidia/Llama-4-Scout-17B-16E-Instruct-FP4")


## DeepSeekV3 ##


def test_deepseek_fp8_block_moe_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    can_initialize("deepseek-ai/DeepSeek-V3.1")


@pytest.mark.skip(
    reason=(
        "Known issue: lack of kernel support. "
        "Expected failure: assert self.block_quant is None"
    )
)
def test_deepseek_fp8_block_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("deepseek-ai/DeepSeek-V3.1")


def test_deepseek_fp8_block_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    can_initialize("deepseek-ai/DeepSeek-V3.1")


def test_deepseek_nvfp4_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("nvidia/DeepSeek-R1-0528-FP4-v2")


@pytest.mark.skip(reason="RuntimeError: No kernel found for the given options")
def test_deepseek_nvfp4_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    can_initialize("nvidia/DeepSeek-R1-0528-FP4-v2")


## GPT-OSS ##


def test_gptoss_mxfp4bf16_moe_flashinfer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_BF16", "1")
    can_initialize("openai/gpt-oss-20b")


def test_gptoss_mxfp4mxfp8_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS", "1")
    can_initialize("openai/gpt-oss-20b")


def test_gptoss_mxfp4mxfp8_moe_flashinfer_trtllm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "1")
    can_initialize("openai/gpt-oss-20b")
