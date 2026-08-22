# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

if not current_platform.is_device_capability(90):
    pytest.skip("This test only runs on Hopper GPUs (SM90).", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def set_test_environment():
    """Sets environment variables required for this test module."""
    os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
    os.environ["FLASHINFER_NVCC_THREADS"] = "16"


# Override the backbone layers to 2 for faster startup
HF_OVERRIDE_TEXT = {
    "num_layers": 2,
    "num_hidden_layers": 2,
}
HF_OVERRIDE_MM = {
    "text_config": {"num_layers": 2, "num_hidden_layers": 2},
}


def can_initialize(
    model: str,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
):
    extra_args = extra_args if extra_args is not None else []
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--gpu-memory-utilization",
        "0.8",
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        json.dumps({"image": 0}),
        *extra_args,
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        max_wait_seconds=1500,
        override_hf_configs=hf_overrides,
    ) as server:
        client = server.get_client()
        completion = client.completions.create(
            model=model,
            prompt=["Hello, World!"],
            temperature=0,
            max_tokens=2,
        )
        assert completion.choices[0].text is not None


def test_deepseek_fp8_block_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("deepseek-ai/DeepSeek-V3.1", hf_overrides=HF_OVERRIDE_TEXT)


def test_llama4_fp8_tensor_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8", hf_overrides=HF_OVERRIDE_MM
    )


def test_deepseek_fp8_block_moe_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    can_initialize("deepseek-ai/DeepSeek-V3.1", hf_overrides=HF_OVERRIDE_TEXT)


def test_qwen3_next_bf16_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    can_initialize("Qwen/Qwen3-Next-80B-A3B-Instruct", hf_overrides=HF_OVERRIDE_TEXT)
