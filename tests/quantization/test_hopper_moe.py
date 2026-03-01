# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(90):
    pytest.skip("This test only runs on Hopper GPUs (SM90x).", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def set_test_environment():
    """Sets environment variables required for this test module."""
    # Make sure TRTLLM attention is available
    os.environ["VLLM_HAS_FLASHINFER_CUBIN"] = "1"
    # Set compilation threads to 16 to speed up startup
    os.environ["FLASHINFER_NVCC_THREADS"] = "16"


# Overide the backbone layers to 4 for faster startup
HF_OVERRIDE_TEXT = {
    "num_layers": 4,
    "num_hidden_layers": 4,
}
HF_OVERRIDE_MM = {
    "text_config": {"num_layers": 4, "num_hidden_layers": 4},
}


def can_initialize(
    model: str,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
):
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
        max_wait_seconds=1500,  # Due to FlashInfer compile
        override_hf_configs=hf_overrides,
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


def test_llama4_fp8_tensor_moe_vllm_triton(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=triton"],
    )


def test_llama4_fp8_tensor_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
        hf_overrides=HF_OVERRIDE_MM,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


## DeepSeekV3 ##


def test_deepseek_fp8_block_moe_deep_gemm(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=deep_gemm"],
    )


def test_deepseek_fp8_block_moe_vllm_triton(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "deepseek-ai/DeepSeek-V3.1",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=tritont"],
    )


## Qwen3 Next ##


def test_qwen3_next_bf16_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )


## NemoTron ##


def test_nemotron_fp8_moe_flashinfer_cutlass(monkeypatch: pytest.MonkeyPatch):
    can_initialize(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--moe-backend=flashinfer_cutlass"],
    )
