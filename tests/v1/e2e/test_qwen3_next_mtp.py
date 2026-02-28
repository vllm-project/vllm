# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

from tests.utils import large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

TEST_PROMPTS = [
    [{"role": "user", "content": "What is 2 + 2?"}],
    [{"role": "user", "content": "Say hello in Chinese."}],
    [{"role": "user", "content": "Count from 1 to 5."}],
    [{"role": "user", "content": "What color is the sky?"}],
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=32, ignore_eos=False)


def _check_model_available(model_name: str) -> bool:
    """Check if model is available."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.model_info(model_name)
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup GPU memory after each test."""
    yield
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


@large_gpu_mark(min_gb=80)
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skipif(
    torch.cuda.device_count() < 1
    or torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3,
    reason="Requires at least 40GB GPU memory",
)
def test_qwen3_next_80b_mtp_cuda_graph(
    num_speculative_tokens: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test Qwen3-Next-80B with MTP and CUDA Graph (Issue #31186 regression)."""
    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

    if not _check_model_available(model_name):
        pytest.skip(f"Model {model_name} not available")

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
    )

    for i in range(3):
        try:
            outputs = llm.chat(TEST_PROMPTS, SAMPLING_PARAMS)
            for output in outputs:
                assert output.outputs[0].text is not None
                assert len(output.outputs[0].text) > 0
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "illegal memory access" in error_msg or "cuda error" in error_msg:
                pytest.fail(f"CUDA error during batch {i + 1} - regression #31186: {e}")
            raise

    del llm


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_QWEN3_NEXT_MTP") != "1",
    reason="Set VLLM_TEST_QWEN3_NEXT_MTP=1 to run",
)
@pytest.mark.parametrize("tensor_parallel_size", [1, 4])
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_qwen3_next_mtp_manual(
    tensor_parallel_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Manual test for Qwen3-Next MTP with various TP sizes."""
    available_gpus = torch.cuda.device_count()
    if available_gpus < tensor_parallel_size:
        pytest.skip(f"Need {tensor_parallel_size} GPUs, have {available_gpus}")

    model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=tensor_parallel_size,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        },
    )

    outputs = llm.chat(TEST_PROMPTS * 5, SAMPLING_PARAMS)

    assert len(outputs) == 20
    for output in outputs:
        assert output.outputs[0].text is not None

    del llm


@pytest.mark.parametrize(
    "model_setup",
    [("XiaomiMiMo/MiMo-7B-Base", "mtp", 20)],
    ids=["mimo_7b"],
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_mtp_cuda_graph_stress(
    model_setup: tuple[str, str, int],
    monkeypatch: pytest.MonkeyPatch,
):
    """Stress test MTP with CUDA Graph using smaller models."""
    model_name, method, min_gpu_gb = model_setup

    if torch.cuda.get_device_properties(0).total_memory < min_gpu_gb * 1024**3:
        pytest.skip(f"Requires at least {min_gpu_gb}GB GPU memory")

    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

    llm = LLM(
        model=model_name,
        max_model_len=2048,
        trust_remote_code=True,
        speculative_config={
            "method": method,
            "num_speculative_tokens": 2,
        },
    )

    for _ in range(5):
        outputs = llm.chat(TEST_PROMPTS, SAMPLING_PARAMS)
        assert len(outputs) == len(TEST_PROMPTS)
        for output in outputs:
            assert output.outputs[0].text is not None

    del llm
