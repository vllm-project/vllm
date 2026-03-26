# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for SimpleCPUOffloadConnector with real models."""

import time

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("Requires CUDA", allow_module_level=True)

# Small models for default CI / local runs (accuracy only).
SMALL_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-3-1b-it",
]

# Large models for optional perf runs only (slow to load and execute).
PERF_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "openai/gpt-oss-20b",
]


def _make_llm(model: str, lazy: bool, cpu_bytes_to_use: int) -> LLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "lazy_offload": lazy,
        },
    )
    return LLM(
        model=model,
        gpu_memory_utilization=0.6,
        disable_hybrid_kv_cache_manager=False,
        enable_prefix_caching=True,
        kv_transfer_config=kv_transfer_config,
    )


def _flush_gpu_cache(llm: LLM, sampling_params: SamplingParams, seed: int = 0):
    """Generate enough filler requests to allocate the entire GPU KV cache.

    This pushes all prior blocks through the free queue so that the lazy
    cursor offloads them to CPU before they are evicted.
    """
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    # Use 1.2x GPU capacity to give the lazy cursor enough scheduling steps
    # to walk past all target blocks near the tail of the free queue.
    total_tokens_needed = int(num_gpu_blocks * block_size * 1.2)

    # Use token-id prompts so each filler is unique (no prefix sharing).
    # Split into multiple requests to stay under max_model_len.
    max_tokens_per_req = 4096
    num_fillers = (total_tokens_needed + max_tokens_per_req - 1) // max_tokens_per_req
    batch_size = 10
    for i in range(0, num_fillers, batch_size):
        batch_end = min(i + batch_size, num_fillers)
        filler_prompts = []
        for j in range(i, batch_end):
            ids = [seed * num_fillers + j + 1] * max_tokens_per_req
            filler_prompts.append(TokensPrompt(prompt_token_ids=ids))
        llm.generate(filler_prompts, sampling_params, use_tqdm=False)


def _accuracy_test(llm: LLM, lazy: bool = False):
    """Verify that CPU-loaded KV produces correct output."""
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    prompt = "hi " * 2000 + "Let's count to ten. One, two, three, "

    # Cold run — populate GPU cache and trigger CPU offload
    cold_output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

    # CPU hit runs
    test_count = 10
    success_count = 0
    expected = cold_output.outputs[0].text
    for i in range(test_count):
        if lazy:
            _flush_gpu_cache(llm, sampling_params, seed=i)

        # Reset GPU prefix cache so next run must load from CPU
        assert llm.reset_prefix_cache(), "GPU prefix cache reset failed"

        output = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        if output.outputs[0].text == expected:
            success_count += 1

    assert success_count >= 0.5 * test_count, (
        f"Accuracy too low: {success_count}/{test_count} matched '{expected}'"
    )


def _latency_test(llm: LLM, lazy: bool = False):
    """Verify CPU cache hit is faster than cold compute."""
    sampling_params = SamplingParams(max_tokens=1, seed=42)
    prompt_token_ids = [0] * 10001

    num_times_cpu_better = 0
    num_tests = 10
    for i in range(num_tests):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # Cold
        assert llm.reset_prefix_cache(), "GPU prefix cache reset failed"
        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start

        if lazy:
            _flush_gpu_cache(llm, sampling_params, seed=i)
        else:
            # Eager mode: GPU hit ensures store completion is processed.
            llm.generate(prompts, sampling_params, use_tqdm=False)

        assert llm.reset_prefix_cache(), "GPU prefix cache reset failed"

        # CPU hit
        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_time = time.time() - start

        if cpu_time < cold_time:
            num_times_cpu_better += 1

    assert num_times_cpu_better >= 0.8 * num_tests, (
        f"CPU hit only faster {num_times_cpu_better}/{num_tests} times"
    )


@pytest.mark.slow_test
@pytest.mark.parametrize("model", SMALL_MODELS)
def test_simple_cpu_offload_accuracy(model: str):
    """Store to CPU, reset GPU, load from CPU; verify output matches baseline."""
    llm = _make_llm(model, False, 1 << 30)  # 1GB
    try:
        _accuracy_test(llm, lazy=False)
    finally:
        del llm


@pytest.mark.optional
@pytest.mark.slow_test
@pytest.mark.parametrize("model", PERF_MODELS)
def test_simple_cpu_offload_perf_latency(model: str):
    """CPU KV hit should beat cold prefill on long context (large models only)."""
    llm = _make_llm(model, False, 10 << 30)  # 10GB
    try:
        _latency_test(llm, lazy=False)
    finally:
        del llm


@pytest.mark.optional
@pytest.mark.slow_test
@pytest.mark.parametrize("model", SMALL_MODELS)
def test_simple_cpu_offload_accuracy_lazy(model: str):
    """Lazy mode: flush GPU cache to trigger CPU offload, then verify hit."""
    # CPU must be larger than GPU KV cache to avoid evicting offloaded blocks.
    llm = _make_llm(model, True, 80 << 30)  # 80GB
    try:
        _accuracy_test(llm, lazy=True)
    finally:
        del llm


@pytest.mark.optional
@pytest.mark.slow_test
@pytest.mark.parametrize("model", PERF_MODELS)
def test_simple_cpu_offload_perf_latency_lazy(model: str):
    """Lazy mode: CPU KV hit should beat cold prefill (large models only)."""
    # CPU must be larger than GPU KV cache to avoid evicting offloaded blocks.
    llm = _make_llm(model, True, 80 << 30)  # 80GB
    try:
        _latency_test(llm, lazy=True)
    finally:
        del llm
