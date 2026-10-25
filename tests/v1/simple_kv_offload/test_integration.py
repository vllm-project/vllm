# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for SimpleCPUOffloadConnector with real models."""

import time

import pytest
import torch
from transformers import Llama4ForCausalLM, Llama4TextConfig

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams, TokensPrompt, envs
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

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


def test_chunked_local_partial_gpu_hit(vllm_runner, tmp_path, monkeypatch):
    """A CPU suffix after a short GPU hit must preserve the cold prediction."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()
    config = Llama4TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        intermediate_size_mlp=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
        attention_chunk_size=64,
        no_rope_layers=[1, 0],
        interleave_moe_layer_step=100,
        num_local_experts=2,
        use_qk_norm=False,
        initializer_range=0.08,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        model = Llama4ForCausalLM(config)
        model.save_pretrained(tmp_path)
        del model

    with vllm_runner(
        str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float16",
        enforce_eager=True,
        compilation_config={"mode": 0},
        attention_config={"backend": "TRITON_ATTN"},
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        block_size=16,
        num_gpu_blocks_override=64,
        kv_cache_memory_bytes=16 << 20,
        enable_prefix_caching=True,
        disable_hybrid_kv_cache_manager=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 16 << 20},
        ),
    ) as runner:
        llm = runner.get_llm()
        tokens = [3 + (i * 37) % 500 for i in range(81)]
        prompt = TokensPrompt(prompt_token_ids=tokens)
        params = SamplingParams(
            temperature=0, max_tokens=1, logprobs=5, ignore_eos=True, detokenize=False
        )
        cold = llm.generate(prompt, params, use_tqdm=False)[0].outputs[0]

        def reset_gpu_cache():
            deadline = time.monotonic() + 30
            while not llm.reset_prefix_cache():
                assert time.monotonic() < deadline, "CPU stores did not finish"
                llm.llm_engine.step()

        reset_gpu_cache()
        cpu_hit = llm.generate(prompt, params, use_tqdm=False)[0]
        assert cpu_hit.num_cached_tokens == 80
        reset_gpu_cache()
        short_params = params.clone()
        short_params.skip_reading_prefix_cache = True
        llm.generate(
            TokensPrompt(prompt_token_ids=tokens[:17]), short_params, use_tqdm=False
        )
        mixed_hit = llm.generate(prompt, params, use_tqdm=False)[0]
        for name, output in (("CPU hit", cpu_hit), ("mixed hit", mixed_hit)):
            cached = output.outputs[0]
            check_logprobs_close(
                outputs_0_lst=[(list(cold.token_ids), "", cold.logprobs)],
                outputs_1_lst=[(list(cached.token_ids), "", cached.logprobs)],
                name_0="cold",
                name_1=name,
                always_check_logprobs=True,
            )
        assert mixed_hit.num_cached_tokens == 16


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
        kv_cache_memory_bytes=40 << 30,  # 40 GiB
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
    assert num_gpu_blocks is not None
    total_tokens_needed = int(num_gpu_blocks * block_size * 1.5)

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
        time.sleep(2)  # let engine core drain pending transfers

        # Reset GPU prefix cache so next run must load from CPU
        if not llm.reset_prefix_cache():
            print(f"GPU prefix cache reset failed for iteration {i}")

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
        time.sleep(2)  # let engine core drain pending transfers
        if not llm.reset_prefix_cache():
            print(f"GPU prefix cache reset failed for iteration {i}")
        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start

        if lazy:
            _flush_gpu_cache(llm, sampling_params, seed=i)
        else:
            # Eager mode: GPU hit ensures store completion is processed.
            llm.generate(prompts, sampling_params, use_tqdm=False)

        time.sleep(2)  # let engine core drain pending transfers
        if not llm.reset_prefix_cache():
            print(f"GPU prefix cache reset failed for iteration {i}")

        # CPU hit
        start = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_time = time.time() - start

        if cpu_time < cold_time:
            num_times_cpu_better += 1

    assert num_times_cpu_better >= 0.8 * num_tests, (
        f"CPU hit only faster {num_times_cpu_better}/{num_tests} times"
    )


@pytest.mark.optional
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
