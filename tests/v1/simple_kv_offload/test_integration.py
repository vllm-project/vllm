# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for SimpleCPUOffloadConnector with real models."""

import time

import pytest
import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Llama4ForCausalLM,
    Llama4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
)

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams, TokensPrompt, envs
from vllm.config import KVTransferConfig, ParallelConfig
from vllm.platforms import current_platform
from vllm.utils import numa_utils
from vllm.v1.metrics.reader import Counter
from vllm.v1.simple_kv_offload.metrics import MetricName

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


def _prompt(seed: int, num_blocks: int) -> TokensPrompt:
    """A seeded prompt of `num_blocks` full 16-token blocks plus one token."""
    tokens = [3 + (seed * 101 + i * 37) % 500 for i in range(num_blocks * 16 + 1)]
    return TokensPrompt(prompt_token_ids=tokens)


def _params(max_tokens: int = 1) -> SamplingParams:
    return SamplingParams(
        temperature=0, max_tokens=max_tokens, ignore_eos=True, detokenize=False
    )


def _drop_gpu_cache(llm: LLM) -> None:
    """Reset the GPU prefix cache once the CPU stores land, so the next
    request can only hit the CPU tier. The engine keeps stepping while
    stores are in flight."""
    deadline = time.monotonic() + 30
    while not llm.reset_prefix_cache():
        assert time.monotonic() < deadline, "CPU stores did not finish"
        time.sleep(0.05)


def _generate_and_drop_gpu_cache(llm: LLM, seed: int, num_blocks: int) -> int | None:
    """Generate one token for a seeded prompt and return its cached tokens."""
    output = llm.generate(_prompt(seed, num_blocks), _params(), use_tqdm=False)[0]
    _drop_gpu_cache(llm)
    return output.num_cached_tokens


def _tiny_offload_runner(vllm_runner, model_dir, num_cpu_blocks: int, **kwargs):
    """A tiny random model with block size 16 and a small CPU tier."""
    runner_kwargs = dict(
        skip_tokenizer_init=True,
        dtype="float16",
        enforce_eager=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=1,
        block_size=16,
        num_gpu_blocks_override=64,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            # The null block takes one CPU block.
            kv_connector_extra_config={
                "cpu_bytes_to_use": 16 << 20,
                "num_cpu_blocks": num_cpu_blocks + 1,
            },
        ),
    )
    runner_kwargs.update(kwargs)
    return vllm_runner(str(model_dir), **runner_kwargs)


def test_cpu_tier_evicts_prefix_tail_first(vllm_runner, tmp_path):
    """A prompt prefilled in chunks keeps its leading CPU blocks when a later
    prompt evicts part of it, so a repeat still hits a prefix on CPU."""
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        LlamaForCausalLM(config).save_pretrained(tmp_path)

    with _tiny_offload_runner(
        vllm_runner,
        tmp_path,
        num_cpu_blocks=8,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4 * 16,  # prefill in 4-block chunks
    ) as runner:
        llm = runner.get_llm()
        # 8 blocks fill the CPU tier; 4 more blocks evict half of them.
        assert _generate_and_drop_gpu_cache(llm, seed=0, num_blocks=8) == 0
        assert _generate_and_drop_gpu_cache(llm, seed=1, num_blocks=4) == 0
        assert _generate_and_drop_gpu_cache(llm, seed=0, num_blocks=8) == 4 * 16


def test_hybrid_cpu_hit_is_requeued_after_it_finishes(vllm_runner, tmp_path):
    """On a full-attention + sliding-window model, a request that hits its CPU
    prefix requeues the blocks the hit needs when it finishes, after the
    blocks other requests stored while it ran."""
    config = Gemma3TextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=256,
        sliding_window=32,
        layer_types=["full_attention", "sliding_attention"],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        Gemma3ForCausalLM(config).save_pretrained(tmp_path)

    # An 8-block prompt keeps 8 full-attention blocks and the last 2-block
    # window on CPU; a 3-block prompt keeps 3 + 2.
    with _tiny_offload_runner(
        vllm_runner,
        tmp_path,
        num_cpu_blocks=15,
        dtype="bfloat16",
        attention_config={"backend": "TRITON_ATTN"},
        disable_hybrid_kv_cache_manager=False,
        max_num_seqs=2,
    ) as runner:
        llm = runner.get_llm()
        assert _generate_and_drop_gpu_cache(llm, seed=0, num_blocks=8) == 0
        # Prompt 0 hits CPU and decodes, short of a new full block, while
        # prompt 1 is stored.
        outputs = llm.generate(
            [_prompt(0, 8), _prompt(1, 3)],
            [_params(max_tokens=15), _params()],
            use_tqdm=False,
        )
        assert [o.num_cached_tokens for o in outputs] == [8 * 16, 0]
        _drop_gpu_cache(llm)
        # The CPU tier is full; prompt 2 evicts the blocks of prompt 1.
        assert _generate_and_drop_gpu_cache(llm, seed=2, num_blocks=3) == 0
        assert _generate_and_drop_gpu_cache(llm, seed=0, num_blocks=8) == 8 * 16


def _make_llm(model: str, lazy: bool, cpu_bytes_to_use: int, **llm_kwargs) -> LLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": cpu_bytes_to_use,
            "lazy_offload": lazy,
        },
    )
    llm_kwargs = (
        dict(
            kv_cache_memory_bytes=40 << 30,  # 40 GiB
            disable_hybrid_kv_cache_manager=False,
            enable_prefix_caching=True,
        )
        | llm_kwargs
    )
    return LLM(
        model=model,
        kv_transfer_config=kv_transfer_config,
        **llm_kwargs,
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


def _assert_worker_auto_numa(worker):
    import subprocess
    from pathlib import Path

    import psutil

    from vllm.utils.cpu_resource_utils import parse_id_list

    parallel_config = worker.parallel_config
    assert parallel_config.numa_bind and parallel_config._numa_bind_auto
    gpu_index = parallel_config.get_local_gpu_index(worker.local_rank)
    node = parallel_config.numa_bind_nodes[gpu_index]
    node_cpus = parse_id_list(
        Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    )
    cpus = set(psutil.Process().cpu_affinity())
    assert cpus and cpus <= set(node_cpus)
    result = subprocess.run(
        ["numactl", "--show"], check=True, capture_output=True, text=True, timeout=10
    )
    policy = dict(line.split(":", 1) for line in result.stdout.splitlines())
    assert policy["policy"].strip() == "preferred"
    assert policy["preferred node"].strip() == str(node)
    return {"node": node, "cpus": sorted(cpus), "policy": "preferred"}


@pytest.mark.slow_test
def test_simple_cpu_offload_with_auto_numa(monkeypatch, enable_pickle):
    """Soft-bound NVIDIA workers must store and restore KV, not silently recompute."""
    if not current_platform.is_cuda():
        pytest.skip("Automatic NUMA KV-offload smoke test requires NVIDIA")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    preflight = ParallelConfig(distributed_executor_backend="mp")
    preflight._numa_bind_auto = True
    if not numa_utils._can_auto_bind(preflight, local_rank=0) or not (
        numa_utils._probe_numactl_args(
            numa_utils._get_numactl_worker_args(preflight, local_rank=0)
        )
    ):
        pytest.skip("Host cannot engage automatic NUMA memory preference")

    model = "Qwen/Qwen3-0.6B"
    llm = _make_llm(
        model,
        False,
        256 << 20,
        kv_cache_memory_bytes=128 << 20,
        max_model_len=1024,
        distributed_executor_backend="mp",
        enforce_eager=True,
        disable_log_stats=False,
    )

    def loaded_blocks(llm: LLM):
        return sum(
            metric.value
            for metric in llm.get_metrics()
            if isinstance(metric, Counter)
            and metric.name == MetricName.LOAD_BLOCKS.removesuffix("_total")
            and metric.labels.get("model_name") == model
        )

    try:
        states = llm.collective_rpc(_assert_worker_auto_numa, timeout=30)
        print(f"NUMA worker state: {states}")
        sampling_params = SamplingParams(max_tokens=8, temperature=0)
        prompt = "hi " * 256 + "Let's count to ten. One, two, three, "
        cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        llm.generate(prompt, sampling_params, use_tqdm=False)
        deadline = time.monotonic() + 30
        while not llm.reset_prefix_cache(reset_connector=False):
            assert time.monotonic() < deadline, "CPU offload did not drain"
            llm.generate("0", sampling_params, use_tqdm=False)
        loaded_before = loaded_blocks(llm)
        restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
        loaded = loaded_blocks(llm) - loaded_before
        assert loaded > 0
        assert restored.outputs[0].token_ids == cold.outputs[0].token_ids
        print(f"KV offload blocks restored={loaded}; tokens match")
    finally:
        del llm


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
