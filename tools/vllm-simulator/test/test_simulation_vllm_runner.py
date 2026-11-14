"""Test vLLM framework hijacking via vllm_simulator hooks.

1. Creates VLLMWorker + MultiInstanceBenchmarkRunner
2. Builds random dataset aligned to block_size
3. Validates completions, throughput, and created_time
4. Tests L1 (device) prefix cache hit (>95%)
5. Tests L2 (host/CPU) cache hit via native kv_offloading (>95%)
"""

import os
import random
from copy import deepcopy

import numpy as np
from transformers import AutoTokenizer

from vllm_simulator.simulation.benchmark import (
    BenchmarkConfig,
    MultiInstanceBenchmarkRunner,
    from_sample_requests,
)
from vllm_simulator.simulation.vllm.vllm_worker import VLLMWorker, EngineArgs

# After vllm_worker: its import installs the hooks vLLM must see first.
from vllm.benchmarks.datasets import RandomDataset  # noqa: E402

os.environ["VLLM_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def check_prefix_cache_and_offload(engine_args: dict, prefix_blocks: int = 4):
    """Check completions, L1 (device) prefix-cache hit, L2 (CPU offload) hit
    after eviction, and flush for one model.

    Prompts span ``prefix_blocks`` full blocks plus one token.
    """
    random.seed(0)
    np.random.seed(0)

    worker = VLLMWorker(engine_args=EngineArgs(**engine_args))
    runner = MultiInstanceBenchmarkRunner(workers=[worker])

    # Prefix-cache granularity.  The engine may enlarge the requested size
    # (backend preference, hybrid alignment) and reports the smallest KV
    # group's size, which compressed caches (DeepSeek-V4) shrink below it.
    effective_block_size = max(
        engine_args["block_size"] or 0,
        worker._llm.vllm_config.cache_config.block_size,
    )
    max_model_len = engine_args.get("max_model_len", 2048)
    # Prefix caching requires at least 2 full blocks AND prompts must fit in max_model_len
    min_required_input = effective_block_size * prefix_blocks + 1
    can_test_prefix_cache = (effective_block_size < max_model_len) and (min_required_input <= max_model_len)

    # Benchmark settings
    benchmark_config = BenchmarkConfig(request_rate=10, ignore_request_timestamp=True)

    # Build random requests aligned to effective block_size
    tokenizer = AutoTokenizer.from_pretrained(
        engine_args["model"], trust_remote_code=engine_args["trust_remote_code"]
    )

    if not can_test_prefix_cache:
        # Hybrid model with huge block_size: just test basic completions
        print(f"[SKIP] Prefix cache tests: effective block_size={effective_block_size} >= max_model_len={max_model_len}")
        requests = from_sample_requests(RandomDataset().sample(
            tokenizer=tokenizer, num_requests=4, input_len=24, output_len=1,
        ))
        metrics = runner.benchmark(benchmark_config, requests=requests)
        assert metrics["completed"] == len(requests)
        runner.shutdown()
        return

    # Ensure prompts span enough full blocks for high cache ratio
    min_input = effective_block_size * prefix_blocks + 1
    # Calculate how many prompts needed to evict all cached blocks
    blocks_per_prompt = min_input // effective_block_size
    num_gpu_blocks = engine_args.get("num_gpu_blocks_override", 100)
    num_cached = 8
    # Need enough eviction prompts to overflow the cache
    evict_prompts_needed = (num_gpu_blocks // blocks_per_prompt) + 1
    num_prompts = num_cached + 2 + evict_prompts_needed  # cached + gap + eviction
    requests = from_sample_requests(RandomDataset().sample(
        tokenizer=tokenizer,
        num_requests=num_prompts,
        input_len=min_input,
        output_len=1,
    ))

    # Split requests for cache tests
    cached_reqs = requests[:num_cached]
    evict_l1_reqs = requests[num_cached + 2:]

    # First run: warm up cache (populates both L1 GPU and L2 CPU)
    metrics = runner.benchmark(benchmark_config, requests=cached_reqs)
    assert metrics["completed"] == len(cached_reqs)

    request_stats = runner.get_request_stats()
    for idx, req in enumerate(request_stats):
        assert (
            idx == 0 or req["created_time"] != 0
        ), "created_time should not be zero when request_rate=10"

    assert metrics["prompt_tokens_cached_ratio"] == 0

    # Second run: hit device cache (L1)
    metrics = runner.benchmark(benchmark_config, requests=cached_reqs)
    local_hit = metrics["prompt_tokens_local_cache_hit_ratio"]
    assert local_hit > 0.95, (
        f"Expected local cache hit ratio > 0.95, got {local_hit}"
    )

    # Evict from device cache by filling GPU with other requests
    _ = runner.benchmark(benchmark_config, requests=evict_l1_reqs)

    # Third run: should hit host (L2 CPU) cache
    metrics = runner.benchmark(benchmark_config, requests=cached_reqs)
    ext_hit = metrics["prompt_tokens_external_kv_transfer_ratio"]
    assert ext_hit > 0.95, (
        f"Expected external KV transfer ratio > 0.95, got {ext_hit}"
    )

    # Fourth run: flush_cache clears both device and host caches
    runner.flush_cache()
    metrics = runner.benchmark(benchmark_config, requests=cached_reqs)
    assert metrics["prompt_tokens_cached_ratio"] == 0, (
        f"Expected no cache hit after flush, got {metrics['prompt_tokens_cached_ratio']}"
    )

    runner.shutdown()


def test_prefix_cache_and_offload():
    model_engine_args_list = [
        {"model": "Qwen/Qwen3-8B"},  # MHA
        {
            "model": "Qwen/Qwen3.5-27B",  # Hybrid (full_attention + linear_attention)
            # Blocks are aligned to 800 tokens so the attention page matches
            # the ~3.2MB/layer mamba state page; each KV block costs ~210MB.
            # Size the host pool so the eviction round (~204 blocks) cannot
            # evict the cached prompts (~32 blocks): 48 GiB / 210MB ≈ 245
            # blocks.
            "kv_offloading_size": 48.0,
            "max_model_len": 4096,
        },
        {"model": "zai-org/GLM-5", "block_size": None},  # Sparse MLA (DSA)
        {
            "model": "moonshotai/Kimi-K3",  # Hybrid (MLA + KDA), MXFP4 MoE
            # Hybrid alignment sets 5760-token blocks of ~637MB each:
            # 160 GiB / 637MB ≈ 269 blocks.
            "block_size": None,
            "max_model_len": 32768,
            "kv_offloading_size": 160.0,
        },
        {
            "model": "deepseek-ai/DeepSeek-V4-Pro",  # Compressed sparse MLA, FP8
            # The last sliding window (128 tokens) of a prompt is never a
            # prefix hit, so prompts span 12 blocks to keep the ratio > 0.95.
            # A 4096-token request needs ~1700 blocks; each block costs
            # ~11.5MB: 32 GiB / 11.5MB ≈ 2990 blocks.
            "block_size": 256,
            "prefix_blocks": 12,
            "max_model_len": 4096,
            "num_gpu_blocks_override": 2000,
            "kv_offloading_size": 32.0,
        },
    ]

    common_args = {
        "load_format": "dummy",
        "trust_remote_code": True,
        "block_size": 16,
        "gpu_memory_utilization": 0.9,
        "kv_offloading_size": 4.0,
        "kv_offloading_backend": "native",
        "num_gpu_blocks_override": 200,
        "max_model_len": 2048,
        "enable_prefix_caching": True,
    }

    for model_args in model_engine_args_list:
        engine_args = deepcopy(common_args)
        engine_args.update(model_args)
        prefix_blocks = engine_args.pop("prefix_blocks", 4)
        check_prefix_cache_and_offload(engine_args, prefix_blocks)


if __name__ == "__main__":
    test_prefix_cache_and_offload()
