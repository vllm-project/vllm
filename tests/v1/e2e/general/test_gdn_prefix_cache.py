# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for GDN (Qwen3.5) prefix caching in 'all' mode.

Validates that GDN's cumulative compressed state snapshots saved at block
boundaries produce identical generation output compared to computing
from scratch without cache.
"""
import os

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

MODEL = "Qwen/Qwen3.5-0.8B"
# 4 layers gives 3 GDN + 1 attention (full_attention_interval=4 pattern).
NUM_HIDDEN_LAYERS = 4
MAX_MODEL_LEN = 4096


def _create_engine(
    enable_prefix_caching: bool,
    mamba_cache_mode: str = "all",
) -> LLM:
    """Create an LLM engine with the specified prefix caching config."""
    return LLM(
        model=MODEL,
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=mamba_cache_mode,
        max_model_len=MAX_MODEL_LEN,
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        enforce_eager=True,
        seed=42,
        gpu_memory_utilization=0.3,
    )


def _generate(engine: LLM, prompts: list[str], max_tokens: int = 20) -> list[str]:
    """Generate outputs for given prompts."""
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = engine.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


@create_new_process_for_each_test()
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="GDN prefix caching requires CUDA GPU",
)
def test_gdn_prefix_cache_all_mode_correctness():
    """Verify that prefix cache hits produce deterministic correct outputs.

    Uses a single engine with prefix caching enabled:
    1. Request A ("prefix + suffix_A"): full prefill, fills prefix cache.
    2. Request B ("prefix + suffix_B"): partial cache hit on shared prefix.
    3. Repeat request B: full cache hit, output must match step 2.
    4. Repeat request A: full cache hit, output must match step 1.

    This proves that restoring from cached GDN state snapshots at block
    boundaries produces identical results to the original computation.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # In "all" mode, the block size for Qwen3.5-0.8B (4 layers) is 2048
    # tokens. Only FULL blocks can be prefix-cached, so the shared prefix
    # must exceed one complete block for any cache hit to occur.
    base_sentence = (
        "The theory of general relativity fundamentally changed "
        "our understanding of space time and gravity in physics. "
    )
    # ~15 tokens per repetition; 200 reps ≈ 3000 tokens > 2048
    prefix = base_sentence * 200

    suffix_a = "What is the speed of light in vacuum?"
    suffix_b = "When was general relativity published?"

    prompt_a = prefix + suffix_a
    prompt_b = prefix + suffix_b

    engine = _create_engine(enable_prefix_caching=True, mamba_cache_mode="all")

    # Step 1: First request — full prefill, populates prefix cache.
    output_a_first = _generate(engine, [prompt_a])[0]
    # Step 2: Different suffix — partial cache hit on shared prefix.
    output_b_first = _generate(engine, [prompt_b])[0]
    # Step 3: Repeat request B — full cache hit, must match step 2.
    output_b_second = _generate(engine, [prompt_b])[0]
    # Step 4: Repeat request A — full cache hit, must match step 1.
    output_a_second = _generate(engine, [prompt_a])[0]

    del engine
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    # Verify full cache hits produce identical outputs.
    assert output_a_first == output_a_second, (
        f"Full cache hit mismatch for prompt A.\n"
        f"  First:  {output_a_first!r}\n"
        f"  Second: {output_a_second!r}"
    )
    assert output_b_first == output_b_second, (
        f"Full cache hit mismatch for prompt B.\n"
        f"  First:  {output_b_first!r}\n"
        f"  Second: {output_b_second!r}"
    )


@create_new_process_for_each_test()
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="GDN prefix caching requires CUDA GPU",
)
def test_gdn_prefix_cache_all_mode_repeated_request():
    """Verify that re-running the exact same prompt produces identical output.

    When the same prompt is submitted twice, the second run should fully hit
    the prefix cache and produce the same output as the first run.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # In "all" mode, the block size for Qwen3.5-0.8B (4 layers) is 2048
    # tokens. Only FULL blocks can be prefix-cached, so the prefix must
    # exceed one complete block for any cache hit to occur.
    base_sentence = (
        "Machine learning is a subset of artificial intelligence that "
        "focuses on developing systems that learn from data. "
    )
    # ~15 tokens per repetition; 200 reps ≈ 3000 tokens > 2048
    prefix = base_sentence * 200
    prompt = prefix + "What is the key innovation of transformer architecture?"

    engine = _create_engine(enable_prefix_caching=True, mamba_cache_mode="all")

    # First generation: computes everything from scratch, caches state.
    output_first = _generate(engine, [prompt])[0]
    # Second generation: should hit cache fully, same output expected.
    output_second = _generate(engine, [prompt])[0]

    assert output_first == output_second, (
        f"Repeated request output mismatch.\n"
        f"  First:  {output_first!r}\n"
        f"  Second: {output_second!r}"
    )

    del engine
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()
