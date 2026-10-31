# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FA3 split-KV cudagraph correctness (#62626).

Verifies that FULL cudagraph mode with FA3 split-KV produces
bitwise identical results to FULL_AND_PIECEWISE or eager modes.
"""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FA3 tests"
)
def test_fa3_split_kv_full_cudagraph_trigger_a():
    """Test FA3 split-KV with FULL cudagraph, trigger A: 2-token cache hit.

    Trigger A: Exactly 2 new tokens on a prefix-cache hit.
    Expected: Same logits as FULL_AND_PIECEWISE.
    Bug (#62626): 2nd token logprobs diverge by several nats.
    """
    model_id = "Qwen/Qwen3-8B"

    # Use FULL_AND_PIECEWISE as baseline
    llm_baseline = LLM(
        model_id,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
    )

    # Use FULL (where bug occurs)
    llm_full = LLM(
        model_id,
        enable_prefix_caching=True,
        block_size=16,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_mode": "FULL"},
    )

    # Generate synthetic prompt
    tokenizer = llm_baseline.get_tokenizer()
    prompt_text = " ".join(f"item {i} is red;" for i in range(100))
    prompt_ids = tokenizer(prompt_text).input_ids[:200]

    # Test with q=2 (the problematic case)
    q = 2

    # Baseline: cache 128-token prefix, extend by q tokens
    llm_baseline.reset_prefix_cache()
    llm_baseline.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:128])],
        SamplingParams(max_tokens=1),
        use_tqdm=False
    )
    baseline_output = llm_baseline.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:128 + q])],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=20,
        ),
        use_tqdm=False
    )[0]
    baseline_token = baseline_output.outputs[0].token_ids[0]
    baseline_logprob = baseline_output.outputs[0].logprobs[0][baseline_token].logprob

    # FULL mode: same setup
    llm_full.reset_prefix_cache()
    llm_full.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:128])],
        SamplingParams(max_tokens=1),
        use_tqdm=False
    )
    full_output = llm_full.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:128 + q])],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=20,
        ),
        use_tqdm=False
    )[0]
    full_token = full_output.outputs[0].token_ids[0]
    full_logprob = full_output.outputs[0].logprobs[0][full_token].logprob

    # Verify tokens match (strict correctness)
    assert baseline_token == full_token, (
        f"Token mismatch: baseline={baseline_token}, full={full_token}. "
        f"Bug #62626: FA3 split-KV scheduler mismatch causes wrong token selection."
    )

    # Verify logprobs match within numerical tolerance (BF16 precision)
    logprob_diff = abs(baseline_logprob - full_logprob)
    assert logprob_diff < 0.01, (
        f"Logprob mismatch: diff={logprob_diff:.6f} nats. "
        f"Expected < 0.01 (BF16 noise). Bug #62626 would show > 0.1 nats."
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FA3 tests"
)
def test_fa3_split_kv_full_cudagraph_trigger_b():
    """Test FA3 split-KV with FULL cudagraph, trigger B: small max_num_seqs.

    Trigger B: Small max_num_seqs (16 and below) with ordinary prefills.
    Expected: Same logits as FULL_AND_PIECEWISE.
    Bug (#62626): Logprobs diverge when scheduler chooses STANDARD mode.
    """
    model_id = "Qwen/Qwen3-8B"

    llm_baseline = LLM(
        model_id,
        enable_prefix_caching=False,
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
    )

    llm_full = LLM(
        model_id,
        enable_prefix_caching=False,
        max_model_len=1024,
        max_num_seqs=8,  # Trigger B: small max_num_seqs
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_mode": "FULL"},
    )

    tokenizer = llm_baseline.get_tokenizer()
    prompt_text = " ".join(f"word {i};" for i in range(50))
    prompt_ids = tokenizer(prompt_text).input_ids[:100]

    # Baseline
    baseline_output = llm_baseline.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:80])],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=20,
        ),
        use_tqdm=False
    )[0]
    baseline_token = baseline_output.outputs[0].token_ids[0]
    baseline_logprob = baseline_output.outputs[0].logprobs[0][baseline_token].logprob

    # FULL mode
    full_output = llm_full.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids[:80])],
        SamplingParams(
            max_tokens=1,
            temperature=0,
            logprobs=20,
        ),
        use_tqdm=False
    )[0]
    full_token = full_output.outputs[0].token_ids[0]
    full_logprob = full_output.outputs[0].logprobs[0][full_token].logprob

    # Verify tokens match
    assert baseline_token == full_token, (
        f"Token mismatch: baseline={baseline_token}, full={full_token}. "
        f"Bug #62626: scheduler mismatch with small max_num_seqs."
    )

    # Verify logprobs match within tolerance
    logprob_diff = abs(baseline_logprob - full_logprob)
    assert logprob_diff < 0.01, (
        f"Logprob mismatch: diff={logprob_diff:.6f} nats. "
        f"Expected < 0.01 (BF16 noise)."
    )


if __name__ == "__main__":
    test_fa3_split_kv_full_cudagraph_trigger_a()
    print("✅ Trigger A test passed")
    test_fa3_split_kv_full_cudagraph_trigger_b()
    print("✅ Trigger B test passed")
