# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import random

import pytest
import torch
from utils import (
    BACKENDS,
    TEST_MODEL,
    _extract_step_logprobs,
    _random_prompt,
    is_device_capability_below_90,
    skip_unsupported,
)

import vllm.envs as envs
from vllm import LLM, SamplingParams


@skip_unsupported
@pytest.mark.flaky(reruns=3)
@pytest.mark.timeout(1000)
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
def test_v1_generation_is_deterministic_across_batch_sizes_with_needle(
    backend,
):
    """
    Ensures that the same request (the 'needle' prompt) yields identical output
    whether run alone (bs=1) or mixed into a larger batch (e.g., bs=64),
    using the high-level v1 LLM() API only (no manual batching).

    Strategy:
    - Create a single LLM engine configured for the larger batch limit (N).
    - Compute a baseline output for the needle prompt when it is run alone.
    - For many trials, generate a mixed batch (size N) where the needle appears
      at a random position among random filler prompts using the same engine.
    - Track how many trials match vs mismatch, and report totals at the end.
      The test fails if any mismatches occur, but we still dump pass/fail
      counts.

    Notes:
    - Use seeded stochastic sampling with a fixed seed to test determinism.
    - Outputs are intentionally longer and sampled at higher temperature/top_p
      to produce a more random-sounding phrase, yet remain deterministic by
      seed.
    - Keep max_tokens and max_model_len bounded for speed and memory use.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    attention_config = {"backend": backend}
    # Allow overrides from environment (useful for CI tuning)
    # "facebook/opt-125m" is too small, doesn't reliably test determinism
    model = TEST_MODEL
    num_trials = int(os.getenv("VLLM_NEEDLE_TRIALS", "5"))
    max_batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "128"))
    min_random_prompt = int(os.getenv("VLLM_MIN_PROMPT", "1024"))
    max_random_prompt = int(os.getenv("VLLM_MAX_PROMPT", "2048"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to mix needle."

    # Keep GPU memory usage low to avoid startup allocation failures.
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "5120"))

    # Sampling parameters: longer outputs with a more random-sounding
    # continuation,but still deterministic due to fixed seed.
    temperature = float(os.getenv("VLLM_NEEDLE_TEMPERATURE", "0.0"))
    top_p = float(os.getenv("VLLM_NEEDLE_TOP_P", "0.95"))
    max_tokens = int(os.getenv("VLLM_NEEDLE_MAX_TOKENS", "128"))

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=20240919,
    )

    needle_prompt = "There once was a "

    llm = None
    try:
        llm = LLM_with_max_seqs(
            model=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            attention_config=attention_config,
        )

        # Baseline generation for the needle prompt alone.
        baseline_out = llm.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        assert len(baseline_out[0].outputs) >= 1
        baseline_text = baseline_out[0].outputs[0].text

        mismatches = 0

        for trial in range(num_trials):
            # Create a batch of size `max_batch_size` and insert the needle at
            # a random index
            prompts: list[str] = []
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)
            for i in range(batch_size):
                if i == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(_random_prompt(min_random_prompt, max_random_prompt))

            # Generate with the same engine but in a larger batch.
            outputs = llm.generate(prompts, sampling)
            # Find the needle output by position
            needle_output = outputs[needle_pos]
            assert needle_output.prompt == needle_prompt
            assert len(needle_output.outputs) >= 1
            text = needle_output.outputs[0].text

            if text != baseline_text:
                print(f"{text}\n\n== Not the same as ==\n\n{baseline_text}\n\n")
                mismatches += 1

        passes = num_trials - mismatches
        # Dump how many passed vs failed
        print(
            f"[determinism] total={num_trials}, passed={passes}, "
            f"failed={mismatches}, max_batch_size={max_batch_size}"
        )

        if mismatches > 0:
            pytest.fail(
                f"Nondeterministic outputs detected: {mismatches} failed out "
                f"of {num_trials} trials (max_batch_size={max_batch_size})."
            )

    finally:
        # Ensure engines are shutdown to free GPU/VRAM across test sessions
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
@pytest.mark.parametrize(
    "block_m,block_n",
    [(16, 16), (8, 16)],
)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    backend,
    block_m,
    block_n,
):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    # For batch invariance, disable custom all-reduce to ensure deterministic
    # all-reduce operations (custom all-reduce may not be deterministic)
    import vllm.envs as envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=tp_size,
        max_num_seqs=128,
        max_model_len=8192,
        dtype="auto",  # not everything is supported
        gpu_memory_utilization=0.9,
        attention_config={
            "backend": backend,
            "flex_attn_block_m": block_m,
            "flex_attn_block_n": block_n,
        },
    )

    # Use more realistic prompts for better token generation
    prompts = [_random_prompt(10, 50) for _ in range(32)]

    # TODO: Update prompts to have ragged lengths in order to test chunked prefill
    #       The above tests are not currently long enough to exercise chunking.
    # prompts = (
    #     [_random_prompt(10, 50) for _ in range(28)]
    #     + [_random_prompt(256, 512) for _ in range(50)]
    #     + [_random_prompt(2048, 4096) for _ in range(50)]
    # )

    sp = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=16,
        seed=1234,
        logprobs=5,
    )

    # BS=1: run prompts individually and collect logprobs per step.
    print("\n" + "=" * 80)
    print("STARTING BS=1 RUNS (each prompt individually)")
    print("=" * 80 + "\n")

    bs1_logprobs_per_prompt = []
    bs1_tokens_per_prompt = []
    for idx, p in enumerate(prompts):
        print(f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}...")
        outs = llm.generate([p], sp, use_tqdm=False)
        assert len(outs) == 1
        step_logprobs, token_ids = _extract_step_logprobs(outs[0])
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bs1_logprobs_per_prompt.append(step_logprobs)
        bs1_tokens_per_prompt.append(token_ids)
        print(f"[BS=1] Prompt {idx} generated tokens: {token_ids}")

    # BS=N: run prompts in a batch and collect logprobs per step for each
    # prompt.
    print("\n" + "=" * 80)
    print(f"STARTING BS={len(prompts)} RUN (all prompts batched)")
    print("=" * 80 + "\n")

    outs_batched = llm.generate(prompts, sp, use_tqdm=False)
    assert len(outs_batched) == len(prompts)
    bsN_logprobs_per_prompt = []
    bsN_tokens_per_prompt = []

    print(f"\n[BS={len(prompts)}] Processing batched outputs...")
    for idx, o in enumerate(outs_batched):
        tokens = o.outputs[0].token_ids if o.outputs else "N/A"
        print(f"[BS={len(prompts)}] Prompt {idx} generated tokens: {tokens}")
        step_logprobs, token_ids = _extract_step_logprobs(o)
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bsN_logprobs_per_prompt.append(step_logprobs)
        bsN_tokens_per_prompt.append(token_ids)

    # Compare step-by-step logprobs for each prompt between BS=1 and BS=N runs.
    failed_prompts = []
    for i, (logprobs_bs1, logprobs_bsN, tokens_bs1, tokens_bsN) in enumerate(
        zip(
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
        )
    ):
        if len(logprobs_bs1) != len(logprobs_bsN):
            reason = (
                f"Different number of steps: {len(logprobs_bs1)} (BS=1) "
                f"vs {len(logprobs_bsN)} (BS=N)"
            )
            failed_prompts.append(
                {
                    "prompt_idx": i,
                    "step": "all",
                    "reason": reason,
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                }
            )
            continue

        # Check if tokens match first
        if tokens_bs1 != tokens_bsN:
            failed_prompts.append(
                {
                    "prompt_idx": i,
                    "step": "sampling",
                    "reason": "Different tokens sampled",
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                    "bs1_all_logprobs": [
                        logprobs_bs1[s].tolist() for s in range(len(logprobs_bs1))
                    ],
                    "bsN_all_logprobs": [
                        logprobs_bsN[s].tolist() for s in range(len(logprobs_bsN))
                    ],
                }
            )
            continue

        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a.shape != b.shape:
                failed_prompts.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    }
                )
                break

            if not torch.equal(a, b):
                max_diff = torch.abs(a - b).max().item()
                # Print which token failed
                print(f"\n[DIVERGENCE] Prompt {i}, Token {t}: max_diff={max_diff:.6e}")
                bs1_tok = tokens_bs1[t] if t < len(tokens_bs1) else "N/A"
                bsN_tok = tokens_bsN[t] if t < len(tokens_bsN) else "N/A"
                print(f"  Token IDs: bs1={bs1_tok}, bsN={bsN_tok}")
                print(f"  BS=1 logprob: {a.tolist()}")
                print(f"  BS=N logprob: {b.tolist()}")
                failed_prompts.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Bitwise mismatch (max_diff={max_diff:.6e})",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                        "bs1_all_logprobs": [
                            logprobs_bs1[s].tolist() for s in range(len(logprobs_bs1))
                        ],
                        "bsN_all_logprobs": [
                            logprobs_bsN[s].tolist() for s in range(len(logprobs_bsN))
                        ],
                    }
                )
                break

    # Print summary of all failures
    if failed_prompts:
        print(f"\n{'=' * 80}")
        fail_msg = (
            f"BATCH INVARIANCE FAILURES: {len(failed_prompts)}/"
            f"{len(prompts)} prompts failed"
        )
        print(fail_msg)
        print(f"{'=' * 80}")
        for fail in failed_prompts:
            print(f"\nPrompt {fail['prompt_idx']} (step {fail['step']}):")
            print(f"  Reason: {fail['reason']}")
            print(f"  Preview: {fail['prompt_preview']}...")

            # Always show the tokens
            if "bs1_tokens" in fail:
                print(f"  BS=1 tokens: {fail['bs1_tokens']}")
            if "bsN_tokens" in fail:
                print(f"  BS=N tokens: {fail['bsN_tokens']}")

            if "bs1_all_logprobs" in fail:
                print(f"  BS=1 logprobs for all {len(fail['bs1_all_logprobs'])} steps:")
                for step_idx, logprobs in enumerate(fail["bs1_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
                print(f"  BS=N logprobs for all {len(fail['bsN_all_logprobs'])} steps:")
                for step_idx, logprobs in enumerate(fail["bsN_all_logprobs"]):
                    print(f"    Step {step_idx}: {logprobs}")
        print(f"{'=' * 80}\n")

        # Fail the test with summary
        msg = (
            f"Batch invariance violated in {len(failed_prompts)}/"
            f"{len(prompts)} prompts. See output above for details."
        )
        pytest.fail(msg)


@skip_unsupported
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
def test_simple_generation(backend):
    """
    Simple test that runs the model with a basic prompt and prints the output.
    Useful for quick smoke testing and debugging.
    """
    model = TEST_MODEL

    llm = LLM(
        model=model,
        max_num_seqs=1,
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="auto",
        enable_prefix_caching=False,
        attention_config={"backend": backend},
    )

    prompt = "the capital of france is"
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
    )

    print(f"\n{'=' * 80}")
    print("Running simple generation test")
    print(f"Prompt: '{prompt}'")
    print(f"{'=' * 80}\n")

    try:
        outputs = llm.generate([prompt], sampling_params)

        assert len(outputs) == 1
        output_text = outputs[0].outputs[0].text

        print(f"Output: '{output_text}'")
        print(f"\n{'=' * 80}")
        print(f"Full completion: '{prompt}{output_text}'")
        print(f"{'=' * 80}\n")

    finally:
        with contextlib.suppress(Exception):
            llm.shutdown()


@skip_unsupported
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
def test_logprobs_without_batch_invariance_should_fail(
    backend, monkeypatch: pytest.MonkeyPatch
):
    """
    This test is the inverse of test_logprobs_bitwise_batch_invariance_bs1_vs_bsN.
    It DISABLES batch invariance mode and expects to see non-deterministic behavior
    between BS=1 and BS=N runs. This demonstrates that batch invariance is actually
    doing something useful.

    The test will PASS if we detect differences (proving batch invariance matters).
    The test will FAIL if everything matches (suggesting batch invariance isn't needed).
    """
    # CRITICAL: Disable batch invariance for this test
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    print(f"\n{'=' * 80}")
    print("BATCH INVARIANCE DISABLED: Expecting non-deterministic behavior")
    print(f"{'=' * 80}\n")

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=tp_size,
        max_num_seqs=32,
        max_model_len=8192,
        dtype="auto",
        attention_config={"backend": backend},
    )

    # build ragged prompts to change shapes significantly across BS=1 vs BS=N
    long_min = int(os.getenv("VLLM_MIN_PROMPT", "768"))
    long_max = int(os.getenv("VLLM_MAX_PROMPT", "2048"))
    prompts: list[str] = []
    options = [
        (max(long_min, 1536), max(long_max, 3072)),  # very long
        (max(1024, long_min), max(2048, long_max)),  # long
        (256, 512),  # mid
        (10, 20),  # short
    ]

    for _ in range(32):
        lo, hi = random.choice(options)
        prompts.append(_random_prompt(lo, hi))

    sp = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=8,
        seed=1234,
        logprobs=5,
    )

    # BS=1: run prompts individually and collect logprobs per step.
    print("\n" + "=" * 80)
    print("STARTING BS=1 RUNS (each prompt individually)")
    print("=" * 80 + "\n")

    bs1_logprobs_per_prompt = []
    bs1_tokens_per_prompt = []
    for idx, p in enumerate(prompts):
        print(f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}...")
        outs = llm.generate([p], sp, use_tqdm=False)
        assert len(outs) == 1
        step_logprobs, token_ids = _extract_step_logprobs(outs[0])
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bs1_logprobs_per_prompt.append(step_logprobs)
        bs1_tokens_per_prompt.append(token_ids)
        print(f"[BS=1] Prompt {idx} generated tokens: {token_ids}")

    # BS=N: run prompts in a batch and collect logprobs per step for each prompt.
    print("\n" + "=" * 80)
    print(f"STARTING BS={len(prompts)} RUN (all prompts batched)")
    print("=" * 80 + "\n")

    outs_batched = llm.generate(prompts, sp, use_tqdm=False)
    assert len(outs_batched) == len(prompts)
    bsN_logprobs_per_prompt = []
    bsN_tokens_per_prompt = []

    print(f"\n[BS={len(prompts)}] Processing batched outputs...")
    for idx, o in enumerate(outs_batched):
        tokens = o.outputs[0].token_ids if o.outputs else "N/A"
        print(f"[BS={len(prompts)}] Prompt {idx} generated tokens: {tokens}")
        step_logprobs, token_ids = _extract_step_logprobs(o)
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bsN_logprobs_per_prompt.append(step_logprobs)
        bsN_tokens_per_prompt.append(token_ids)

    # Compare step-by-step logprobs for each prompt between BS=1 and BS=N runs.
    differences_found = []
    for i, (logprobs_bs1, logprobs_bsN, tokens_bs1, tokens_bsN) in enumerate(
        zip(
            bs1_logprobs_per_prompt,
            bsN_logprobs_per_prompt,
            bs1_tokens_per_prompt,
            bsN_tokens_per_prompt,
        )
    ):
        if len(logprobs_bs1) != len(logprobs_bsN):
            reason = (
                f"Different number of steps: {len(logprobs_bs1)} (BS=1) "
                f"vs {len(logprobs_bsN)} (BS=N)"
            )
            differences_found.append(
                {
                    "prompt_idx": i,
                    "step": "all",
                    "reason": reason,
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                }
            )
            continue

        # Check if tokens match first
        if tokens_bs1 != tokens_bsN:
            differences_found.append(
                {
                    "prompt_idx": i,
                    "step": "sampling",
                    "reason": "Different tokens sampled",
                    "prompt_preview": prompts[i][:100],
                    "bs1_tokens": tokens_bs1,
                    "bsN_tokens": tokens_bsN,
                }
            )
            continue

        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bsN)):
            if a.shape != b.shape:
                differences_found.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Shape mismatch: {a.shape} vs {b.shape}",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    }
                )
                break

            if not torch.equal(a, b):
                max_diff = torch.abs(a - b).max().item()
                print(
                    f"\n[EXPECTED DIVERGENCE FOUND] Prompt {i}, "
                    f"Token {t}: max_diff={max_diff:.6e}"
                )
                bs1_tok = tokens_bs1[t] if t < len(tokens_bs1) else "N/A"
                bsN_tok = tokens_bsN[t] if t < len(tokens_bsN) else "N/A"
                print(f"  Token IDs: bs1={bs1_tok}, bsN={bsN_tok}")
                print(f"  BS=1 logprob: {a.tolist()}")
                print(f"  BS=N logprob: {b.tolist()}")
                differences_found.append(
                    {
                        "prompt_idx": i,
                        "step": t,
                        "reason": f"Bitwise mismatch (max_diff={max_diff:.6e})",
                        "prompt_preview": prompts[i][:100],
                        "bs1_tokens": tokens_bs1,
                        "bsN_tokens": tokens_bsN,
                    }
                )
                break

    # Print summary
    print(f"\n{'=' * 80}")
    if differences_found:
        success_msg = (
            f"✓ SUCCESS: Batch invariance is doing something! "
            f"Found {len(differences_found)}/{len(prompts)} prompts "
            f"with differences when batch invariance was DISABLED."
        )
        print(success_msg)
        print(f"{'=' * 80}")
        for diff in differences_found:
            print(f"\nPrompt {diff['prompt_idx']} (step {diff['step']}):")
            print(f"  Reason: {diff['reason']}")
            print(f"  Preview: {diff['prompt_preview']}...")
            if "bs1_tokens" in diff:
                print(f"  BS=1 tokens: {diff['bs1_tokens']}")
            if "bsN_tokens" in diff:
                print(f"  BS=N tokens: {diff['bsN_tokens']}")
        print(f"{'=' * 80}\n")
        # Test PASSES because we found differences (batch invariance matters!)
        return
    else:
        # Test FAILS because everything matched even without batch invariance
        fail_msg = (
            f"✗ UNEXPECTED: All {len(prompts)} prompts matched "
            f"between BS=1 and BS=N even with batch invariance DISABLED. "
            f"This suggests batch invariance might not be necessary, "
            f"or the test needs more sensitive prompts."
        )
        print(fail_msg)
        print(f"{'=' * 80}\n")
        pytest.fail(fail_msg)


@skip_unsupported
@pytest.mark.parametrize("backend", ["FLASH_ATTN"])
def test_decode_logprobs_match_prefill_logprobs(
    backend,
):
    """
    Test that verifies decode logprobs match prefill logprobs.

    For each decoded token at position i:
    1. Run decode to generate N tokens and collect their logprobs
    2. For each position i in [0, N):
       - Take prefix = prompt + tokens[0:i]
       - Run prefill(prefix + tokens[i]) to get logprob of tokens[i]
       - Verify prefill logprob matches decode logprob bitwise

    This ensures that the logprobs from decode are consistent with what
    we would get if we ran prefill on each prefix.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    import vllm.envs as envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=tp_size,
        max_num_seqs=32,
        max_model_len=8192,
        dtype="auto",
        attention_config={"backend": backend},
    )

    # Use a few test prompts
    num_test_prompts = int(os.getenv("VLLM_DECODE_PREFILL_NUM_PROMPTS", "4"))
    prompts = [_random_prompt(10, 50) for _ in range(num_test_prompts)]

    # Generate longer sequences to test multiple decode steps
    max_tokens = int(os.getenv("VLLM_DECODE_PREFILL_MAX_TOKENS", "16"))

    sp = SamplingParams(
        temperature=0.0,  # Greedy for determinism
        max_tokens=max_tokens,
        logprobs=5,
    )

    print("\n" + "=" * 80)
    print("STEP 1: Running decode to generate tokens and collect logprobs")
    print("=" * 80 + "\n")

    # Step 1: Run decode and collect logprobs
    decode_outputs = llm.generate(prompts, sp, use_tqdm=False)

    failed_comparisons = []

    for prompt_idx, (prompt, decode_output) in enumerate(zip(prompts, decode_outputs)):
        print(f"\n[Prompt {prompt_idx}] Testing: {prompt[:80]}...")

        # Extract decode logprobs and tokens
        decode_logprobs, token_ids = _extract_step_logprobs(decode_output)
        if decode_logprobs is None:
            pytest.skip(
                "Logprobs are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )

        print(f"[Prompt {prompt_idx}] Generated {len(token_ids)} tokens: {token_ids}")
        print(f"[Prompt {prompt_idx}] Decode logprobs: {decode_logprobs.tolist()}")

        # Step 2: For each token position, run prefill and compare
        print(f"\n[Prompt {prompt_idx}] Verifying each token via prefill...")

        for token_idx in range(len(token_ids)):
            # Construct the prefix up to (but not including) this token
            current_token = token_ids[token_idx]

            # We need to detokenize to get the text prefix
            # For this, we'll use the tokenizer from the LLM
            # However, the LLM API doesn't expose tokenizer easily, so we'll
            # construct the prefix by decoding from the original prompt

            # Get text up to this point by using the output text
            # This is approximate but should work for verification
            if token_idx == 0:
                prefix_prompt = prompt
            else:
                # Use the partial output text up to this token
                # We'll need to construct this from the full output
                prefix_output = decode_output.outputs[0]
                # Get the text for tokens 0 to token_idx-1
                # Unfortunately, we don't have per-token text, so we'll use
                # a different approach: run prefill with prompt + tokens[0:token_idx]

                # Actually, we need to get the actual text. Let's use a workaround:
                # Run a generation with max_tokens = token_idx to get that prefix
                prefix_sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=token_idx,
                    logprobs=1,
                )
                prefix_output = llm.generate([prompt], prefix_sp, use_tqdm=False)[0]
                prefix_prompt = prompt + prefix_output.outputs[0].text

            # Now run prefill with max_tokens=1 to get the logprob of the next token
            prefill_sp = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=5,
            )

            print(
                f"  [Token {token_idx}] Running prefill for prefix "
                f"(len={len(prefix_prompt)})..."
            )
            prefill_output = llm.generate([prefix_prompt], prefill_sp, use_tqdm=False)[
                0
            ]
            prefill_logprobs, prefill_token_ids = _extract_step_logprobs(prefill_output)

            if prefill_logprobs is None:
                print(f"  [Token {token_idx}] Warning: No prefill logprobs available")
                continue

            # The first token from prefill should match the current token
            prefill_token = prefill_token_ids[0]
            prefill_logprob = prefill_logprobs[0].item()
            decode_logprob = decode_logprobs[token_idx].item()

            print(
                f"  [Token {token_idx}] Decode token: {current_token}, "
                f"logprob: {decode_logprob:.8f}"
            )
            print(
                f"  [Token {token_idx}] Prefill token: {prefill_token}, "
                f"logprob: {prefill_logprob:.8f}"
            )

            # Check if tokens match
            if current_token != prefill_token:
                failed_comparisons.append(
                    {
                        "prompt_idx": prompt_idx,
                        "token_idx": token_idx,
                        "reason": "Token mismatch",
                        "decode_token": current_token,
                        "prefill_token": prefill_token,
                        "decode_logprob": decode_logprob,
                        "prefill_logprob": prefill_logprob,
                        "prompt_text": prompt[:100],
                        "prefix_text": prefix_prompt[:100],
                    }
                )
                print(f"  [Token {token_idx}] ✗ TOKEN MISMATCH!")
                continue

            # Check if logprobs match bitwise
            if decode_logprob != prefill_logprob:
                diff = abs(decode_logprob - prefill_logprob)
                failed_comparisons.append(
                    {
                        "prompt_idx": prompt_idx,
                        "token_idx": token_idx,
                        "reason": "Logprob mismatch",
                        "decode_token": current_token,
                        "prefill_token": prefill_token,
                        "decode_logprob": decode_logprob,
                        "prefill_logprob": prefill_logprob,
                        "diff": diff,
                        "prompt_text": prompt[:100],
                        "prefix_text": prefix_prompt[:100],
                        "decode_all_tokens": token_ids,
                        "decode_all_logprobs": decode_logprobs.tolist(),
                    }
                )
                print(f"  [Token {token_idx}] ✗ LOGPROB MISMATCH! diff={diff:.8e}")
            else:
                print(f"  [Token {token_idx}] ✓ Match (bitwise equal)")

    # Print summary
    print(f"\n{'=' * 80}")
    if failed_comparisons:
        print(f"DECODE-PREFILL MISMATCH: {len(failed_comparisons)} failures detected")
        print(f"{'=' * 80}")

        # Group failures by prompt for better readability
        failures_by_prompt: dict[int, list[dict]] = {}
        for fail in failed_comparisons:
            pid = fail["prompt_idx"]
            if pid not in failures_by_prompt:
                failures_by_prompt[pid] = []
            failures_by_prompt[pid].append(fail)

        for prompt_idx, failures in failures_by_prompt.items():
            print(f"\n{'=' * 80}")
            print(f"PROMPT {prompt_idx}: {failures[0]['prompt_text']}...")
            print(f"{'=' * 80}")
            print(f"Total failures for this prompt: {len(failures)}")

            # Show where mismatches occur (which token positions)
            mismatch_positions = [f["token_idx"] for f in failures]
            print(f"Mismatch at token positions: {mismatch_positions}")

            # Show first few failures in detail
            for i, fail in enumerate(failures[:5]):  # Show first 5 failures per prompt
                print(f"\n  [Failure {i + 1}] Token position {fail['token_idx']}:")
                print(f"    Reason: {fail['reason']}")
                print(f"    Prefix text: '{fail['prefix_text']}...'")
                print(
                    f"    Decode:  token={fail['decode_token']}, "
                    f"logprob={fail['decode_logprob']:.10f}"
                )
                print(
                    f"    Prefill: token={fail['prefill_token']}, "
                    f"logprob={fail['prefill_logprob']:.10f}"
                )
                if "diff" in fail:
                    print(f"    Difference: {fail['diff']:.10e}")
                    # Show in hex to see bitwise difference
                    import struct

                    decode_hex = struct.pack("f", fail["decode_logprob"]).hex()
                    prefill_hex = struct.pack("f", fail["prefill_logprob"]).hex()
                    print(f"    Decode logprob (hex):  0x{decode_hex}")
                    print(f"    Prefill logprob (hex): 0x{prefill_hex}")

                # If we have all tokens/logprobs, show the context
                if "decode_all_tokens" in fail and "decode_all_logprobs" in fail:
                    token_idx = fail["token_idx"]
                    all_tokens = fail["decode_all_tokens"]
                    all_logprobs = fail["decode_all_logprobs"]

                    # Show context: 2 tokens before and after
                    start = max(0, token_idx - 2)
                    end = min(len(all_tokens), token_idx + 3)

                    print(f"    Context (tokens {start} to {end - 1}):")
                    for j in range(start, end):
                        marker = " <-- MISMATCH" if j == token_idx else ""
                        print(
                            f"      [{j}] token={all_tokens[j]}, "
                            f"logprob={all_logprobs[j]:.8f}{marker}"
                        )

            if len(failures) > 5:
                print(f"\n  ... and {len(failures) - 5} more failures for this prompt")

        print(f"\n{'=' * 80}\n")

        pytest.fail(
            f"Decode logprobs do not match prefill logprobs: "
            f"{len(failed_comparisons)} mismatches found."
        )
    else:
        print("✓ SUCCESS: All decode logprobs match prefill logprobs bitwise!")
        print(f"{'=' * 80}\n")


def LLM_with_max_seqs(
    model: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    attention_config: dict | None = None,
) -> LLM:
    """
    Helper to construct an LLM with a specific max_num_seqs (batch-size limit)
    using the high-level v1 LLM API, while constraining memory usage.
    """
    return LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="auto",
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        enable_prefix_caching=False,
        attention_config=attention_config,
        # Enable for MOE models
        # enable_expert_parallel=True,
    )


def _make_ragged_batch(
    needle_prompt: str,
    max_batch_size: int,
    needle_pos: int,
    filler_sizes: list[tuple[int, int]] | None = None,
    unique_fillers: bool = False,
) -> list[str]:
    if filler_sizes is None:
        filler_sizes = [(16, 64), (256, 512), (1024, 2048), (2048, 4096)]
    prompts: list[str] = []
    for i in range(max_batch_size):
        if i == needle_pos:
            prompts.append(needle_prompt)
        else:
            lo, hi = random.choice(filler_sizes)
            filler = _random_prompt(lo, hi)
            if unique_fillers:
                filler = f"Unique filler request {i}: {filler}"
            prompts.append(filler)
    return prompts


def _assert_logprobs_match(
    baseline_logprobs,
    cached_logprobs,
    baseline_tokens,
    cached_tokens,
    label: str,
) -> None:
    assert cached_tokens == baseline_tokens, (
        f"{label}: token IDs differ after prefix-cache hit.\n"
        f"baseline={baseline_tokens}\ncached={cached_tokens}"
    )
    assert len(cached_logprobs) == len(baseline_logprobs), (
        f"{label}: step count differs.\n"
        f"baseline={len(baseline_logprobs)}, cached={len(cached_logprobs)}"
    )
    for step, (a, b) in enumerate(zip(baseline_logprobs, cached_logprobs)):
        if not torch.equal(a, b):
            max_diff = torch.abs(a - b).max().item()
            pytest.fail(
                f"{label}: batch invariance violated at decode step {step}: "
                f"max_diff={max_diff:.6e}\n"
                f"baseline_tokens={baseline_tokens}\n"
                f"cached_tokens={cached_tokens}\n"
                f"baseline_logprob={a.tolist()}\n"
                f"cached_logprob={b.tolist()}\n"
            )


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", BACKENDS)
def test_partial_prefix_cache_hit_preserves_batch_invariance(monkeypatch, backend):
    """
    Verifies that a partial prefix-cache hit (only the shared prefix is cached,
    the suffix still needs a fresh prefill) produces bit-identical logprobs
    compared to a full-prefill baseline run alone.

    Cache state going into Case B:
      - tokens 0..prefix_len-1 : cached (from the warmup run)
      - tokens prefix_len..end  : NOT cached → fresh suffix prefill

    This exercises the path where prefix caching changes the effective prefill
    length and block-table layout, while the user-visible prompt is identical.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    max_batch_size = int(os.getenv("VLLM_PREFIX_CACHE_BATCH_SIZE", "32"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    fixed_chunk_size = int(os.getenv("VLLM_PREFIX_CACHE_FIXED_CHUNK_SIZE", "0"))
    fixed_chunk_size = int(os.getenv("VLLM_PREFIX_CACHE_FIXED_CHUNK_SIZE", "0"))
    prefix_len = int(os.getenv("VLLM_PREFIX_CACHE_PREFIX_LEN", "2048"))
    suffix_len = int(os.getenv("VLLM_PREFIX_CACHE_SUFFIX_LEN", "512"))

    common_prefix = _random_prompt(prefix_len, prefix_len + 64)
    suffix = _random_prompt(suffix_len, suffix_len + 64)
    needle_prompt = common_prefix + "\n\nQuestion:\n" + suffix

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16,
                        seed=1234, logprobs=5)
    warm_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                             seed=1234)

    llm_kwargs = {}
    if fixed_chunk_size > 0:
        llm_kwargs["long_prefill_token_threshold"] = fixed_chunk_size

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=tp_size,
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=gpu_mem_util,
            enable_prefix_caching=True,
            enforce_eager=is_device_capability_below_90(),
            attention_config={"backend": backend},
            **llm_kwargs,
        )

        # Case A: full prefill with empty cache → baseline logprobs.
        baseline_out = llm.generate([needle_prompt], sp, use_tqdm=False)
        baseline_logprobs, baseline_tokens = _extract_step_logprobs(baseline_out[0])
        if baseline_logprobs is None:
            pytest.skip("Logprobs not available.")

        # Clear the cache so Case A's full needle is not still resident.
        # Without this, the warmup below would be a no-op (Case A already
        # cached the full needle), and Case B would get a *full* cache hit
        # (decode path) instead of the intended *partial* hit (suffix prefill).
        llm.reset_prefix_cache()
        
        tokenizer = llm.get_tokenizer()
        common_ids = tokenizer.encode(common_prefix)
        needle_ids = tokenizer.encode(needle_prompt)
        
        assert needle_ids[:len(common_ids)] == common_ids, (
            "common_prefix is not an exact token prefix of needle_prompt\n"
            f"common_len={len(common_ids)}\n"
            f"needle_prefix_tail={needle_ids[:len(common_ids)][-20:]}\n"
            f"common_tail={common_ids[-20:]}\n"
        )
        
        print(
            "TOKEN_PREFIX_CHECK",
            {
                "common_len": len(common_ids),
                "needle_len": len(needle_ids),
                "prefix_match": (
                    needle_ids[:len(common_ids)] == common_ids
                ),
            },
        )

        # Warm only the shared prefix so Case B gets a partial hit.
        llm.generate([common_prefix], warm_sp, use_tqdm=False)

        # Case B: same full prompt in a ragged batch → partial prefix cache hit.
        needle_pos = random.randint(0, max_batch_size - 1)
        # Small fillers keep total KV footprint well below the cache capacity so
        # the needle's cached prefix blocks are never evicted before it runs.
        prompts = _make_ragged_batch(
            needle_prompt, max_batch_size, needle_pos,
            filler_sizes=[(16, 64), (64, 256), (128, 512)],
            unique_fillers=True,
        )

        cached_outs = llm.generate(prompts, sp, use_tqdm=False)
        cached_needle = cached_outs[needle_pos]
        assert cached_needle.prompt == needle_prompt

        cached_logprobs, cached_tokens = _extract_step_logprobs(cached_needle)
        if cached_logprobs is None:
            pytest.skip("Logprobs not available.")

        _assert_logprobs_match(
            baseline_logprobs, cached_logprobs,
            baseline_tokens, cached_tokens,
            label="partial-prefix-cache-hit",
        )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", BACKENDS)
def test_full_cache_hit_preserves_batch_invariance(monkeypatch, backend):
    """
    Verifies that a full cache hit (the entire prompt is already cached from a
    prior run) produces bit-identical logprobs when the same prompt is re-run
    inside a larger ragged batch.

    Cache state going into Case B:
      - ALL prompt tokens cached (Case A populated them)
      → FA2 decode path for the last prompt token (no prefill needed)

    This specifically exercises the FA2 packed-GQA heuristic that fires when
    max_seqlen_q == 1 for a GQA model, which is suppressed in batch-invariant
    mode by clamping max_seqlen_q to max_model_len.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    max_batch_size = int(os.getenv("VLLM_PREFIX_CACHE_BATCH_SIZE", "32"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    prefix_len = int(os.getenv("VLLM_PREFIX_CACHE_PREFIX_LEN", "2048"))
    suffix_len = int(os.getenv("VLLM_PREFIX_CACHE_SUFFIX_LEN", "512"))

    common_prefix = _random_prompt(prefix_len, prefix_len + 64)
    suffix = _random_prompt(suffix_len, suffix_len + 64)
    needle_prompt = common_prefix + "\n\nQuestion:\n" + suffix

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16,
                        seed=1234, logprobs=5)

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=tp_size,
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=gpu_mem_util,
            enable_prefix_caching=True,
            enforce_eager=is_device_capability_below_90(),
            attention_config={"backend": backend},
        )

        # Case A: full prefill with empty cache → baseline logprobs.
        # This also populates the KV cache for ALL needle tokens.
        baseline_out = llm.generate([needle_prompt], sp, use_tqdm=False)
        baseline_logprobs, baseline_tokens = _extract_step_logprobs(baseline_out[0])
        if baseline_logprobs is None:
            pytest.skip("Logprobs not available.")

        # Case B: same full prompt in a ragged batch.
        # All needle tokens are already cached → full cache hit → decode path.
        needle_pos = random.randint(0, max_batch_size - 1)
        # Small fillers to avoid KV-cache pressure evicting the needle's blocks.
        prompts = _make_ragged_batch(
            needle_prompt, max_batch_size, needle_pos,
            filler_sizes=[(16, 64), (64, 256), (128, 512)],
            unique_fillers=True,
        )

        cached_outs = llm.generate(prompts, sp, use_tqdm=False)
        cached_needle = cached_outs[needle_pos]
        assert cached_needle.prompt == needle_prompt

        cached_logprobs, cached_tokens = _extract_step_logprobs(cached_needle)
        if cached_logprobs is None:
            pytest.skip("Logprobs not available.")

        _assert_logprobs_match(
            baseline_logprobs, cached_logprobs,
            baseline_tokens, cached_tokens,
            label="full-cache-hit",
        )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", BACKENDS)
def test_prefix_cache_hit_length_invariance_fixed_batch(monkeypatch, backend):
    """
    Verifies that different prefix-cache hit depths produce bit-identical
    logprobs when the needle prompt runs in a fixed-size ragged batch.

    Reference: BS=1, cold cache (needle alone).
    Two hit cases run the same fixed batch of max_batch_size sequences:
      short-hit : tokens 0..short_split-1 pre-warmed
      long-hit  : tokens 0..long_split-1 pre-warmed

    Both hit cases are compared against the BS=1 cold-cache reference.
    Using a BS=32 cold run as the reference would compare prefix KV computed
    in a BS=32 batch (no-cache) against prefix KV from a BS=1 warmup
    (hit cases) — these differ numerically even with VLLM_BATCH_INVARIANT.

    Warmup inputs are submitted as prompt_token_ids (not strings) so they
    are guaranteed to be an exact token-prefix of the needle prompt,
    avoiding BPE boundary differences.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    seed = int(os.getenv("VLLM_TEST_SEED", "54"))
    random.seed(seed)

    max_batch_size = int(os.getenv("VLLM_PREFIX_CACHE_BATCH_SIZE", "32"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    fixed_chunk_size = int(os.getenv("VLLM_PREFIX_CACHE_FIXED_CHUNK_SIZE", "0"))

    short_prefix = _random_prompt(1024, 1088)
    mid = _random_prompt(1024, 1088)
    suffix = _random_prompt(512, 576)
    long_prefix = short_prefix + mid
    needle_prompt = long_prefix + "\n\nQuestion:\n" + suffix

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                        seed=1234, logprobs=5)
    warm_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                             seed=1234)

    llm_kwargs = {}
    if fixed_chunk_size > 0:
        llm_kwargs["long_prefill_token_threshold"] = fixed_chunk_size

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=int(os.getenv("VLLM_TEST_TP_SIZE", "1")),
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
            ),
            enable_prefix_caching=True,
            enforce_eager=is_device_capability_below_90(),
            attention_config={"backend": backend},
            **llm_kwargs,
        )

        tokenizer = llm.get_tokenizer()
        needle_ids = tokenizer.encode(needle_prompt)
        # Use sub-string token lengths as approximate split points.
        # Warmup uses prompt_token_ids so BPE boundary differences between
        # tokenize(sub_string) and needle_ids[:N] are irrelevant.
        short_split = len(tokenizer.encode(short_prefix))
        long_split = len(tokenizer.encode(long_prefix))

        assert 0 < short_split < long_split < len(needle_ids), (
            "Split points must be strictly increasing and within needle length"
        )

        print("PREFIX_SPLITS", {
            "needle_tokens": len(needle_ids),
            "short_split": short_split,
            "long_split": long_split,
            "fixed_chunk_size": fixed_chunk_size,
        })

        needle_pos = random.randint(0, max_batch_size - 1)
        prompts = _make_ragged_batch(
            needle_prompt, max_batch_size, needle_pos,
            filler_sizes=[(16, 64), (64, 256), (128, 512)],
        )

        def run_case(warmup_ids, label):
            llm.reset_prefix_cache()
            if warmup_ids is not None:
                llm.generate(
                    [{"prompt_token_ids": warmup_ids}],
                    warm_sp, use_tqdm=False,
                )
            outs = llm.generate(prompts, sp, use_tqdm=False)
            lp, tok = _extract_step_logprobs(outs[needle_pos])
            if lp is None:
                pytest.skip("Logprobs not available.")
            print("FIXED_BATCH_CASE", {
                "label": label,
                "warm_tokens": len(warmup_ids) if warmup_ids is not None else 0,
                "needle_tokens": len(needle_ids),
                "tokens": tok,
            })
            return lp, tok

        # Diagnostic: compare BS=1 cold vs BS=32 cold to isolate batch-size
        # invariance from prefix-cache invariance.
        llm.reset_prefix_cache()
        _diag_bs1 = llm.generate([needle_prompt], sp, use_tqdm=False)
        _diag_lp1, _diag_tok1 = _extract_step_logprobs(_diag_bs1[0])
        print("FIXED_BATCH_DIAG bs1_cold", {
            "tokens": _diag_tok1, "lp": _diag_lp1.tolist() if _diag_lp1 is not None else None
        })

        ref_lp, ref_tok = run_case(None, "no-cache")

        if _diag_lp1 is not None and ref_lp is not None:
            import torch as _torch
            _bs1_vs_bs32 = _torch.abs(_diag_lp1 - ref_lp).max().item()
            print(f"FIXED_BATCH_DIAG bs1_vs_bs32_nocache: max_diff={_bs1_vs_bs32:.6e}")

        short_lp, short_tok = run_case(needle_ids[:short_split], "short-hit")
        long_lp, long_tok = run_case(needle_ids[:long_split], "long-hit")

        if _diag_lp1 is not None:
            print("FIXED_BATCH_DIFF_MATRIX", {
                "bs1_vs_bsN_nocache": torch.abs(_diag_lp1 - ref_lp).max().item(),
                "bs1_vs_short_hit": torch.abs(_diag_lp1 - short_lp).max().item(),
                "bs1_vs_long_hit": torch.abs(_diag_lp1 - long_lp).max().item(),
                "bsN_nocache_vs_short_hit": torch.abs(
                    ref_lp - short_lp
                ).max().item(),
                "bsN_nocache_vs_long_hit": torch.abs(
                    ref_lp - long_lp
                ).max().item(),
                "short_hit_vs_long_hit": torch.abs(
                    short_lp - long_lp
                ).max().item(),
            })

        _assert_logprobs_match(
            ref_lp, short_lp, ref_tok, short_tok, "short-hit-vs-no-cache"
        )
        _assert_logprobs_match(
            ref_lp, long_lp, ref_tok, long_tok, "long-hit-vs-no-cache"
        )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", BACKENDS)
def test_prefix_cache_hit_length_invariance_variable_batch(monkeypatch, backend):
    """
    Verifies that a partial prefix-cache hit in a large ragged batch produces
    bit-identical logprobs to a full prefill of the same prompt run alone
    (BS=1, cold cache).

    This simultaneously exercises two invariant properties:
      1. Batch-size invariance: BS=1 (no cache) vs BS=N (cached)
      2. Prefix-cache-hit-length invariance: short-hit vs long-hit

    Warmup inputs are submitted as prompt_token_ids (not strings) so they
    are guaranteed to be an exact token-prefix of the needle prompt,
    avoiding BPE boundary differences.
    """
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    seed = int(os.getenv("VLLM_TEST_SEED", "54"))
    random.seed(seed)

    max_batch_size = int(os.getenv("VLLM_PREFIX_CACHE_BATCH_SIZE", "32"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))

    short_prefix = _random_prompt(1024, 1088)
    mid = _random_prompt(1024, 1088)
    suffix = _random_prompt(512, 576)
    long_prefix = short_prefix + mid
    needle_prompt = long_prefix + "\n\nQuestion:\n" + suffix

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                        seed=1234, logprobs=5)
    warm_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1,
                             seed=1234)

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=int(os.getenv("VLLM_TEST_TP_SIZE", "1")),
            max_num_seqs=max_batch_size,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
            ),
            enable_prefix_caching=True,
            enforce_eager=is_device_capability_below_90(),
            attention_config={"backend": backend},
        )

        tokenizer = llm.get_tokenizer()
        needle_ids = tokenizer.encode(needle_prompt)
        short_split = len(tokenizer.encode(short_prefix))
        long_split = len(tokenizer.encode(long_prefix))

        assert 0 < short_split < long_split < len(needle_ids), (
            "Split points must be strictly increasing and within needle length"
        )

        print("PREFIX_SPLITS", {
            "needle_tokens": len(needle_ids),
            "short_split": short_split,
            "long_split": long_split,
        })

        needle_pos = random.randint(0, max_batch_size - 1)
        batch_prompts = _make_ragged_batch(
            needle_prompt, max_batch_size, needle_pos,
            filler_sizes=[(16, 64), (64, 256), (128, 512)],
        )

        # BS=1, cold cache: establish baseline.
        llm.reset_prefix_cache()
        ref_outs = llm.generate([needle_prompt], sp, use_tqdm=False)
        ref_lp, ref_tok = _extract_step_logprobs(ref_outs[0])
        if ref_lp is None:
            pytest.skip("Logprobs not available.")

        def run_bsN_case(warmup_ids, label):
            llm.reset_prefix_cache()
            llm.generate(
                [{"prompt_token_ids": warmup_ids}],
                warm_sp, use_tqdm=False,
            )
            outs = llm.generate(batch_prompts, sp, use_tqdm=False)
            lp, tok = _extract_step_logprobs(outs[needle_pos])
            if lp is None:
                pytest.skip("Logprobs not available.")
            print("VARIABLE_BATCH_CASE", {
                "label": label,
                "batch_size": len(batch_prompts),
                "needle_pos": needle_pos,
                "warm_tokens": len(warmup_ids),
                "needle_tokens": len(needle_ids),
                "tokens": tok,
            })
            return lp, tok

        short_lp, short_tok = run_bsN_case(
            needle_ids[:short_split], "short-hit-batchN"
        )
        long_lp, long_tok = run_bsN_case(
            needle_ids[:long_split], "long-hit-batchN"
        )

        _assert_logprobs_match(
            ref_lp, short_lp, ref_tok, short_tok,
            "short-hit-batchN-vs-bs1-no-cache",
        )
        _assert_logprobs_match(
            ref_lp, long_lp, ref_tok, long_tok,
            "long-hit-batchN-vs-bs1-no-cache",
        )

    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()