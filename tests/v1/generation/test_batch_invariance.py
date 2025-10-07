# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import random

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def enable_batch_invariant_mode():
    """Automatically enable batch invariant kernel overrides for all tests."""
    old_value = os.environ.get("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT")
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"
    yield
    # Restore original value after test
    if old_value is None:
        os.environ.pop("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT", None)
    else:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = old_value


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Generate more realistic prompts that will actually produce varied tokens
    # Use a mix of common English text patterns

    prompt_templates = [
        # Question-answer style
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        # Story/narrative style
        "Once upon a time in a distant galaxy, there lived",
        "The old man walked slowly down the street, remembering",
        "In the year 2157, humanity finally discovered",
        # Technical/code style
        "To implement a binary search tree in Python, first we need to",
        "The algorithm works by iterating through the array and",
        "Here's how to optimize database queries using indexing:",
        # Factual/informative style
        "The Renaissance was a period in European history that",
        "Climate change is caused by several factors including",
        "The human brain contains approximately 86 billion neurons which",
        # Conversational style
        "I've been thinking about getting a new laptop because",
        "Yesterday I went to the store and bought",
        "My favorite thing about summer is definitely",
    ]

    # Pick a random template
    base_prompt = random.choice(prompt_templates)

    # Add some padding to vary the length if needed
    if min_words > 50:
        # For longer prompts, repeat context
        padding_text = (
            " This is an interesting topic that deserves more explanation. "
            * (min_words // 50)
        )
        base_prompt = base_prompt + padding_text

    return base_prompt


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Batch invariance tests only supported on Hopper (SM90)",
)
@pytest.mark.timeout(1000)
def test_v1_generation_is_deterministic_across_batch_sizes_with_needle():
    """
    Ensures that the same request (the 'needle' prompt) yields identical output
    whether run alone (bs=1) or mixed into a larger batch (e.g., bs=64),
    using the high-level v1 LLM() API only (no manual batching).

    Strategy:
    - Create two LLM engines with identical config except max_num_seqs: 1 vs N.
    - Compute a baseline output for the needle prompt with the bs=1 engine.
    - For many trials, generate a batch (size N) where the needle appears at a
      random position among random filler prompts using the bs=N engine.
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

    # Allow overrides from environment (useful for CI tuning)
    # "facebook/opt-125m" is too small, doesn't reliably test determinism
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")
    num_trials = int(os.getenv("VLLM_NEEDLE_TRIALS", "5"))
    max_batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "128"))
    min_random_prompt = int(os.getenv("VLLM_MIN_PROMPT", "1024"))
    max_random_prompt = int(os.getenv("VLLM_MAX_PROMPT", "2048"))
    assert max_batch_size >= 2, "Batch size should be >= 2 to mix needle."

    # Keep GPU memory usage low to avoid startup allocation failures.
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.4"))
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

    llm_bs1 = None
    llm_bsN = None
    try:
        # Engine with bs=1 behavior
        llm_bs1 = LLM_with_max_seqs(
            model=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
        )

        # Baseline generation for the needle prompt alone.
        baseline_out = llm_bs1.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        assert len(baseline_out[0].outputs) >= 1
        baseline_text = baseline_out[0].outputs[0].text

        # Engine with larger batch limit (e.g., 64)
        llm_bsN = LLM_with_max_seqs(
            model=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
        )

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

            # Generate with the larger-batch engine
            outputs = llm_bsN.generate(prompts, sampling)
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
        if llm_bs1 is not None:
            with contextlib.suppress(Exception):
                llm_bs1.shutdown()
        if llm_bsN is not None:
            with contextlib.suppress(Exception):
                llm_bsN.shutdown()


def _extract_step_logprobs(request_output):
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor(
                [
                    inner.logprobs[i][tid].logprob
                    for i, tid in enumerate(inner.token_ids)
                ],
                dtype=torch.float32,
            )
            return t, inner.token_ids

    return None, None


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Batch invariance tests only supported on Hopper (SM90)",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to match production inference path.",
)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
@pytest.mark.forked
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(backend):
    backend = os.getenv("VLLM_ATTENTION_BACKEND", backend)
    os.environ["VLLM_ATTENTION_BACKEND"] = backend

    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    model_name = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    # For batch invariance, disable custom all-reduce to ensure deterministic
    # all-reduce operations (custom all-reduce may not be deterministic)
    from vllm.model_executor.layers.batch_invariant import (
        vllm_kernel_override_batch_invariant,
    )

    disable_custom_ar = vllm_kernel_override_batch_invariant()

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
        max_num_seqs=32,
        max_model_len=8192,
        dtype="bfloat16",  # not everything is supported
    )

    # Use more realistic prompts for better token generation
    prompts = [_random_prompt(10, 50) for i in range(32)]

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


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Batch invariance tests only supported on Hopper (SM90)",
)
def test_simple_generation():
    """
    Simple test that runs the model with a basic prompt and prints the output.
    Useful for quick smoke testing and debugging.
    """
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")

    llm = LLM(
        model=model,
        max_num_seqs=1,
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        dtype="bfloat16",
        enable_prefix_caching=False,
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


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Batch invariance tests only supported on Hopper (SM90)",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to match production inference path.",
)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER"])
@pytest.mark.forked
def test_logprobs_WITHOUT_batch_invariance_should_FAIL(backend):
    """
    This test is the inverse of test_logprobs_bitwise_batch_invariance_bs1_vs_bsN.
    It DISABLES batch invariance mode and expects to see non-deterministic behavior
    between BS=1 and BS=N runs. This demonstrates that batch invariance is actually
    doing something useful.

    The test will PASS if we detect differences (proving batch invariance matters).
    The test will FAIL if everything matches (suggesting batch invariance isn't needed).
    """
    backend = os.getenv("VLLM_ATTENTION_BACKEND", backend)
    os.environ["VLLM_ATTENTION_BACKEND"] = backend

    # CRITICAL: Disable batch invariance for this test
    old_value = os.environ.get("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT")
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "0"

    try:
        seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
        random.seed(seed)
        model_name = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")
        tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

        print(f"\n{'=' * 80}")
        print("BATCH INVARIANCE DISABLED: Expecting non-deterministic behavior")
        print(f"{'=' * 80}\n")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            enable_prefix_caching=False,
            max_num_seqs=32,
            max_model_len=8192,
            dtype="bfloat16",
        )

        # Use more realistic prompts for better token generation
        prompts = [_random_prompt(10, 50) for i in range(32)]

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
            print(
                f"\n[BS=1] Running prompt {idx}/{len(prompts)} - Preview: {p[:80]}..."
            )
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

    finally:
        # Restore original value
        if old_value is None:
            os.environ.pop("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT", None)
        else:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = old_value


def LLM_with_max_seqs(
    model: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
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
        dtype="bfloat16",
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=True,
        # Enable for MOE models
        # enable_expert_parallel=True,
    )
