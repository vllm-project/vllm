# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
import random
import string

import pytest
import torch

from vllm import LLM, SamplingParams


def _random_prompt(min_words: int = 1024, max_words: int = 1024 * 2) -> str:
    # Lightweight random prompt generator to vary prompt lengths and content.
    vocab = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]
    n = random.randint(min_words, max_words)
    words = random.choices(vocab, k=n)

    # Add some noise and punctuation variability
    if random.random() < 0.5:
        words[0] = words[0].capitalize()
    if random.random() < 0.2:
        words.append("".join(random.choices(string.ascii_lowercase, k=5)))
    punct = random.choice([".", "?", "!", "...", ""])
    return " ".join(words) + punct


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
    random.seed(12345)

    # Allow overrides from environment (useful for CI tuning)
    # "facebook/opt-125m" is too small, doesn't reliably test determinism
    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")
    num_trials = int(os.getenv("VLLM_NEEDLE_TRIALS", "5"))
    batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "64"))
    assert batch_size >= 2, "Batch size should be >= 2 to mix needle."

    # Keep GPU memory usage low to avoid startup allocation failures.
    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    swap_space_gb = int(os.getenv("VLLM_SWAP_SPACE_GB", "4"))

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
            max_num_seqs=1,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            swap_space=swap_space_gb,
        )

        # Baseline generation for the needle prompt alone.
        baseline_out = llm_bs1.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        assert len(baseline_out[0].outputs) >= 1
        baseline_text = baseline_out[0].outputs[0].text

        # Engine with larger batch limit (e.g., 64)
        llm_bsN = LLM_with_max_seqs(
            model=model,
            max_num_seqs=batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            swap_space=swap_space_gb,
        )

        mismatches = 0

        for trial in range(num_trials):
            # Create a batch of size `batch_size` and insert the needle at
            # a random index
            prompts: list[str] = []
            needle_pos = random.randint(0, batch_size - 1)
            for i in range(batch_size):
                if i == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    prompts.append(_random_prompt())

            # Generate with the larger-batch engine
            outputs = llm_bsN.generate(prompts, sampling)
            # Find the needle output by position
            needle_output = outputs[needle_pos]
            assert needle_output.prompt == needle_prompt
            assert len(needle_output.outputs) >= 1
            text = needle_output.outputs[0].text

            if text != baseline_text:
                mismatches += 1

        passes = num_trials - mismatches
        # Dump how many passed vs failed
        print(
            f"[determinism] total={num_trials}, passed={passes}, "
            f"failed={mismatches}, batch_size={batch_size}"
        )

        if mismatches > 0:
            pytest.fail(
                f"Nondeterministic outputs detected: {mismatches} failed out "
                f"of {num_trials} trials (batch_size={batch_size})."
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
            return t

    return None


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA to match production inference path.",
)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bs2():
    # model_name = os.getenv("VLLM_TEST_MODEL", "facebook/opt-125m")
    model_name = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B")
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    # Force float32 to avoid precision-induced differences.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        enforce_eager=True,  # helps reduce nondeterminism from some backends
    )

    prompts = [
        "The capital of France is",
        "The capital of Germany is",
    ]

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        # Seed shouldn't matter at temperature=0, but keeping it stable anyway.
        seed=1234,
        logprobs=5,
    )

    # BS=1: run prompts individually and collect logprobs per step.
    bs1_logprobs_per_prompt = []
    for p in prompts:
        outs = llm.generate([p], sp, use_tqdm=False)
        assert len(outs) == 1
        step_logprobs = _extract_step_logprobs(outs[0])
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bs1_logprobs_per_prompt.append(step_logprobs)

    # BS=2: run prompts in a batch and collect logprobs per step for each
    # prompt.
    outs_batched = llm.generate(prompts, sp, use_tqdm=False)
    assert len(outs_batched) == len(prompts)
    bs2_logprobs_per_prompt = []
    for o in outs_batched:
        step_logprobs = _extract_step_logprobs(o)
        if step_logprobs is None:
            pytest.skip(
                "Logits are not available on RequestOutput; "
                "enable logprobs return to run this test."
            )
        bs2_logprobs_per_prompt.append(step_logprobs)

    # Compare step-by-step logprobs for each prompt between BS=1 and BS=2 runs.
    for i, (logprobs_bs1, logprobs_bs2) in enumerate(
        zip(bs1_logprobs_per_prompt, bs2_logprobs_per_prompt)
    ):
        assert len(logprobs_bs1) == len(logprobs_bs2), (
            f"Different number of generation steps for prompt index {i}: "
            f"{len(logprobs_bs1)} (BS=1) vs {len(logprobs_bs2)} (BS=2)"
        )
        for t, (a, b) in enumerate(zip(logprobs_bs1, logprobs_bs2)):
            assert a.shape == b.shape, (
                f"Logits shape mismatch at prompt {i}, step {t}: {a.shape} vs {b.shape}"
            )
            # Bitwise exact equality.
            assert torch.equal(a, b), (
                f"Bitwise logprobs mismatch at prompt {i}, step {t} "
                f"(dtype={a.dtype}, shape={a.shape})."
            )


def LLM_with_max_seqs(
    model: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    swap_space: int,
) -> LLM:
    """
    Helper to construct an LLM with a specific max_num_seqs (batch-size limit)
    using the high-level v1 LLM API, while constraining memory usage.
    """
    return LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        # Constrain GPU memory pool so test can run even on busy GPUs.
        gpu_memory_utilization=gpu_memory_utilization,
        # Keep KV cache footprint small while allowing longer outputs.
        max_model_len=max_model_len,
        # Allow some CPU offload if needed.
        swap_space=swap_space,
        # Keep things lean and CI-friendly.
        dtype="auto",
        # Single-GPU by default; override externally if desired.
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        trust_remote_code=os.getenv("VLLM_TRUST_REMOTE_CODE", "0") == "1",
        enable_prefix_caching=False,
        # Enable for MOE models
        # enable_expert_parallel=True,
    )
