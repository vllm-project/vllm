# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import random

import pytest
import torch
from utils import (
    BACKENDS,
    _extract_step_logprobs,
    _random_prompt,
    is_device_capability_below_90,
    resolve_model_name,
    skip_unsupported,
)

from vllm import LLM, SamplingParams

IS_DEVICE_CAPABILITY_BELOW_90 = is_device_capability_below_90()


@skip_unsupported
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("backend", BACKENDS)
def test_v1_generation_is_deterministic_across_batch_sizes_with_needle(
    backend,
):
    """
    Verify that a specific request ("needle") produces identical output
    when generated alone vs when generated as part of a larger batch.

    The needle is intentionally evaluated inside a mixed batch of
    similar-length prompts to ensure shared execution paths while
    avoiding extreme prompt-length divergence.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    attention_config = {"backend": backend}
    model = resolve_model_name(backend)

    num_trials = int(os.getenv("VLLM_NEEDLE_TRIALS", "5"))
    max_batch_size = int(os.getenv("VLLM_NEEDLE_BATCH_SIZE", "128"))

    gpu_mem_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.4"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "5120"))

    sampling = SamplingParams(
        temperature=float(os.getenv("VLLM_NEEDLE_TEMPERATURE", "0.0")),
        top_p=float(os.getenv("VLLM_NEEDLE_TOP_P", "0.95")),
        max_tokens=int(os.getenv("VLLM_NEEDLE_MAX_TOKENS", "128")),
        seed=20240919,
    )
   
    min_random_prompt = 1
    max_random_prompt = 8
    needle_prompt = _random_prompt(min_random_prompt, max_random_prompt)

    llm_bs1 = None
    llm_bsN = None

    try:
        # Baseline: needle generated alone
        llm_bs1 = LLM_with_max_seqs(
            model=model,
            max_num_seqs=1,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            attention_config=attention_config,
        )

        baseline_out = llm_bs1.generate([needle_prompt], sampling)
        assert len(baseline_out) == 1
        baseline_text = baseline_out[0].outputs[0].text

        # Batched engine
        llm_bsN = LLM_with_max_seqs(
            model=model,
            max_num_seqs=max_batch_size,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            attention_config=attention_config,
        )

        mismatches = 0
        needle_len = len(needle_prompt.split())

        for _ in range(num_trials):
            batch_size = random.randint(max_batch_size // 2, max_batch_size)
            needle_pos = random.randint(0, batch_size - 1)

            prompts: list[str] = []
            for i in range(batch_size):
                if i == needle_pos:
                    prompts.append(needle_prompt)
                else:
                    # Generate prompts of similar length to the needle
                    lo = max(1, needle_len - 16)
                    hi = needle_len + 16
                    prompts.append(_random_prompt(lo, hi))

            # Ensure the needle is evaluated inside a real batch
            assert needle_prompt in prompts
            assert len(prompts) >= 2

            outputs = llm_bsN.generate(prompts, sampling)

            for o in outputs:
                if o.prompt != needle_prompt:
                    continue

                text = o.outputs[0].text
                if text != baseline_text:
                    mismatches += 1

        if mismatches > 0:
            pytest.fail(
                f"Nondeterministic outputs detected: "
                f"{mismatches} mismatches across {num_trials} trials."
            )

    finally:
        if llm_bs1 is not None:
            with contextlib.suppress(Exception):
                llm_bs1.shutdown()
        if llm_bsN is not None:
            with contextlib.suppress(Exception):
                llm_bsN.shutdown()


@skip_unsupported
@pytest.mark.parametrize("backend", BACKENDS)
def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(backend):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)

    model_name = resolve_model_name(backend)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_num_seqs=32,
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
        attention_config={"backend": backend},
    )

    prompts = [_random_prompt(10, 50) for _ in range(32)]

    sp = SamplingParams(
        temperature=0.6,
        top_p=1.0,
        max_tokens=8,
        seed=1234,
        logprobs=5,
    )

    bs1_logprobs = []
    bs1_tokens = []

    for p in prompts:
        out = llm.generate([p], sp)[0]
        step_logprobs, token_ids = _extract_step_logprobs(out)
        bs1_logprobs.append(step_logprobs)
        bs1_tokens.append(token_ids)

    outs_batched = llm.generate(prompts, sp)

    for i, o in enumerate(outs_batched):
        step_logprobs, token_ids = _extract_step_logprobs(o)
        assert token_ids == bs1_tokens[i]
        for a, b in zip(step_logprobs, bs1_logprobs[i]):
            assert torch.equal(a, b)


def LLM_with_max_seqs(
    model: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    attention_config: dict | None = None,
) -> LLM:
    return LLM(
        model=model,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        tensor_parallel_size=int(os.getenv("VLLM_TP_SIZE", "1")),
        enable_prefix_caching=False,
        enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
        attention_config=attention_config,
    )
