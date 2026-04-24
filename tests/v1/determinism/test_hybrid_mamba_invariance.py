# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid-Mamba chunked-prefill and prefix-cache batch invariance tests.

Chunked prefill divergence is a known hazard for Mamba and hybrid
(attention+SSM) models. Mamba-2's chunked scan algorithm computes the
recurrent state over fixed-size token chunks; when a prefill is split by the
scheduler, each split must land on a Mamba chunk boundary. An off-boundary
split produces a different intermediate scan state, which yields divergent
logprobs compared to processing the same sequence in a single chunk.

Prefix caching for Mamba and hybrid models is also experimental. In "align"
mode, cache reuse is only correct if every prefill split still lands on a valid
Mamba chunk boundary. In "all" mode, a cache hit must reproduce exactly the
same serialized Mamba state that was materialized during the cache-filling run.
"""

import contextlib
import os
import random
import time

import pytest
import torch
from utils import (
    _extract_step_logprobs,
    _random_prompt,
    is_device_capability_below_90,
    skip_unsupported,
)

from vllm import LLM, SamplingParams, TokensPrompt

IS_DEVICE_CAPABILITY_BELOW_90 = is_device_capability_below_90()

TEST_MODEL = os.getenv(
    "VLLM_HYBRID_MAMBA_TEST_MODEL",
    "tiiuae/Falcon-H1-0.5B-Base",
)

CHUNKED_PREFILL_MAX_BATCH_SIZE = int(os.getenv("VLLM_DIVERGENCE_MAX_BATCH_SIZE", "10"))
CHUNKED_PREFILL_EXTRA_BATCH_SIZES = os.getenv(
    "VLLM_DIVERGENCE_EXTRA_BATCH_SIZES", "15,16,17"
)
CHUNKED_PREFILL_MAX_NUM_BATCHED_TOKENS = 2048

PREFIX_CACHE_MAX_BATCH_SIZE = int(os.getenv("VLLM_DIVERGENCE_MAX_BATCH_SIZE", "8"))
PREFIX_CACHE_EXTRA_BATCH_SIZES = os.getenv("VLLM_DIVERGENCE_EXTRA_BATCH_SIZES", "")
PREFIX_CACHE_MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("VLLM_TEST_MAX_NUM_BATCHED_TOKENS", "2048")
)
PREFIX_CACHE_RESET_TIMEOUT_S = float(
    os.getenv("VLLM_PREFIX_CACHE_RESET_TIMEOUT_S", "5.0")
)
PREFIX_CACHE_SHARED_PREFIX_TOKEN_COUNT = int(
    os.getenv("VLLM_PREFIX_CACHE_SHARED_PREFIX_TOKENS", "4096")
)
PREFIX_CACHE_UNIQUE_SUFFIX_TOKEN_COUNT = int(
    os.getenv("VLLM_PREFIX_CACHE_UNIQUE_SUFFIX_TOKENS", "128")
)

MAMBA_CACHE_MODES = ["align", "all"]
BACKENDS = ["TRITON_ATTN"]

UNIQUE_SUFFIX_INSTRUCTIONS = [
    "Summarize the shared context in one concise sentence.",
    "State the most concrete fact mentioned in the shared context.",
    "Describe the tone of the shared context in a short phrase.",
    "Name one topic from the shared context and explain why it matters.",
    "Turn the shared context into a short follow-up question.",
    "Identify one action implied by the shared context.",
    "Rewrite the main idea of the shared context plainly.",
    "Give a terse title for the shared context.",
]


def _parse_positive_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    if not raw:
        return values
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        values.append(value)
    return values


def _build_batch_sizes(max_batch_size: int, extra_batch_sizes: str) -> list[int]:
    batch_sizes = list(range(1, max_batch_size + 1))
    batch_sizes.extend(_parse_positive_int_csv(extra_batch_sizes))
    return sorted(set(batch_sizes))


CHUNKED_PREFILL_BATCH_SIZES = _build_batch_sizes(
    CHUNKED_PREFILL_MAX_BATCH_SIZE,
    CHUNKED_PREFILL_EXTRA_BATCH_SIZES,
)
PREFIX_CACHE_BATCH_SIZES = _build_batch_sizes(
    PREFIX_CACHE_MAX_BATCH_SIZE,
    PREFIX_CACHE_EXTRA_BATCH_SIZES,
)


def _reset_prefix_cache(llm: LLM) -> None:
    deadline = time.monotonic() + PREFIX_CACHE_RESET_TIMEOUT_S
    while not llm.reset_prefix_cache():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "reset_prefix_cache did not succeed within "
                f"{PREFIX_CACHE_RESET_TIMEOUT_S}s."
            )
        time.sleep(0.1)


def _build_fixed_length_token_ids(
    tokenizer,
    target_tokens: int,
    *,
    prefix_text: str = "",
    min_words: int,
    max_words: int,
) -> list[int]:
    text = prefix_text
    token_ids = tokenizer.encode(text) if text else []
    while len(token_ids) < target_tokens:
        fragment = _random_prompt(min_words=min_words, max_words=max_words)
        text = f"{text}\n{fragment}".strip()
        token_ids = tokenizer.encode(text)
    return token_ids[:target_tokens]


def _build_prefix_cache_prompt_variants(
    llm: LLM,
    num_prompts: int,
) -> tuple[list[TokensPrompt], int]:
    tokenizer = llm.get_tokenizer()
    shared_prefix_token_ids = _build_fixed_length_token_ids(
        tokenizer,
        PREFIX_CACHE_SHARED_PREFIX_TOKEN_COUNT,
        min_words=512,
        max_words=1024,
    )

    prompts: list[TokensPrompt] = []
    for idx in range(num_prompts):
        instruction = UNIQUE_SUFFIX_INSTRUCTIONS[idx % len(UNIQUE_SUFFIX_INSTRUCTIONS)]
        suffix_token_ids = _build_fixed_length_token_ids(
            tokenizer,
            PREFIX_CACHE_UNIQUE_SUFFIX_TOKEN_COUNT,
            prefix_text=(
                f"Prompt variant {idx + 1}: {instruction} "
                "Answer in one sentence with at most eighteen words, and do "
                "not mention the variant number."
            ),
            min_words=64,
            max_words=128,
        )
        prompts.append(
            TokensPrompt(
                prompt_token_ids=shared_prefix_token_ids + suffix_token_ids,
            )
        )
    return prompts, len(shared_prefix_token_ids)


def _extract_request_result(
    request_output,
) -> tuple[torch.Tensor, list[int]]:
    step_logprobs, token_ids = _extract_step_logprobs(request_output)
    if step_logprobs is None or token_ids is None:
        pytest.skip(
            "Logits are not available on RequestOutput; "
            "enable logprobs return to run this test."
        )
    return step_logprobs, token_ids


def _assert_outputs_equal(
    baseline_logprobs: torch.Tensor,
    baseline_tokens: list[int],
    current_logprobs: torch.Tensor,
    current_tokens: list[int],
    run_label: str,
) -> None:
    if len(baseline_logprobs) != len(current_logprobs):
        print(
            f"\n[DIVERGENCE] {run_label}: different step count "
            f"{len(baseline_logprobs)} vs {len(current_logprobs)}"
        )
        print(f"  Run-1 tokens: {baseline_tokens}")
        print(f"  Run-2 tokens: {current_tokens}")
        pytest.fail(
            f"Divergence at {run_label}: step count mismatch "
            f"({len(baseline_logprobs)} vs {len(current_logprobs)})."
        )

    if baseline_tokens != current_tokens:
        print(f"\n[DIVERGENCE] {run_label}: different tokens sampled")
        print(f"  Run-1 tokens: {baseline_tokens}")
        print(f"  Run-2 tokens: {current_tokens}")
        pytest.fail(f"Divergence at {run_label}: different tokens sampled.")

    for step_idx, (baseline_step, current_step) in enumerate(
        zip(baseline_logprobs, current_logprobs)
    ):
        if baseline_step.shape != current_step.shape:
            print(
                f"\n[DIVERGENCE] {run_label}, step {step_idx}: "
                f"shape mismatch {baseline_step.shape} vs {current_step.shape}"
            )
            pytest.fail(
                f"Divergence at {run_label}, step {step_idx}: "
                f"shape mismatch {baseline_step.shape} vs {current_step.shape}."
            )

        if not torch.equal(baseline_step, current_step):
            max_diff = torch.abs(baseline_step - current_step).max().item()
            baseline_tok = (
                baseline_tokens[step_idx] if step_idx < len(baseline_tokens) else "N/A"
            )
            current_tok = (
                current_tokens[step_idx] if step_idx < len(current_tokens) else "N/A"
            )
            print(
                f"\n[DIVERGENCE] {run_label}, step {step_idx}: max_diff={max_diff:.6e}"
            )
            print(f"  Token IDs: Run-1={baseline_tok}, Run-2={current_tok}")
            print(f"  Run-1 logprob: {baseline_step.tolist()}")
            print(f"  Run-2 logprob: {current_step.tolist()}")
            print(f"  Run-1 all logprobs: {baseline_logprobs.tolist()}")
            print(f"  Run-2 all logprobs: {current_logprobs.tolist()}")
            pytest.fail(
                f"Divergence at {run_label}, step {step_idx}: "
                f"bitwise mismatch (max_diff={max_diff:.6e})."
            )


@skip_unsupported
@pytest.mark.parametrize("backend", BACKENDS)
def test_logprobs_chunked_prefill_batch_invariance(backend: str):
    """
    Submits the same seeded random prompt in multiple batch sizes with chunked
    prefill enabled and verifies that every output produces bitwise-identical
    logprobs regardless of batch size.
    """
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    from vllm import envs

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
        max_num_batched_tokens=CHUNKED_PREFILL_MAX_NUM_BATCHED_TOKENS,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
        attention_config={"backend": backend},
        mamba_cache_dtype="float32",
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=0.5,
        max_tokens=128,
        seed=1234,
        logprobs=5,
    )

    min_random_prompt = int(os.getenv("VLLM_MIN_PROMPT", "2048"))
    max_random_prompt = int(os.getenv("VLLM_MAX_PROMPT", "4096"))
    prompt = _random_prompt(min_random_prompt, max_random_prompt)
    prompt_token_ids = llm.get_tokenizer().encode(prompt)

    baseline_logprobs = None
    baseline_tokens = None

    for batch_size in CHUNKED_PREFILL_BATCH_SIZES:
        prompts = [
            TokensPrompt(prompt=prompt, prompt_token_ids=prompt_token_ids)
            for _ in range(batch_size)
        ]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        assert len(outs) == batch_size

        for run_in_batch, output in enumerate(outs):
            step_logprobs, token_ids = _extract_request_result(output)

            if baseline_logprobs is None:
                baseline_logprobs = step_logprobs
                baseline_tokens = token_ids
                continue
            assert baseline_tokens is not None
            run_label = f"batch_size={batch_size}, idx {run_in_batch}"
            _assert_outputs_equal(
                baseline_logprobs,
                baseline_tokens,
                step_logprobs,
                token_ids,
                run_label,
            )


@skip_unsupported
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mamba_cache_mode", MAMBA_CACHE_MODES)
def test_logprobs_prefix_caching_batch_invariance(
    backend: str,
    mamba_cache_mode: str,
):
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    random.seed(seed)
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))

    from vllm import envs

    disable_custom_ar = envs.VLLM_BATCH_INVARIANT

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print(f"{'=' * 80}\n")

    llm = None
    try:
        llm = LLM(
            model=TEST_MODEL,
            tensor_parallel_size=tp_size,
            max_num_seqs=64,
            max_model_len=8192,
            max_num_batched_tokens=PREFIX_CACHE_MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_cache_mode,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
            attention_config={"backend": backend},
            mamba_cache_dtype="float32",
        )

        effective_mode = llm.llm_engine.vllm_config.cache_config.mamba_cache_mode
        if effective_mode != mamba_cache_mode:
            pytest.skip(
                f"Requested mamba_cache_mode={mamba_cache_mode!r}, "
                f"but model resolved to {effective_mode!r}."
            )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.5,
            max_tokens=64,
            seed=1234,
            logprobs=5,
        )

        max_batch_size = max(PREFIX_CACHE_BATCH_SIZES)
        tokenized_prompts, _ = _build_prefix_cache_prompt_variants(
            llm,
            max_batch_size,
        )

        for batch_size in PREFIX_CACHE_BATCH_SIZES:
            _reset_prefix_cache(llm)

            batch_prompts = tokenized_prompts[:batch_size]
            run1_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            run2_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            assert len(run1_outputs) == batch_size
            assert len(run2_outputs) == batch_size

            run2_cached_tokens = [
                int(output.num_cached_tokens or 0) for output in run2_outputs
            ]

            if not all(cached_tokens > 0 for cached_tokens in run2_cached_tokens):
                pytest.fail(
                    f"Expected cache hits on second pass for batch_size={batch_size}, "
                    f"but got num_cached_tokens={run2_cached_tokens}."
                )

            for request_idx, (run1_output, run2_output) in enumerate(
                zip(run1_outputs, run2_outputs)
            ):
                run1_logprobs, run1_tokens = _extract_request_result(run1_output)
                run2_logprobs, run2_tokens = _extract_request_result(run2_output)
                _assert_outputs_equal(
                    run1_logprobs,
                    run1_tokens,
                    run2_logprobs,
                    run2_tokens,
                    run_label=(
                        f"backend={backend}, mode={effective_mode}, "
                        f"batch_size={batch_size}, request={request_idx}"
                    ),
                )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.llm_engine.engine_core.shutdown()
