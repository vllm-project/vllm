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
from vllm.transformers_utils.config import get_config

IS_DEVICE_CAPABILITY_BELOW_90 = is_device_capability_below_90()

TEST_MODEL = os.getenv(
    "VLLM_HYBRID_MAMBA_TEST_MODEL",
    "tiiuae/Falcon-H1-0.5B-Base",
)

TEST_SEED = 12345
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.4"))
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_CHUNKED_PREFILL_PROMPT_TOKENS = 4096
PREFIX_CACHE_RESET_TIMEOUT_S = 5.0
CHUNKED_PREFILL_MAX_TOKENS = 128
PREFIX_CACHE_MAX_TOKENS = 64

MAMBA_CACHE_MODES = ["align", "all"]
BACKENDS = ["TRITON_ATTN"]
CHUNKED_PREFILL_BATCH_SIZES = list(range(1, 11)) + [15, 16, 17]
PREFIX_CACHE_BATCH_SIZES = list(range(1, 9))

PREFIX_CACHE_SUFFIX_INSTRUCTION = (
    "Summarize the shared context in one concise sentence. "
    "Answer in one sentence with at most eighteen words."
)


def _get_model_mamba_chunk_size(model: str) -> int:
    hf_config = get_config(model, trust_remote_code=False)
    get_text_config = getattr(hf_config, "get_text_config", None)
    hf_text_config = get_text_config() if get_text_config is not None else hf_config
    chunk_size = getattr(hf_text_config, "mamba_chunk_size", None)
    if chunk_size is None:
        chunk_size = getattr(hf_text_config, "chunk_size", None)
    if chunk_size is None:
        chunk_size = 2048
    return int(chunk_size)


def _chunk_misaligned_default(target_tokens: int, mamba_chunk_size: int) -> int:
    return max(target_tokens, mamba_chunk_size) + max(1, mamba_chunk_size // 2)


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


def _build_prefix_cache_probe_prompts(
    llm: LLM,
    shared_prefix_token_count: int,
    unique_suffix_token_count: int,
) -> tuple[list[int], list[int]]:
    tokenizer = llm.get_tokenizer()
    shared_prefix_token_ids = _build_fixed_length_token_ids(
        tokenizer,
        shared_prefix_token_count,
        min_words=512,
        max_words=1024,
    )

    suffix_token_ids = _build_fixed_length_token_ids(
        tokenizer,
        unique_suffix_token_count,
        prefix_text=PREFIX_CACHE_SUFFIX_INSTRUCTION,
        min_words=64,
        max_words=128,
    )
    return shared_prefix_token_ids, shared_prefix_token_ids + suffix_token_ids


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
    random.seed(TEST_SEED)

    mamba_chunk_size = _get_model_mamba_chunk_size(TEST_MODEL)
    chunks = (DEFAULT_MAX_NUM_BATCHED_TOKENS + mamba_chunk_size - 1) // mamba_chunk_size
    max_num_batched_tokens = max(mamba_chunk_size, chunks * mamba_chunk_size)
    prompt_token_count = max(
        DEFAULT_CHUNKED_PREFILL_PROMPT_TOKENS, max_num_batched_tokens
    ) + max(1, mamba_chunk_size // 2)
    max_model_len = max(
        DEFAULT_MAX_MODEL_LEN,
        prompt_token_count + CHUNKED_PREFILL_MAX_TOKENS,
    )

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=128,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
        attention_config={"backend": backend},
        mamba_cache_dtype="float32",
    )

    sp = SamplingParams(
        temperature=0.0,
        top_p=0.5,
        max_tokens=CHUNKED_PREFILL_MAX_TOKENS,
        seed=TEST_SEED,
        logprobs=5,
    )

    prompt_token_ids = _build_fixed_length_token_ids(
        llm.get_tokenizer(),
        prompt_token_count,
        min_words=512,
        max_words=1024,
    )

    baseline_logprobs = None
    baseline_tokens = None

    for batch_size in CHUNKED_PREFILL_BATCH_SIZES:
        prompts = [
            TokensPrompt(prompt_token_ids=prompt_token_ids) for _ in range(batch_size)
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
    random.seed(TEST_SEED)

    mamba_chunk_size = _get_model_mamba_chunk_size(TEST_MODEL)
    max_num_batched_tokens = _chunk_misaligned_default(
        DEFAULT_MAX_NUM_BATCHED_TOKENS,
        mamba_chunk_size,
    )
    shared_prefix_token_count = max_num_batched_tokens
    unique_suffix_token_count = max_num_batched_tokens + max(1, mamba_chunk_size // 2)
    prompt_token_count = shared_prefix_token_count + unique_suffix_token_count
    max_model_len = max(
        DEFAULT_MAX_MODEL_LEN,
        prompt_token_count + PREFIX_CACHE_MAX_TOKENS,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.5,
        max_tokens=PREFIX_CACHE_MAX_TOKENS,
        seed=TEST_SEED,
        logprobs=5,
    )

    llm = LLM(
        model=TEST_MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=64,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=True,
        mamba_cache_mode=mamba_cache_mode,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
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

    shared_prefix_token_ids, prompt_token_ids = _build_prefix_cache_probe_prompts(
        llm,
        shared_prefix_token_count,
        unique_suffix_token_count,
    )

    baseline_output = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        sampling_params,
        use_tqdm=False,
    )[0]
    baseline_cached_tokens = int(baseline_output.num_cached_tokens or 0)
    if baseline_cached_tokens != 0:
        pytest.fail(
            "Expected uncached baseline run, but got "
            f"num_cached_tokens={baseline_cached_tokens}."
        )
    baseline_logprobs, baseline_tokens = _extract_request_result(baseline_output)

    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=TEST_SEED,
    )

    for batch_size in PREFIX_CACHE_BATCH_SIZES:
        _reset_prefix_cache(llm)

        llm.generate(
            [TokensPrompt(prompt_token_ids=shared_prefix_token_ids)],
            warmup_params,
            use_tqdm=False,
        )

        batch_prompts = [
            TokensPrompt(prompt_token_ids=prompt_token_ids) for _ in range(batch_size)
        ]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        assert len(outputs) == batch_size

        cached_tokens = [int(output.num_cached_tokens or 0) for output in outputs]

        if not all(num_cached_tokens > 0 for num_cached_tokens in cached_tokens):
            pytest.fail(
                f"Expected cache hits for batch_size={batch_size}, "
                f"but got num_cached_tokens={cached_tokens}."
            )

        for request_idx, output in enumerate(outputs):
            step_logprobs, token_ids = _extract_request_result(output)
            _assert_outputs_equal(
                baseline_logprobs,
                baseline_tokens,
                step_logprobs,
                token_ids,
                run_label=(
                    f"backend={backend}, mode={effective_mode}, "
                    f"batch_size={batch_size}, request={request_idx}"
                ),
            )
