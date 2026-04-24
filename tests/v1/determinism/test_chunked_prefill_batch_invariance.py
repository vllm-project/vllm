# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Chunked-prefill batch invariance test.

Chunked prefill divergence is a known hazard for Mamba and hybrid
(attention+SSM) models.  Mamba-2's chunked scan algorithm computes the
recurrent state over fixed-size token chunks; when a prefill is split by the
scheduler, each split must land on a Mamba chunk boundary.  An off-boundary
split produces a different intermediate scan state, which yields divergent
logprobs compared to processing the same sequence in a single chunk.  This
test sets max_num_batched_tokens=2048 so that batches of 5+ requests with a
~500-token prompt will trigger chunked prefill, exposing the divergence if the
implementation is incorrect.
"""

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

from vllm import LLM, SamplingParams, TokensPrompt

IS_DEVICE_CAPABILITY_BELOW_90 = is_device_capability_below_90()

MAX_BATCH_SIZE = int(os.getenv("VLLM_DIVERGENCE_MAX_BATCH_SIZE", "10"))
EXTRA_BATCH_SIZES = os.getenv("VLLM_DIVERGENCE_EXTRA_BATCH_SIZES", "15,16,17")


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


def _build_batch_sizes() -> list[int]:
    # Keep the original sweep and add boundary sizes where cudagraph buckets
    # commonly change behavior.
    batch_sizes = list(range(1, MAX_BATCH_SIZE + 1))
    batch_sizes.extend(_parse_positive_int_csv(EXTRA_BATCH_SIZES))
    return sorted(set(batch_sizes))


BATCH_SIZES = _build_batch_sizes()


@skip_unsupported
@pytest.mark.parametrize(
    "backend",
    BACKENDS,
)
def test_logprobs_chunked_prefill_batch_invariance(
    backend,
):
    """
    Submits the same seeded random prompt in multiple batch sizes with chunked
    prefill enabled (max_num_batched_tokens=2048) and verifies that every
    output produces bitwise-identical logprobs regardless of batch size.

    With the default prompt length and MAX_BATCH_SIZE=10, larger batches will
    exceed the 2048-token window and trigger chunked prefill.  For Mamba and
    hybrid (attention+SSM) models the scheduler must ensure splits land on
    Mamba chunk boundaries; this test exposes any off-boundary divergence.

    The test fails fast on the first divergence to avoid wasting GPU time.
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
        max_num_batched_tokens=2048,
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
    total_runs = sum(BATCH_SIZES)

    print("\n" + "=" * 80)
    print(
        f"STARTING CHUNKED-PREFILL BATCH INVARIANCE TEST "
        f"(batch sizes {BATCH_SIZES}, {total_runs} total runs, "
        f"prompt tokens={len(prompt_token_ids)})"
    )
    print("=" * 80 + "\n")

    baseline_logprobs = None
    baseline_tokens = None
    total_checked = 0

    for batch_size in BATCH_SIZES:
        prompts = [
            TokensPrompt(prompt=prompt, prompt_token_ids=prompt_token_ids)
            for _ in range(batch_size)
        ]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        assert len(outs) == batch_size

        for run_in_batch, o in enumerate(outs):
            step_logprobs, token_ids = _extract_step_logprobs(o)
            if step_logprobs is None:
                pytest.skip(
                    "Logits are not available on RequestOutput; "
                    "enable logprobs return to run this test."
                )

            if baseline_logprobs is None:
                baseline_logprobs = step_logprobs
                baseline_tokens = token_ids
                print(f"[Baseline] batch_size=1, idx 0 tokens={token_ids}")
                total_checked += 1
                continue
            assert baseline_tokens is not None
            total_checked += 1
            run_label = f"batch_size={batch_size}, idx {run_in_batch}"

            if len(baseline_logprobs) != len(step_logprobs):
                print(
                    f"\n[DIVERGENCE] {run_label}: "
                    f"different step count "
                    f"{len(baseline_logprobs)} vs {len(step_logprobs)}"
                )
                print(f"  Baseline tokens: {baseline_tokens}")
                print(f"  Current  tokens: {token_ids}")
                pytest.fail(
                    f"Divergence at {run_label}: step count mismatch "
                    f"({len(baseline_logprobs)} vs {len(step_logprobs)}). "
                    f"Checked {total_checked}/{total_runs} runs."
                )

            if baseline_tokens != token_ids:
                print(f"\n[DIVERGENCE] {run_label}: different tokens sampled")
                print(f"  Baseline tokens: {baseline_tokens}")
                print(f"  Current  tokens: {token_ids}")
                pytest.fail(
                    f"Divergence at {run_label}: different tokens sampled. "
                    f"Checked {total_checked}/{total_runs} runs."
                )

            for t, (a, b) in enumerate(zip(baseline_logprobs, step_logprobs)):
                if a.shape != b.shape:
                    print(
                        f"\n[DIVERGENCE] {run_label}, "
                        f"step {t}: shape mismatch {a.shape} vs {b.shape}"
                    )
                    pytest.fail(
                        f"Divergence at {run_label}, step {t}: "
                        f"shape mismatch {a.shape} vs {b.shape}. "
                        f"Checked {total_checked}/{total_runs} runs."
                    )

                if not torch.equal(a, b):
                    max_diff = torch.abs(a - b).max().item()
                    print(
                        f"\n[DIVERGENCE] {run_label}, step {t}: max_diff={max_diff:.6e}"
                    )
                    baseline_tok = (
                        baseline_tokens[t] if t < len(baseline_tokens) else "N/A"
                    )
                    current_tok = token_ids[t] if t < len(token_ids) else "N/A"
                    print(
                        f"  Token IDs: baseline={baseline_tok}, current={current_tok}"
                    )
                    print(f"  Baseline logprobs: {a.tolist()}")
                    print(f"  Current  logprobs: {b.tolist()}")
                    print(
                        "  Baseline all logprobs: "
                        + str(
                            [
                                baseline_logprobs[s].tolist()
                                for s in range(len(baseline_logprobs))
                            ]
                        )
                    )
                    print(
                        "  Current  all logprobs: "
                        + str(
                            [
                                step_logprobs[s].tolist()
                                for s in range(len(step_logprobs))
                            ]
                        )
                    )
                    pytest.fail(
                        f"Divergence at {run_label}, step {t}: "
                        f"bitwise mismatch (max_diff={max_diff:.6e}). "
                        f"Checked {total_checked}/{total_runs} runs."
                    )

        print(
            f"[batch_size={batch_size}] {batch_size} runs OK "
            f"(total checked: {total_checked}/{total_runs})"
        )

    print(f"\n{'=' * 80}")
    print(
        f"SUCCESS: All {total_checked} runs produced bitwise-identical "
        f"logprobs across batch sizes {BATCH_SIZES} with chunked prefill "
        f"(max_num_batched_tokens=2048)."
    )
    print(f"{'=' * 80}\n")
