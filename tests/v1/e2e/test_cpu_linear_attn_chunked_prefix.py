# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU chunked-prefill / prefix-caching correctness for linear-attention models.

Compares multi-chunk and warm-cache output against the full-prefill / cold-cache
reference. Batched-vs-batched: full prefill is not bit-stable across batch
composition, so the reference uses the same prompt list as the chunked run.
"""

import os

import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory; these engines
# only need a few thousand tokens.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "Qwen/Qwen3.5-0.8B"
CHUNK_TOKENS = 128  # max_num_batched_tokens for the chunked engine
SP = SamplingParams(max_tokens=32, temperature=0)


def _long_prompt(repeat: int) -> str:
    return "Solve the following arithmetic step by step. " * repeat + "What is 7*8?"


# Prompts long enough to span several CHUNK_TOKENS-sized chunks; a single-chunk
# prompt is bit-identical to full prefill regardless of the bug.
PROMPTS = [_long_prompt(r) for r in (40, 60, 80)]
# Spans several full cache blocks; prefix caching only reuses complete blocks.
PREFIX_PROMPT = "You are a helpful assistant. " * 230 + " Now answer: what is 2+2?"


def _make_llm(**overrides) -> LLM:
    base = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    base.update(overrides)
    return LLM(**base)


@pytest.fixture(scope="module")
def full_prefill_refs() -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """Reference greedy token IDs for PROMPTS and PREFIX_PROMPT under full prefill."""
    llm = _make_llm(enable_chunked_prefill=False, enable_prefix_caching=False)
    refs = [o.outputs[0].token_ids for o in llm.generate(PROMPTS, SP)]
    prefix_ref = llm.generate([PREFIX_PROMPT], SP)[0].outputs[0].token_ids
    del llm
    return refs, prefix_ref


def test_chunked_prefill_matches_full_prefill(full_prefill_refs):
    """Batched multi-chunk prefill must match per-prompt full prefill.

    The prompts are scheduled together so the scheduler interleaves prefill
    chunks from several in-flight requests (the cross-request path where the
    gsm8k accuracy gap was strongest).
    """
    refs, _ = full_prefill_refs
    llm = _make_llm(
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK_TOKENS,
        enable_prefix_caching=False,
    )
    got = [o.outputs[0].token_ids for o in llm.generate(PROMPTS, SP)]
    del llm

    mismatches = [i for i, (r, g) in enumerate(zip(refs, got)) if r != g]
    assert not mismatches, (
        f"chunked-prefill diverged from full prefill for prompts {mismatches}"
    )


def test_prefix_cache_hit_matches_cold_cache(full_prefill_refs):
    """A prefix-cache hit must match the cold-cache (reference) output.

    The warm run reuses cached blocks and continues prefill from the restored
    GDN state; the num_cached_tokens check guards against a vacuous (no-hit) pass.
    """
    _, ref = full_prefill_refs
    llm = _make_llm(enable_prefix_caching=True)
    llm.generate([PREFIX_PROMPT], SP)  # prime the cache
    warm_out = llm.generate([PREFIX_PROMPT], SP)[0]
    warm = warm_out.outputs[0].token_ids
    del llm

    assert warm_out.num_cached_tokens > 0, (
        "expected a prefix-cache hit but num_cached_tokens=0; "
        "PREFIX_PROMPT may be shorter than one cache block"
    )
    assert warm == ref, (
        f"prefix-cache-hit output diverged from cold-cache reference\n"
        f"  cold: {ref}\n  warm: {warm}"
    )
