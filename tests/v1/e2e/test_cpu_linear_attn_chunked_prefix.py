# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU chunked-prefill / prefix-caching correctness for linear-attention models."""

import os

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory; these engines
# only need a few thousand tokens.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "Qwen/Qwen3.5-0.8B"
CHUNK_TOKENS = 128  # max_num_batched_tokens for the chunked engine
NUM_LOGPROBS = 5
SP = SamplingParams(max_tokens=32, temperature=0, logprobs=NUM_LOGPROBS)


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


def _tuples(outputs) -> list[tuple[list[int], str, object]]:
    """(token_ids, text, sample_logprobs) per request, for check_logprobs_close."""
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


@pytest.fixture(scope="module")
def full_prefill_refs():
    """Reference (ids, text, logprobs) for PROMPTS and PREFIX_PROMPT, full prefill."""
    llm = _make_llm(enable_chunked_prefill=False, enable_prefix_caching=False)
    refs = _tuples(llm.generate(PROMPTS, SP))
    prefix_ref = _tuples(llm.generate([PREFIX_PROMPT], SP))[0]
    del llm
    return refs, prefix_ref


def test_chunked_prefill_matches_full_prefill(full_prefill_refs):
    """Batched multi-chunk prefill must stay close to per-prompt full prefill.

    Prompts are scheduled together so the scheduler interleaves prefill chunks
    across requests (the cross-request path where the accuracy gap was strongest).
    """
    refs, _ = full_prefill_refs
    llm = _make_llm(
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK_TOKENS,
        enable_prefix_caching=False,
    )
    got = _tuples(llm.generate(PROMPTS, SP))
    del llm

    check_logprobs_close(
        outputs_0_lst=refs,
        outputs_1_lst=got,
        name_0="full_prefill",
        name_1="chunked_prefill",
    )


def test_prefix_cache_hit_matches_cold_cache(full_prefill_refs):
    """A prefix-cache hit must stay close to the cold-cache (reference) output.

    The warm run continues prefill from the restored GDN state; the
    num_cached_tokens check guards against a vacuous (no-hit) pass.
    """
    _, ref = full_prefill_refs
    llm = _make_llm(enable_prefix_caching=True)
    llm.generate([PREFIX_PROMPT], SP)  # prime the cache
    warm_out = llm.generate([PREFIX_PROMPT], SP)[0]
    warm = _tuples([warm_out])[0]
    del llm

    assert warm_out.num_cached_tokens > 0, (
        "expected a prefix-cache hit but num_cached_tokens=0; "
        "PREFIX_PROMPT may be shorter than one cache block"
    )
    check_logprobs_close(
        outputs_0_lst=[ref],
        outputs_1_lst=[warm],
        name_0="cold_cache",
        name_1="warm_cache",
    )
