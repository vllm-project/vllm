# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session KV reuse must not change outputs: continuing from retained KV gives
the same tokens and logprobs as recomputing the full history."""

import pytest

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _generate(llm: LLM, prompts: list[list[int]], max_tokens: int):
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    return llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False
    )


def _assert_same(reused, cold, tolerance: float) -> None:
    assert reused.outputs[0].token_ids == cold.outputs[0].token_ids
    for got, want in zip(reused.outputs[0].logprobs, cold.outputs[0].logprobs):
        token = next(iter(want))
        assert abs(got[token].logprob - want[token].logprob) <= tolerance


def _block_floor(llm: LLM, num_tokens: int) -> int:
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    return num_tokens - num_tokens % block_size


def _llm(model: str) -> LLM:
    return LLM(
        model,
        session_cache=True,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        seed=0,
        # Batch invariance is validated with the Triton attention backend.
        attention_backend="TRITON_ATTN",
    )


def test_attention_reuse_matches_recompute(monkeypatch: pytest.MonkeyPatch):
    """Attention-only: resumes at block boundaries and shares running prompts;
    bitwise equal to recomputation under batch invariance."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    tolerance = 0.0
    llm = _llm("Qwen/Qwen3-0.6B")
    tokenizer = llm.get_tokenizer()
    system = tokenizer.encode("You are a careful assistant. " * 40)
    first = system + tokenizer.encode("List three prime numbers.")
    (turn1,) = _generate(llm, [first], 24)
    history = first + list(turn1.outputs[0].token_ids)

    prompts = [
        # Same session, next turn.
        history + tokenizer.encode(" Now list three more."),
        # A fork from the middle of the first turn's output.
        history[: len(first) + 10] + tokenizer.encode(" Actually, stop."),
        # Another conversation sharing only the system prompt.
        system + tokenizer.encode("Name a color."),
    ]
    reused = _generate(llm, prompts, 24)
    # Attention-only models resume at the last block boundary.
    assert _block_floor(llm, len(history) - 1) == reused[0].num_cached_tokens
    assert _block_floor(llm, len(first) + 10) == reused[1].num_cached_tokens
    assert _block_floor(llm, len(system)) == reused[2].num_cached_tokens

    # Third turn on the same session after a concurrent fork.
    third = prompts[0] + list(reused[0].outputs[0].token_ids)
    (reused_third,) = _generate(llm, [third + tokenizer.encode(" Thanks.")], 16)
    assert _block_floor(llm, len(third) - 1) == reused_third.num_cached_tokens

    for prompt, got in zip(prompts, reused):
        assert llm.reset_prefix_cache()
        (cold,) = _generate(llm, [prompt], 24)
        assert cold.num_cached_tokens == 0
        _assert_same(got, cold, tolerance)
    assert llm.reset_prefix_cache()
    (cold_third,) = _generate(llm, [third + tokenizer.encode(" Thanks.")], 16)
    _assert_same(reused_third, cold_third, tolerance)


def test_hybrid_reuse_matches_recompute():
    """Hybrid (gated delta net): resumes only where a turn stopped. GDN has no
    batch-invariant kernels and a resumed recurrent state is summed in a
    different chunk order than one full prefill, so logprobs match to the
    tolerance prefix caching shows for the same resume (max ~0.08 measured)."""
    llm = _llm("Qwen/Qwen3.5-0.8B")
    tokenizer = llm.get_tokenizer()
    first = tokenizer.encode("You are a careful assistant. Tell me about rivers. " * 90)
    (turn1,) = _generate(llm, [first], 24)
    history = first + list(turn1.outputs[0].token_ids)
    prompts = [
        history + tokenizer.encode(" Please continue."),
        # Diverges before the commit point: nothing to resume from.
        history[: len(first) + 5] + tokenizer.encode(" Stop."),
    ]
    reused = _generate(llm, prompts, 16)
    assert reused[0].num_cached_tokens == len(history) - 1
    assert reused[1].num_cached_tokens == 0
    assert llm.reset_prefix_cache()
    (cold,) = _generate(llm, [prompts[0]], 16)
    assert cold.num_cached_tokens == 0
    got, want = reused[0].outputs[0], cold.outputs[0]
    assert got.token_ids[:8] == want.token_ids[:8]
    diffs = [
        abs(g[t].logprob - w[t].logprob)
        for t, g, w in zip(want.token_ids[:8], got.logprobs, want.logprobs)
    ]
    assert max(diffs) < 0.1


def test_hybrid_conversations_share_a_system_prompt():
    """Hybrid: a request that diverges from an earlier one inside its prompt
    saves its state there, so a third conversation with the same system
    prompt resumes from that state and matches a cold prefill."""
    llm = _llm("Qwen/Qwen3.5-0.8B")
    tokenizer = llm.get_tokenizer()
    system = tokenizer.encode(
        "You are a careful assistant. Tell me about rivers. " * 90
    )
    questions = [
        tokenizer.encode(f" Question {i}: name a river in country number {i}.")
        for i in range(3)
    ]
    _generate(llm, [system + questions[0]], 8)
    (second,) = _generate(llm, [system + questions[1]], 8)
    assert second.num_cached_tokens == 0
    (reused,) = _generate(llm, [system + questions[2]], 16)
    assert reused.num_cached_tokens >= len(system)
    assert llm.reset_prefix_cache()
    (cold,) = _generate(llm, [system + questions[2]], 16)
    got, want = reused.outputs[0], cold.outputs[0]
    assert got.token_ids[:8] == want.token_ids[:8]
    diffs = [
        abs(g[t].logprob - w[t].logprob)
        for t, g, w in zip(want.token_ids[:8], got.logprobs, want.logprobs)
    ]
    assert max(diffs) < 0.1


def test_eagle_reuse_matches_recompute():
    """Speculative decoding with an EAGLE3 drafter: a continuation reuses the
    previous turn and generates what a cold prefill generates."""
    llm = LLM(
        "meta-llama/Llama-3.1-8B-Instruct",
        speculative_config={
            "method": "eagle3",
            "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
            "num_speculative_tokens": 3,
        },
        session_cache=True,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        seed=0,
        attention_backend="TRITON_ATTN",
    )
    tokenizer = llm.get_tokenizer()
    first = tokenizer.encode("Tell me about the rivers of Europe. " * 40)
    (turn1,) = _generate(llm, [first], 24)
    history = first + list(turn1.outputs[0].token_ids)
    prompt = history + tokenizer.encode(" Now name the longest one.")
    (reused,) = _generate(llm, [prompt], 16)
    # At least one token short of the shared content, then block aligned.
    assert reused.num_cached_tokens >= _block_floor(llm, len(history) - 2)
    assert llm.reset_prefix_cache()
    (cold,) = _generate(llm, [prompt], 16)
    assert cold.num_cached_tokens == 0
    assert reused.outputs[0].token_ids[:8] == cold.outputs[0].token_ids[:8]


def test_hybrid_mtp_reuse_matches_recompute():
    """Hybrid with MTP speculative decoding: the state after verified tokens
    sits in a speculative block chosen by acceptance, so a continuation
    resumes where the previous prompt's prefill ended."""
    llm = LLM(
        "Qwen/Qwen3.5-0.8B",
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        session_cache=True,
        max_model_len=4096,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        seed=0,
        attention_backend="TRITON_ATTN",
    )
    tokenizer = llm.get_tokenizer()
    first = tokenizer.encode("You are a careful assistant. Tell me about rivers. " * 90)
    (turn1,) = _generate(llm, [first], 24)
    history = first + list(turn1.outputs[0].token_ids)
    prompt = history + tokenizer.encode(" Please continue.")
    (reused,) = _generate(llm, [prompt], 16)
    assert reused.num_cached_tokens == len(first)
    assert llm.reset_prefix_cache()
    (cold,) = _generate(llm, [prompt], 16)
    got, want = reused.outputs[0], cold.outputs[0]
    assert got.token_ids[:8] == want.token_ids[:8]
    diffs = [
        abs(g[t].logprob - w[t].logprob)
        for t, g, w in zip(want.token_ids[:8], got.logprobs, want.logprobs)
    ]
    assert max(diffs) < 0.1
