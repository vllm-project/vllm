# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the background-warmed DecodeStream prefix decode.

FastIncrementalDetokenizer primes tokenizers' DecodeStream with the full
prompt, whose O(prompt) prefix decode is deferred into the first step().
When constructed under a running event loop, the detokenizer primes with
ids[:-1] and steps the last prompt id in a background thread so that the
prefix decode overlaps prefill. These tests pin warm-vs-unwarmed output
equivalence (including prompts that end mid-UTF-8), the fail-closed
fallbacks, and the zero-per-token-overhead unshadowing of decode_next.
"""

import asyncio

import pytest
import tokenizers.decoders
from transformers import AutoTokenizer

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MIN_STEPS = 200

GEN_TEXT = (
    "Résultats de l'analyse: la frontière a obtenu « un niveau de calme "
    "inédit depuis les années 1960 » selon les données 🤖🎉 officielles. "
    "日本語のテキストと한국어 텍스트が混ざった文章です。"
    "Ombré façade naïve coöperate — ¿cuál es el propósito? "
) * 6


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _make_request(prompt_token_ids: list[int], **params_kwargs) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="test",
        external_req_id="test-ext",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(**params_kwargs),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _build_warmed(tokenizer, prompt_ids: list[int], **params_kwargs):
    """Construct under a running loop so the background warm arms.

    The warm submission is deferred one loop iteration (so it cannot delay
    the engine-core enqueue that follows the constructor); yield once so it
    runs. asyncio.run() shuts down the default executor before returning,
    so the warm step is guaranteed complete when this returns.
    """

    async def _ctor():
        detok = IncrementalDetokenizer.from_new_request(
            tokenizer, _make_request(prompt_ids, **params_kwargs)
        )
        # The submission must not run before the constructing task yields.
        if detok._warm_event is not None:
            assert detok._warm_pending_id is not None
        await asyncio.sleep(0)
        if detok._warm_event is not None:
            assert detok._warm_pending_id is None
        return detok

    return asyncio.run(_ctor())


def _run_and_collect(detok, gen_ids: list[int]) -> list[str]:
    texts = []
    for token_id in gen_ids:
        detok.update([token_id], False)
        texts.append(detok.output_text)
    return texts


def _edge_corpus(tokenizer) -> list[list[int]]:
    base_ids = tokenizer.encode(
        "The common prefix of the conversation so far. ", add_special_tokens=False
    )
    corpus = [base_ids]
    # Every split point of a multi-token emoji + CJK tail, covering prompts
    # that end mid-UTF-8 (byte-level BPE splits these across tokens).
    for tail in ("🤖", "🤖🎉🚀", "🏳️‍🌈", "日本語です", "naïve café"):
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        for k in range(1, len(tail_ids) + 1):
            corpus.append(base_ids + tail_ids[:k])
    corpus.append(base_ids * 40)  # long prompt
    corpus.append(tokenizer.encode("Hi", add_special_tokens=False)[:2])
    corpus = [ids for ids in corpus if len(ids) > 1]
    # Self-check the coverage claim: the corpus must contain prompts whose
    # decode ends mid-codepoint (U+FFFD), the class where the warm step
    # returns None and must fail closed to the lazy stream (#48854 class).
    assert any(tokenizer.decode(ids).endswith("�") for ids in corpus), (
        "edge corpus lost its mid-codepoint prompts"
    )
    return corpus


def test_warm_vs_unwarmed_equivalence(tokenizer):
    gen_ids = tokenizer.encode(GEN_TEXT, add_special_tokens=False)
    assert len(gen_ids) >= MIN_STEPS

    corpus = _edge_corpus(tokenizer)
    assert len(corpus) >= 10
    for prompt_ids in corpus:
        baseline = IncrementalDetokenizer.from_new_request(
            tokenizer, _make_request(prompt_ids)
        )
        assert type(baseline).__name__ == "FastIncrementalDetokenizer"
        assert baseline._warm_event is None

        warmed = _build_warmed(tokenizer, prompt_ids)
        assert warmed._warm_event is not None
        assert warmed._warm_event.is_set()

        assert _run_and_collect(warmed, gen_ids) == _run_and_collect(baseline, gen_ids)
        assert warmed.output_token_ids == baseline.output_token_ids


@pytest.mark.parametrize("tail", ["🤖🎉", "🏳️‍🌈"])
def test_warm_boundary_crossing_generation(tokenizer, tail):
    """Prompt ends mid-emoji and generation completes it."""
    base_ids = tokenizer.encode("Prefix text ", add_special_tokens=False)
    emoji_ids = tokenizer.encode(tail, add_special_tokens=False)
    gen_tail = tokenizer.encode(GEN_TEXT, add_special_tokens=False)[:MIN_STEPS]
    mid_codepoint_splits = 0
    for k in range(1, len(emoji_ids)):
        prompt_ids = base_ids + emoji_ids[:k]
        if tokenizer.decode(prompt_ids).endswith("�"):
            mid_codepoint_splits += 1
        gen_ids = emoji_ids[k:] + gen_tail
        baseline = IncrementalDetokenizer.from_new_request(
            tokenizer, _make_request(prompt_ids)
        )
        warmed = _build_warmed(tokenizer, prompt_ids)
        assert _run_and_collect(warmed, gen_ids) == _run_and_collect(baseline, gen_ids)
    if tail == "🏳️‍🌈":
        # Self-check: this tail must split mid-codepoint so the completion
        # crosses a prompt boundary the warm step could not decode.
        assert mid_codepoint_splits > 0


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_warm_special_token_tail(tokenizer, skip_special_tokens):
    prompt_ids = tokenizer.encode("Hello there", add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    gen_ids = tokenizer.encode(GEN_TEXT, add_special_tokens=False)[:MIN_STEPS]
    kwargs = dict(
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=skip_special_tokens,
    )
    baseline = IncrementalDetokenizer.from_new_request(
        tokenizer, _make_request(prompt_ids, **kwargs)
    )
    warmed = _build_warmed(tokenizer, prompt_ids, **kwargs)
    assert warmed._warm_event is not None
    assert _run_and_collect(warmed, gen_ids) == _run_and_collect(baseline, gen_ids)


def test_short_prompt_skips_warm(tokenizer):
    prompt_ids = tokenizer.encode("Hi", add_special_tokens=False)[:1]

    detok = _build_warmed(tokenizer, prompt_ids)
    assert detok._warm_event is None
    assert "decode_next" not in detok.__dict__

    gen_ids = tokenizer.encode(GEN_TEXT, add_special_tokens=False)[:32]
    baseline = IncrementalDetokenizer.from_new_request(
        tokenizer, _make_request(prompt_ids)
    )
    assert _run_and_collect(detok, gen_ids) == _run_and_collect(baseline, gen_ids)


def test_sync_engine_skips_warm(tokenizer):
    """No running event loop (sync LLMEngine) keeps today's lazy path."""
    prompt_ids = tokenizer.encode("A sync engine prompt", add_special_tokens=False)
    detok = IncrementalDetokenizer.from_new_request(
        tokenizer, _make_request(prompt_ids)
    )
    assert detok._warm_event is None
    assert "decode_next" not in detok.__dict__


def test_decode_next_unshadowed_after_first_token(tokenizer):
    """Steady per-token path must be the unmodified class method."""
    prompt_ids = tokenizer.encode("Some prompt for streaming", add_special_tokens=False)
    detok = _build_warmed(tokenizer, prompt_ids)
    assert "decode_next" in detok.__dict__
    detok.update(tokenizer.encode("Hello", add_special_tokens=False)[:1], False)
    assert "decode_next" not in detok.__dict__


def test_warm_step_failure_falls_back(tokenizer, monkeypatch):
    """A failing warm step rebuilds the fully primed lazy stream."""
    prompt_ids = tokenizer.encode(
        "A prompt whose warm step is going to fail 🤖", add_special_tokens=False
    )
    gen_ids = tokenizer.encode(GEN_TEXT, add_special_tokens=False)[:MIN_STEPS]
    baseline = IncrementalDetokenizer.from_new_request(
        tokenizer, _make_request(prompt_ids)
    )

    real_stream_cls = tokenizers.decoders.DecodeStream
    instances: list[object] = []

    class FailFirstStepStream:
        def __init__(self, *args, **kwargs):
            self._inner = real_stream_cls(*args, **kwargs)
            # Poison only the first constructed stream (the warm target);
            # the fallback stream must behave normally.
            self._poisoned = not instances
            instances.append(self)

        def step(self, tok, token_id):
            if self._poisoned:
                self._poisoned = False
                raise RuntimeError("injected warm failure")
            return self._inner.step(tok, token_id)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", FailFirstStepStream)
    warmed = _build_warmed(tokenizer, prompt_ids)
    assert warmed._warm_event is not None
    assert warmed._warm_event.is_set()
    assert len(instances) == 2

    assert _run_and_collect(warmed, gen_ids) == _run_and_collect(baseline, gen_ids)
