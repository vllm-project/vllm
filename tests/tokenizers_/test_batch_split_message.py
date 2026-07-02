# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tokenizer_mode='batch_split_message'.

Covers losslessness (segmented encode is byte-for-byte identical to a single
encode), the optional segment-id cache (hit/eviction bounds), the
split_delimiter override, and thread-safety under concurrent encode.

Uses a small Qwen tokenizer whose eos `<|im_end|>` is a registered special
token (hence a valid lossless split delimiter); its repo id is identical on
HF and ModelScope so the suite runs on either.
"""

import array
import copy
import threading

import pytest

from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.batch_split_message import (
    _segment_digest,
    _SegmentIdsCache,
    make_batch_split_message_tokenizer,
)

# A small open-source Qwen tokenizer. Its repo id is identical on HF and
# ModelScope, so upstream CI uses HF while an offline box can set
# VLLM_USE_MODELSCOPE=1 -- same test, no test-only env vars. Qwen uses ChatML,
# so eos (`<|im_end|>`) is the natural message delimiter and it adds no
# specials on encode (num_special_tokens_to_add == 0).
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EOS = "<|im_end|>"

# A mix of: no delimiter, leading/trailing delimiter, consecutive delimiters,
# multi-turn-like text, and empty string.
LOSSLESS_SAMPLES = [
    "",
    "hello world",
    f"a{EOS}b",
    f"{EOS}leading",
    f"trailing{EOS}",
    f"a{EOS}{EOS}b",
    f"system prompt here{EOS}user: hi there{EOS}assistant: hello!{EOS}",
    "no special tokens, just a longer plain sentence to encode in one shot.",
]


@pytest.fixture(scope="module")
def baseline():
    return get_tokenizer(MODEL, tokenizer_mode="hf", use_fast=True)


@pytest.fixture(scope="module")
def split_tok():
    return get_tokenizer(MODEL, tokenizer_mode="batch_split_message")


@pytest.fixture(scope="module")
def split_tok_cached():
    return get_tokenizer(
        MODEL, tokenizer_mode="batch_split_message", segment_cache=True
    )


@pytest.mark.parametrize("text", LOSSLESS_SAMPLES)
def test_lossless_matches_baseline(baseline, split_tok, text):
    assert list(split_tok.encode(text)) == list(baseline.encode(text))


@pytest.mark.parametrize("text", LOSSLESS_SAMPLES)
def test_lossless_with_cache(baseline, split_tok_cached, text):
    # Encode twice: first populates the cache, second hits it; both must match.
    assert list(split_tok_cached.encode(text)) == list(baseline.encode(text))
    assert list(split_tok_cached.encode(text)) == list(baseline.encode(text))


def test_add_special_tokens_false_matches_baseline(baseline, split_tok):
    text = f"a{EOS}b{EOS}c"
    assert list(split_tok.encode(text, add_special_tokens=False)) == list(
        baseline.encode(text, add_special_tokens=False)
    )


def test_truncation_left_keeps_tail(baseline):
    tok = get_tokenizer(MODEL, tokenizer_mode="batch_split_message")
    text = f"one{EOS}two{EOS}three{EOS}four"
    full = list(baseline.encode(text))
    got = tok.encode(text, truncation=True, max_length=3)
    assert got == full[-3:]  # generate runner -> truncation_side="left"


def test_completion_path_falls_back_when_tokenizer_adds_specials(baseline, monkeypatch):
    # Qwen adds no specials (n_special == 0). Force a tokenizer that *does* add
    # specials, so the completion path (add_special_tokens=True) must fall back
    # to a single encode instead of repeating them on every segment. monkeypatch
    # avoids depending on a real BOS-adding model being downloadable.
    base = copy.copy(baseline)  # independent instance; make_ swaps its __class__
    monkeypatch.setattr(base, "num_special_tokens_to_add", lambda *a, **k: 1)
    tok = make_batch_split_message_tokenizer(base, segment_cache=False)

    text = f"a{EOS}b{EOS}c"  # multi-segment
    # completion path (True): falls back to a single encode (not per-segment)
    assert list(tok.encode(text, add_special_tokens=True)) == list(
        baseline.encode(text, add_special_tokens=True)
    )
    # chat path (False): still segmented and lossless
    assert list(tok.encode(text, add_special_tokens=False)) == list(
        baseline.encode(text, add_special_tokens=False)
    )


def test_split_delimiter_override(baseline):
    # Override the delimiter; result must still equal a single encode().
    tok = get_tokenizer(
        MODEL, tokenizer_mode="batch_split_message", split_delimiter=EOS
    )
    text = f"x{EOS}y"
    assert list(tok.encode(text)) == list(baseline.encode(text))


def test_is_fast_and_decode_delegated(split_tok):
    # Non-encode methods are inherited from the underlying HF tokenizer.
    assert split_tok.is_fast
    ids = split_tok.encode("round trip test")
    assert isinstance(split_tok.decode(ids), str)


def test_concurrent_encode_consistent(baseline):
    tok = get_tokenizer(MODEL, tokenizer_mode="batch_split_message", segment_cache=True)
    texts = [f"thread {i} says hi{EOS}and bye{EOS}" for i in range(32)]
    expected = {t: list(baseline.encode(t)) for t in set(texts)}
    results: dict[int, list[int]] = {}
    errors: list[Exception] = []

    def worker(idx: int):
        try:
            results[idx] = list(tok.encode(texts[idx]))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(texts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i, text in enumerate(texts):
        assert results[i] == expected[text]


# ---------------------------------------------------------------------------
# _SegmentIdsCache unit tests
# ---------------------------------------------------------------------------


def _ids(n: int) -> "array.array":
    return array.array("i", range(n))


def test_cache_hit_refreshes_recency():
    c = _SegmentIdsCache(max_entries=2, max_total_tokens=1000)
    ka, kb = _segment_digest("a"), _segment_digest("b")
    c.put(ka, _ids(1))
    c.put(kb, _ids(1))
    # Touch "a" so "b" becomes LRU.
    assert c.get(ka) is not None
    c.put(_segment_digest("c"), _ids(1))  # evicts LRU == "b"
    assert c.get(ka) is not None
    assert c.get(kb) is None


def test_cache_evicts_on_entry_bound():
    c = _SegmentIdsCache(max_entries=2, max_total_tokens=1000)
    for s in ("a", "b", "c"):
        c.put(_segment_digest(s), _ids(1))
    assert len(c._data) == 2
    assert c.get(_segment_digest("a")) is None  # oldest evicted


def test_cache_evicts_on_token_bound():
    c = _SegmentIdsCache(max_entries=100, max_total_tokens=10)
    c.put(_segment_digest("a"), _ids(6))
    c.put(_segment_digest("b"), _ids(6))  # 12 > 10 -> evict "a"
    assert c.get(_segment_digest("a")) is None
    assert c.get(_segment_digest("b")) is not None
    assert c._total_tokens == 6


def test_cache_rejects_oversized_segment():
    c = _SegmentIdsCache(max_entries=100, max_total_tokens=10)
    c.put(_segment_digest("big"), _ids(20))  # larger than whole budget
    assert c.get(_segment_digest("big")) is None
    assert c._total_tokens == 0


def test_cache_concurrent_put_get_keeps_invariant():
    c = _SegmentIdsCache(max_entries=64, max_total_tokens=10_000)
    errors: list[Exception] = []

    def worker(base: int):
        try:
            for i in range(200):
                k = _segment_digest(f"{base}-{i % 80}")
                if c.get(k) is None:
                    c.put(k, _ids((i % 5) + 1))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Invariant: tracked total equals the sum of cached segment lengths.
    assert c._total_tokens == sum(len(v) for v in c._data.values())
    assert c._total_tokens <= 10_000
    assert len(c._data) <= 64
