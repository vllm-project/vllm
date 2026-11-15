# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The prefix token cache: ids equal to a whole encode, reuse across turns."""

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm.renderers import prefix_token_cache as ptc

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def _tokenizer(rstrip: bool = False) -> PreTrainedTokenizerFast:
    vocab = {chr(c): i for i, c in enumerate(range(32, 127))}
    merges = [("t", "h"), ("th", "e"), ("a", "n"), ("an", "d")]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=" "))
    bpe.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe.add_special_tokens(
        [AddedToken(s, rstrip=rstrip, normalized=False) for s in SPECIALS]
    )
    return PreTrainedTokenizerFast(tokenizer_object=bpe)


def _turns(n: int) -> list[str]:
    text, out = "", []
    for i in range(n):
        text += f"<|im_start|>user the and the turn {i} " * 40 + "<|im_end|>"
        out.append(text + "<|im_start|>assistant\n<think>")
        text += f"<|im_start|>assistant\n<think>and then</think>answer {i}<|im_end|>"
    return out


@pytest.fixture(autouse=True)
def _small(monkeypatch):
    monkeypatch.setattr(ptc, "MIN_CHARS", 1)


def test_ids_equal_whole_encode_and_reuse():
    tok = _tokenizer()
    cache = ptc.PrefixTokenCache.maybe_create(tok)
    assert cache is not None
    encoded = []
    real = cache._backend.encode_batch

    def counting(pieces, **kw):
        encoded.append(sum(len(p) for p in pieces))
        return real(pieces, **kw)

    cache._backend.encode_batch = counting
    texts = _turns(6)
    for text in texts:
        assert cache.encode(text) == tok.encode(text, add_special_tokens=False)
    assert cache.enabled
    # a later turn encodes only the characters the previous turn did not have
    for n, prev, text in zip(encoded[1:], texts, texts[1:]):
        assert n <= len(text) - len(prev) + len("and then</think>answer 0<|im_end|>")


def test_not_created_for_stripping_added_tokens():
    assert ptc.PrefixTokenCache.maybe_create(_tokenizer(rstrip=True)) is None


def test_eviction_keeps_ids_exact(monkeypatch):
    monkeypatch.setattr(ptc, "MAX_TOKENS", 2000)
    tok = _tokenizer()
    cache = ptc.PrefixTokenCache.maybe_create(tok)
    for text in _turns(8) + _turns(3):
        assert cache.encode(text) == tok.encode(text, add_special_tokens=False)
    assert cache._tokens <= 2000 or len(cache._entries) == 1
