# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the DRY (Don't Repeat Yourself) penalty.

Tokenizer-dependent setup for the DRY logits processor
(vllm/v1/worker/gpu/sample/dry.py). The match computation itself lives in
vllm/v1/sample/dry_core.py.
"""

import weakref
from typing import Any, NamedTuple

import numpy as np

# llama.cpp's FLOAT_MAX_LOG (src/llama-sampler.cpp): ln(float32 max).
_FLOAT_MAX_LOG = 88.7228391

# Breaker strings are truncated to this length before being matched against
# the vocabulary. llama.cpp's cap is this many BYTES (std::string::resize in
# llama_sampler_init_dry); ours is this many code points, which is the same
# cut for an ASCII breaker and a longer one for a multi-byte breaker.
_MAX_BREAKER_CHAR_LEN = 40

# Upper bound on distinct breaker sets cached per tokenizer; an evicted set
# re-resolves with a lookup per single-character breaker and a vocabulary
# scan per multi-character one.
_MAX_CACHED_BREAKER_SETS = 64


def max_exponent(base: float) -> int:
    """Exponent clamp, mirroring llama.cpp bit-for-bit.

    llama.cpp computes ``FLOAT_MAX_LOG / std::log(dry_base)`` entirely in
    float32. Computing the quotient in float64 lands on the wrong side of
    the integer truncation for some bases: at ``base=2.0`` float64 gives
    127.99999998 -> 127 while float32 gives exactly 128.0 -> 128.
    """
    if base <= 1.000001:
        return 0
    return int(np.float32(_FLOAT_MAX_LOG) / np.log(np.float32(base)))


class _VocabIndex(NamedTuple):
    """A tokenizer's decoded vocabulary, plus a character index into it.

    ``char_ids`` maps every character occurring anywhere in ``texts`` to the
    ids of the tokens whose text contains it. Breaker resolution runs in the
    worker's ``add_request``, inside ``execute_model``, where scanning a 150k
    vocabulary stalls the whole batch; a single-character breaker - which is
    what llama.cpp's defaults and the usual custom sets are - resolves by
    lookup instead. Multi-character breakers are rare and still scan ``texts``.

    Every character present is indexed, so one missing from ``char_ids``
    occurs in no token and its breaker resolves to no ids.
    """

    texts: list[str]
    char_ids: dict[str, list[int]]


# Per-tokenizer caches, weakly keyed so tokenizers can be collected:
# tokenizer -> its decoded vocabulary and character index, and
# tokenizer -> {breaker string tuple -> resolved breaker ids}.
_BreakerIdsPerTokenizer = dict[tuple[str, ...], list[int]]
_vocab_index_cache: "weakref.WeakKeyDictionary[Any, _VocabIndex]" = (
    weakref.WeakKeyDictionary()
)
_breaker_ids_cache: "weakref.WeakKeyDictionary[Any, _BreakerIdsPerTokenizer]" = (
    weakref.WeakKeyDictionary()
)


def _vocab_index(tokenizer: Any) -> _VocabIndex:
    """Decode every token of ``tokenizer`` once and index its characters."""
    index = _vocab_index_cache.get(tokenizer)
    if index is None:
        # The full id range, not vocab_size: added/special tokens (chat
        # markers like <|im_start|>) live above vocab_size on many
        # tokenizers, and llama.cpp's containment scan covers them too.
        n_ids = getattr(tokenizer, "max_token_id", tokenizer.vocab_size - 1) + 1
        texts = tokenizer.batch_decode([[i] for i in range(n_ids)])
        char_ids: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            for char in set(text):
                char_ids.setdefault(char, []).append(i)
        index = _VocabIndex(texts, char_ids)
        _vocab_index_cache[tokenizer] = index
    return index


def resolve_dry_breakers(tokenizer: Any, breaker_strs: tuple[str, ...]) -> list[int]:
    """Resolve breaker strings to single-token breaker ids.

    llama.cpp parity (``get_overlapping_token_sequences``): every
    vocabulary token whose decoded text contains a breaker string acts
    as a single-token breaker, not just exact encodings (~3.9k of 128k
    ids for the default set on a Llama-3 tokenizer). llama.cpp
    additionally builds multi-token restart sequences from
    partially-overlapping tokens; those are not supported here.

    The O(vocab) decode-and-index pass runs once per tokenizer and the
    resolved ids once per (tokenizer, breaker set), both cached. A
    single-character breaker then costs a lookup in the character index;
    only a multi-character one scans the vocabulary. Breaker strings are
    truncated to ``_MAX_BREAKER_CHAR_LEN`` code points before caching and
    matching, which bounds the length of a cache key, not how many keys
    there can be: ``_MAX_CACHED_BREAKER_SETS`` bounds that. llama.cpp's
    cap is the same number in bytes; see ``_MAX_BREAKER_CHAR_LEN``.
    """
    breaker_strs = tuple(s[:_MAX_BREAKER_CHAR_LEN] for s in breaker_strs if s)
    if not breaker_strs:
        return []
    per_tok = _breaker_ids_cache.setdefault(tokenizer, {})
    cached = per_tok.get(breaker_strs)
    if cached is not None:
        return list(cached)

    index = _vocab_index(tokenizer)
    ids: set[int] = set()
    for s in breaker_strs:
        if len(s) == 1:
            ids.update(index.char_ids.get(s, ()))
        else:
            ids.update(i for i, text in enumerate(index.texts) if s in text)
    result = sorted(ids)
    if len(per_tok) >= _MAX_CACHED_BREAKER_SETS:
        # Evict the oldest entry (dict preserves insertion order).
        per_tok.pop(next(iter(per_tok)))
    per_tok[breaker_strs] = result
    return list(result)
