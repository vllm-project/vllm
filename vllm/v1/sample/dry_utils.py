# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the DRY (Don't Repeat Yourself) penalty.

Used by both sampler stacks: the V1-runner builtin logits processor
(vllm/v1/sample/logits_processor/dry.py) and the V2-runner native module
(vllm/v1/worker/gpu/sample/dry.py).
"""

import weakref
from typing import Any

import numpy as np

# llama.cpp's FLOAT_MAX_LOG (src/llama-sampler.cpp): ln(float32 max).
_FLOAT_MAX_LOG = 88.7228391

# llama.cpp truncates sequence breaker strings to this many characters
# before matching them against the vocabulary.
_MAX_BREAKER_CHAR_LEN = 40

# Upper bound on distinct breaker sets cached per tokenizer; evicted sets
# re-resolve in a few ms.
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


# Per-tokenizer caches, weakly keyed so tokenizers can be collected:
# tokenizer -> decoded text of every vocab id, and
# tokenizer -> {breaker string tuple -> resolved breaker ids}.
_BreakerIdsPerTokenizer = dict[tuple[str, ...], list[int]]
_vocab_texts_cache: "weakref.WeakKeyDictionary[Any, list[str]]" = (
    weakref.WeakKeyDictionary()
)
_breaker_ids_cache: "weakref.WeakKeyDictionary[Any, _BreakerIdsPerTokenizer]" = (
    weakref.WeakKeyDictionary()
)


def resolve_dry_breakers(tokenizer: Any, breaker_strs: tuple[str, ...]) -> list[int]:
    """Resolve breaker strings to single-token breaker ids.

    llama.cpp parity (``get_overlapping_token_sequences``): every
    vocabulary token whose decoded text contains a breaker string acts
    as a single-token breaker, not just exact encodings (~3.9k of 128k
    ids for the default set on a Llama-3 tokenizer). llama.cpp
    additionally builds multi-token restart sequences from
    partially-overlapping tokens; those are not supported here.

    The O(vocab) decode runs once per tokenizer and the containment scan
    once per (tokenizer, breaker set), both cached. Breaker strings are
    truncated to ``_MAX_BREAKER_CHAR_LEN`` code points before caching and
    matching, bounding the cache key space (llama.cpp truncates to the
    same count of bytes; identical for ASCII breakers).
    """
    breaker_strs = tuple(s[:_MAX_BREAKER_CHAR_LEN] for s in breaker_strs if s)
    if not breaker_strs:
        return []
    per_tok = _breaker_ids_cache.setdefault(tokenizer, {})
    cached = per_tok.get(breaker_strs)
    if cached is not None:
        return list(cached)

    texts = _vocab_texts_cache.get(tokenizer)
    if texts is None:
        # The full id range, not vocab_size: added/special tokens (chat
        # markers like <|im_start|>) live above vocab_size on many
        # tokenizers, and llama.cpp's containment scan covers them too.
        n_ids = getattr(tokenizer, "max_token_id", tokenizer.vocab_size - 1) + 1
        texts = tokenizer.batch_decode([[i] for i in range(n_ids)])
        _vocab_texts_cache[tokenizer] = texts

    ids: set[int] = set()
    for s in breaker_strs:
        ids.update(i for i, text in enumerate(texts) if s in text)
    result = sorted(ids)
    if len(per_tok) >= _MAX_CACHED_BREAKER_SETS:
        # Evict the oldest entry (dict preserves insertion order).
        per_tok.pop(next(iter(per_tok)))
    per_tok[breaker_strs] = result
    return list(result)
