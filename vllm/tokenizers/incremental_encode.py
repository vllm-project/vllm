# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact incremental prompt encoding for multi-turn conversations.

In multi-turn chat, each request re-sends the whole conversation, so the
rendered prompt of turn ``N`` is almost always a strict string prefix of the
prompt of turn ``N + 1``. Re-tokenizing the full prompt costs time linear in
the conversation length even though only the tail is new.
[`IncrementalEncodeCache`][vllm.tokenizers.incremental_encode.IncrementalEncodeCache]
reuses the token ids of a previously encoded prefix and re-encodes only the
tail, while remaining token-exact with respect to a full re-encode.

Correctness rests on two mechanisms:

1. *Pre-token boundary invariant*: BPE merges never cross pre-token
   boundaries, so the token ids strictly before a pre-token boundary do not
   depend on the text at or after it. The splice point is chosen by backing
   up ``backup_chars`` characters from the end of the cached prefix and
   asking the backend pre-tokenizer (``pre_tokenizer.pre_tokenize_str``) for
   boundary candidates at least ``margin_chars`` away from both window
   edges, keeping the seam far from the newly appended text. The seam is
   the first candidate whose neighborhood is free of special-token markers
   and which coincides with a token boundary in the *cached* encoding
   (checked via its character offsets), which rules out seams inside
   added/special tokens.

2. *Overlap verification with full-encode fallback*: the tail re-encode
   overlaps the cached encoding by roughly ``backup_chars`` characters. The
   overlapping token ids and offsets must match the cached ones exactly
   (excluding tokens within ``margin_chars`` of the cached text's end, which
   may legitimately merge with the appended text). Any mismatch — or any
   condition this module cannot prove safe: no pre-token boundary in the
   window, special-token markers near the seam, a normalizer that rewrites
   the window, incompatible encode kwargs, or a result that would exceed a
   truncation limit — falls back to a full re-encode, which is also what
   (re)fills the cache.

The cache is renderer-level state shared across requests: the OpenAI
frontend is stateless, so entries are matched purely by longest string
prefix within a bounded LRU of recent conversations, optionally partitioned
by ``cache_salt``. It is thread-safe: only the cheap lookup/insert steps
take the lock, and all tokenizer calls run outside of it against whatever
thread-safe tokenizer the caller passes in (in vLLM, the renderer's
pool-backed HF fast tokenizer).

This module deliberately has no vLLM dependencies so that the cache can be
unit-tested against plain ``transformers`` tokenizers.
"""

import contextlib
import logging
import threading
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_Offsets = list[tuple[int, int]]

_SPECIAL_MARKER = "<|"
"""Common special-token sigil; seams near it are treated as unsafe."""

_AFFIX_PROBE_TEXTS = ("Hello, world!", "def f(x):\n    return x  # 你好")
"""Texts used to validate the ``prefix + body + suffix`` special-token
model of a tokenizer (e.g. a leading BOS token)."""


@dataclass
class IncrementalEncodeStats:
    hits: int = 0
    """Requests served by splicing (or exactly matching) a cached prefix."""

    misses: int = 0
    """Requests with no cached prefix; fully encoded and cached."""

    fallbacks: int = 0
    """Requests with a cached prefix that could not be proven safe to
    splice; fully encoded and cached."""


@dataclass
class _CacheEntry:
    salt: str | None
    text: str
    token_ids: array
    """Token ids of ``text`` encoded without special tokens (``"i"`` typecode
    to bound memory for very long conversations)."""

    offsets: _Offsets
    """Character offsets of the trailing tokens of ``token_ids`` (only the
    tail is needed to locate and verify future splice points)."""

    offsets_start: int
    """Index into ``token_ids`` of the token described by ``offsets[0]``."""


class IncrementalEncodeCache:
    """Bounded LRU of ``(text, token_ids)`` pairs with prefix splicing.

    A cache instance must only ever be used with a single logical tokenizer;
    tokenizer capabilities and special-token affixes are probed once and
    memoized. Requests that this cache cannot serve exactly (unsupported
    tokenizer, incompatible kwargs, short prompts, truncation overflow)
    return ``None`` and must be handled by the caller's regular encode path.
    """

    def __init__(
        self,
        *,
        max_entries: int = 16,
        min_chars: int = 16384,
        backup_chars: int = 1024,
        margin_chars: int = 96,
    ) -> None:
        if min_chars < backup_chars + 2 * margin_chars:
            raise ValueError(
                "min_chars must be at least backup_chars + 2 * margin_chars "
                f"({backup_chars + 2 * margin_chars}), got {min_chars}"
            )
        self.max_entries = max_entries
        self.min_chars = min_chars
        self.backup_chars = backup_chars
        self.margin_chars = margin_chars
        self.stats = IncrementalEncodeStats()

        self._entries: list[_CacheEntry] = []
        self._lock = threading.Lock()
        self._supported: bool | None = None
        self._pre_tokenizer: Any = None
        self._normalizer: Any = None
        self._affixes: dict[bool, tuple[list[int], list[int]] | None] = {}

    def encode(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        cache_salt: str | None = None,
        **kwargs: Any,
    ) -> list[int] | None:
        """Encode ``text``, reusing a cached prefix when provably exact.

        Returns the token ids (identical to
        ``tokenizer(text, add_special_tokens=..., truncation=...,
        max_length=...)["input_ids"]``), or ``None`` if this request cannot
        be served by the cache and the caller should encode normally.
        """
        if kwargs or not isinstance(text, str) or len(text) < self.min_chars:
            return None
        if truncation and max_length is None:
            return None
        if not self._check_tokenizer(tokenizer):
            return None

        affixes = self._get_affixes(tokenizer, add_special_tokens)
        if affixes is None:
            return None
        prefix_ids, suffix_ids = affixes

        entry = self._find_prefix(text, cache_salt)
        if entry is None:
            self.stats.misses += 1
            body = self._encode_and_store(tokenizer, text, cache_salt)
        elif entry.text == text:
            self.stats.hits += 1
            self._touch(entry)
            body = entry.token_ids
        else:
            body = self._splice(tokenizer, entry, text, cache_salt)
            if body is None:
                self.stats.fallbacks += 1
                body = self._encode_and_store(tokenizer, text, cache_salt)
            else:
                self.stats.hits += 1

        token_ids = prefix_ids + body.tolist() + suffix_ids
        if truncation and max_length is not None and len(token_ids) > max_length:
            # The cache is warm, but truncation semantics (side, pairing
            # with special tokens) belong to the regular encode path.
            return None
        return token_ids

    # Tokenizer capability probing

    def _check_tokenizer(self, tokenizer: "PreTrainedTokenizerFast") -> bool:
        if self._supported is None:
            backend = (
                getattr(tokenizer, "backend_tokenizer", None)
                if getattr(tokenizer, "is_fast", False)
                else None
            )
            pre_tokenizer = getattr(backend, "pre_tokenizer", None)
            if pre_tokenizer is None:
                logger.debug(
                    "Incremental encoding disabled: tokenizer %s has no "
                    "fast-backend pre-tokenizer",
                    type(tokenizer).__name__,
                )
                self._supported = False
            else:
                self._pre_tokenizer = pre_tokenizer
                self._normalizer = backend.normalizer
                self._supported = True
        return self._supported

    def _get_affixes(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        add_special_tokens: bool,
    ) -> tuple[list[int], list[int]] | None:
        """Special tokens the tokenizer adds around the body, or ``None``.

        Only tokenizers whose special tokens form a content-independent
        ``prefix + body + suffix`` template (validated against probe texts)
        are supported for ``add_special_tokens=True``.
        """
        key = bool(add_special_tokens)
        if key not in self._affixes:
            self._affixes[key] = self._compute_affixes(tokenizer, key)
        return self._affixes[key]

    @staticmethod
    def _compute_affixes(
        tokenizer: "PreTrainedTokenizerFast",
        add_special_tokens: bool,
    ) -> tuple[list[int], list[int]] | None:
        if not add_special_tokens:
            return [], []
        try:
            empty = list(tokenizer("", add_special_tokens=True)["input_ids"])
            probes = [
                (
                    list(tokenizer(t, add_special_tokens=True)["input_ids"]),
                    list(tokenizer(t, add_special_tokens=False)["input_ids"]),
                )
                for t in _AFFIX_PROBE_TEXTS
            ]
        except Exception:
            logger.debug(
                "Incremental encoding: special-token probe failed", exc_info=True
            )
            return None
        for split in range(len(empty), -1, -1):
            prefix, suffix = empty[:split], empty[split:]
            if all(with_st == prefix + body + suffix for with_st, body in probes):
                return prefix, suffix
        logger.debug(
            "Incremental encoding disabled for add_special_tokens=True: "
            "special tokens of %s do not follow a prefix/suffix template",
            type(tokenizer).__name__,
        )
        return None

    # Cache bookkeeping (all lock-protected sections are O(max_entries))

    def _find_prefix(self, text: str, salt: str | None) -> _CacheEntry | None:
        with self._lock:
            candidates = [e for e in self._entries if e.salt == salt]
        best = None
        for entry in candidates:
            if (
                len(entry.text) <= len(text)
                and text.startswith(entry.text)
                and (best is None or len(entry.text) > len(best.text))
            ):
                best = entry
        return best

    def _touch(self, entry: _CacheEntry) -> None:
        with self._lock:
            with contextlib.suppress(ValueError):  # concurrently evicted
                self._entries.remove(entry)
            self._entries.append(entry)

    def _store(
        self,
        salt: str | None,
        text: str,
        token_ids: array,
        offsets: _Offsets,
        offsets_start: int,
    ) -> None:
        # Future splice points fall within the last `backup_chars` of `text`,
        # so only the offsets of the trailing tokens need to be retained;
        # keep 2x for slack against short subsequent turns.
        keep_from = len(text) - 2 * self.backup_chars
        cut = 0
        while cut < len(offsets) and offsets[cut][1] < keep_from:
            cut += 1
        entry = _CacheEntry(
            salt=salt,
            text=text,
            token_ids=token_ids,
            offsets=offsets[cut:],
            offsets_start=offsets_start + cut,
        )
        with self._lock:
            self._entries = [
                e for e in self._entries if not (e.salt == salt and e.text == text)
            ]
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                del self._entries[0]

    # Encoding

    def _encode_with_offsets(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        text: str,
    ) -> tuple[array, _Offsets]:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        return array("i", encoding["input_ids"]), encoding["offset_mapping"]

    def _encode_and_store(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        text: str,
        salt: str | None,
    ) -> array:
        token_ids, offsets = self._encode_with_offsets(tokenizer, text)
        self._store(salt, text, token_ids, offsets, 0)
        return token_ids

    def _splice(
        self,
        tokenizer: "PreTrainedTokenizerFast",
        entry: _CacheEntry,
        text: str,
        salt: str | None,
    ) -> array | None:
        """Reuse ``entry``'s token ids up to a safe boundary, re-encode the
        rest of ``text``, and cache the result. ``None`` if unsafe."""
        cached_len = len(entry.text)
        base = cached_len - self.backup_chars
        window = text[base:cached_len]

        if self._normalizer is not None:
            try:
                if self._normalizer.normalize_str(window) != window:
                    return None
            except Exception:
                return None

        boundary = split = None
        for candidate in self._pre_token_boundaries(window, base):
            seam = text[candidate - self.margin_chars : candidate + self.margin_chars]
            if _SPECIAL_MARKER in seam:
                continue
            split = self._boundary_token_index(entry, candidate)
            if split is not None:
                boundary = candidate
                break
        if boundary is None or split is None:
            return None

        tail_ids, tail_offsets = self._encode_with_offsets(tokenizer, text[boundary:])
        if not self._verify_overlap(entry, split, boundary, tail_ids, tail_offsets):
            logger.debug(
                "Incremental encoding: overlap verification failed at "
                "boundary %d; falling back to full encode",
                boundary,
            )
            return None

        token_ids = entry.token_ids[:split]
        token_ids.extend(tail_ids)
        offsets_head = entry.offsets[: split - entry.offsets_start]
        offsets = offsets_head + [
            (start + boundary, end + boundary) for start, end in tail_offsets
        ]
        self._store(salt, text, token_ids, offsets, entry.offsets_start)
        return token_ids

    def _pre_token_boundaries(self, window: str, base: int) -> list[int]:
        """Absolute char indices of pre-token boundaries at least
        ``margin_chars`` away from both ends of ``window``, earliest first
        (i.e. candidates furthest from the appended text come first)."""
        try:
            segments = self._pre_tokenizer.pre_tokenize_str(window)
        except Exception:
            logger.debug("Incremental encoding: pre_tokenize_str failed", exc_info=True)
            return []
        limit = len(window) - self.margin_chars
        return [
            base + start
            for _piece, (start, _end) in segments
            if self.margin_chars <= start <= limit
        ]

    @staticmethod
    def _boundary_token_index(entry: _CacheEntry, boundary: int) -> int | None:
        """Index of the first cached token starting at ``boundary``, given
        that the previous token ends exactly there; ``None`` otherwise."""
        offsets = entry.offsets
        for k in range(len(offsets) - 1, -1, -1):
            end = offsets[k][1]
            if end < boundary:
                return None
            if end == boundary:
                if k + 1 >= len(offsets) or offsets[k + 1][0] != boundary:
                    return None
                return entry.offsets_start + k + 1
        return None

    def _verify_overlap(
        self,
        entry: _CacheEntry,
        split: int,
        boundary: int,
        tail_ids: array,
        tail_offsets: _Offsets,
    ) -> bool:
        """Check that the tail re-encode reproduces the cached tokens over
        the overlap region (tokens within ``margin_chars`` of the cached
        text's end are excluded, as they may merge with appended text)."""
        verify_end = len(entry.text) - self.margin_chars
        k = split - entry.offsets_start
        num_verified = 0
        while k < len(entry.offsets) and entry.offsets[k][1] <= verify_end:
            k += 1
            num_verified += 1
        if num_verified == 0:
            return False
        if entry.token_ids[split : split + num_verified] != tail_ids[:num_verified]:
            return False
        expected = entry.offsets[
            split - entry.offsets_start : split - entry.offsets_start + num_verified
        ]
        actual = [
            (start + boundary, end + boundary)
            for start, end in tail_offsets[:num_verified]
        ]
        return expected == actual
