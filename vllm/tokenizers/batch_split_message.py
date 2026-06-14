# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-split-message tokenizer: lossless segmented batch encoding.

``encode(text)`` is split on a special-token delimiter (default: the
tokenizer's ``eos_token``), the resulting segments are encoded in a single
``encode_batch`` call -- which the underlying fast (Rust) tokenizer
parallelizes across segments -- and the segment ids are concatenated back with
the delimiter ids in between.

This is lossless **iff** the delimiter is a hard BPE boundary, i.e. a
registered special token. Special tokens never merge with adjacent text, so

    encode(a + DELIM + b) == encode(a) + encode(DELIM) + encode(b)

and the produced id sequence is byte-for-byte identical to a single
``encode(text)``. Because the token ids are unchanged, downstream prefix
KV-cache hashing is unaffected -- the win here is purely in the tokenization
stage, and it applies to every request independently (segments encode in
parallel; see also the optional segment cache below).

An optional segment-id LRU cache memoizes per-segment ids keyed by a content
digest, so repeated segments (shared system prompts, prior conversation turns,
repeated RAG documents) skip re-encoding. This reuse is at the tokenizer layer
and is independent of vLLM's KV prefix cache.

Note on ``add_special_tokens`` (completion vs chat). Per-segment encoding only
stays lossless when no special token is injected *per call*: otherwise a model
that adds a BOS/EOS on ``encode`` would get that BOS/EOS repeated on **every**
segment. vLLM passes ``add_special_tokens=False`` for chat (the chat template
already wrote the special tokens into the text) and ``True`` for completion. So
for a tokenizer that adds specials (``num_special_tokens_to_add() > 0``), the
``add_special_tokens=True`` (completion) path falls back to a single,
non-segmented ``encode`` -- still correct, just without the speedup. Tokenizers
that add nothing (e.g. Qwen/ChatML, where eos is a plain delimiter,
``num_special_tokens_to_add() == 0``) keep the segmented fast path in both
modes. Net effect: the chat path is always lossless and fast; the only
no-speedup case is completion on a BOS/EOS-adding model. Note also that
completion text rarely contains the delimiter at all, so it usually has nothing
to segment regardless.

Enable via ``--tokenizer-mode batch_split_message``. Configure through
``--tokenizer-mode-config`` (forwarded to :meth:`from_pretrained`)::

    --tokenizer-mode-config '{"split_delimiter": "<|im_end|>",
                         "segment_cache": true,
                         "cache_max_entries": 100000,
                         "cache_max_tokens": 8000000}'

All keys are optional; ``split_delimiter`` defaults to the tokenizer's
``eos_token`` and the cache is off by default, so the default behavior is a
lossless, parallel re-implementation of ``encode``.
"""

from __future__ import annotations

import array
import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from vllm.logger import init_logger
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tokenizers.protocol import TokenizerLike

logger = init_logger(__name__)

# Defaults for the optional segment cache. The cache is only built when
# ``segment_cache=True`` is passed; these bound it when it is.
_DEFAULT_CACHE_MAX_ENTRIES = 100_000
_DEFAULT_CACHE_MAX_TOKENS = 8_000_000


def _segment_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


class _SegmentIdsCache:
    """Thread-safe LRU cache of per-segment token ids.

    Bounded by both entry count and total cached tokens; the least-recently
    used segment is evicted when either bound is exceeded. Keyed by a content
    digest of the segment text.
    """

    def __init__(self, max_entries: int, max_total_tokens: int) -> None:
        self._max_entries = max_entries
        self._max_total_tokens = max_total_tokens
        self._data: OrderedDict[bytes, array.array] = OrderedDict()
        self._total_tokens = 0
        self._lock = threading.Lock()

    def get(self, key: bytes) -> array.array | None:
        with self._lock:
            ids = self._data.get(key)
            if ids is not None:
                self._data.move_to_end(key)
            return ids

    def put(self, key: bytes, ids: array.array) -> None:
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._total_tokens -= len(old)
            # A single segment larger than the whole budget is never cached.
            if len(ids) > self._max_total_tokens:
                return
            self._data[key] = ids
            self._total_tokens += len(ids)
            while (
                len(self._data) > self._max_entries
                or self._total_tokens > self._max_total_tokens
            ):
                _, evicted = self._data.popitem(last=False)
                self._total_tokens -= len(evicted)


def _encode_segments(
    tokenizer: PreTrainedTokenizerFast,
    items: list[str],
    cache: _SegmentIdsCache | None,
    add_special_tokens: bool,
) -> list[Any]:
    # backend_tokenizer is the underlying ``tokenizers.Tokenizer`` whose
    # ``encode_batch`` parallelizes across segments. Using ``self``'s backend
    # (rather than a captured one) keeps each thread-pool copy on its own Rust
    # tokenizer, so this stays thread-safe under HfRenderer's tokenizer pool.
    backend = tokenizer.backend_tokenizer
    if cache is None:
        encs = backend.encode_batch(items, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encs]

    digests = [_segment_digest(s) for s in items]
    seg_ids: list[Any] = [cache.get(d) for d in digests]
    miss = [i for i, ids in enumerate(seg_ids) if ids is None]
    if miss:
        encs = backend.encode_batch(
            [items[i] for i in miss], add_special_tokens=add_special_tokens
        )
        for j, enc in zip(miss, encs):
            ids = array.array("i", enc.ids)
            seg_ids[j] = ids
            cache.put(digests[j], ids)
    return seg_ids


def _warn_if_not_lossless(
    tokenizer: PreTrainedTokenizerFast, base_cls: type, delim: str
) -> None:
    """Sanity-check that segmented encoding matches a single ``encode``.

    Compares our overridden ``encode`` against the unmodified base-class
    ``encode`` on a couple of samples. A mismatch means the chosen delimiter is
    not a clean boundary for this tokenizer, so segmentation would not be
    lossless -- warn loudly rather than silently produce a different id
    sequence.

    Checks ``add_special_tokens=False``. That covers both segmented paths: the
    ``False`` path directly, and the ``True`` path when ``n_special == 0``
    (there ``add_special_tokens`` has no effect, so it encodes identically to
    ``False``). The only other case, ``True`` with ``n_special > 0``, falls back
    to a single ``encode`` and is lossless by construction, so it needs no check.
    """
    samples = [f"foo{delim}bar baz", "hello world"]
    for sample in samples:
        segmented = tokenizer.encode(sample, add_special_tokens=False)
        # unbound: bypass override
        baseline = base_cls.encode(  # type: ignore[attr-defined]
            tokenizer, sample, add_special_tokens=False
        )
        if list(segmented) != list(baseline):
            logger.warning(
                "batch_split_message: segmented encode differs from baseline "
                "for sample %r (%d vs %d ids). The delimiter %r may not be a "
                "lossless boundary for this tokenizer; results will diverge "
                "from a single encode().",
                sample,
                len(segmented),
                len(baseline),
                delim,
            )
            return


def make_batch_split_message_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    *,
    split_delimiter: str | None = None,
    segment_cache: bool = False,
    cache_max_entries: int = _DEFAULT_CACHE_MAX_ENTRIES,
    cache_max_tokens: int = _DEFAULT_CACHE_MAX_TOKENS,
) -> PreTrainedTokenizerFast:
    """Wrap ``tokenizer`` in-place so ``encode`` uses segmented batch encoding.

    Follows vLLM's ``get_cached_tokenizer`` pattern: a dynamic subclass is
    installed as ``tokenizer.__class__`` and only ``encode`` is overridden, so
    every other ``TokenizerLike`` method is inherited from the real tokenizer.
    The delimiter ids and segment cache live in this function's closure, so
    they are shared across the deep-copied tokenizer pool that
    ``HfRenderer`` builds for thread safety (the cache is not copied per pool
    entry).
    """
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "tokenizer_mode='batch_split_message' requires a fast tokenizer, "
            f"got {type(tokenizer).__name__}."
        )

    delim = split_delimiter if split_delimiter is not None else tokenizer.eos_token
    if not delim:
        raise ValueError(
            "batch_split_message: no split delimiter. The tokenizer has no "
            "eos_token; pass split_delimiter via --tokenizer-mode-config."
        )
    delim_ids = list(tokenizer.encode(delim, add_special_tokens=False))

    # Losslessness requires the delimiter to be a hard BPE boundary; a
    # registered special token always is.
    if delim not in set(tokenizer.all_special_tokens) and len(delim_ids) != 1:
        logger.warning(
            "batch_split_message: delimiter %r is not a registered special "
            "token and encodes to %d ids; segmented encoding may not match "
            "encode(text). Prefer a special token such as eos.",
            delim,
            len(delim_ids),
        )

    # add_special_tokens=True injects BOS/EOS on *every* segment for tokenizers
    # that add them, which breaks losslessness. Probe how many specials this
    # tokenizer adds so encode() can fall back to a single encode in that case.
    n_special = tokenizer.num_special_tokens_to_add()

    cache = (
        _SegmentIdsCache(cache_max_entries, cache_max_tokens) if segment_cache else None
    )

    base_cls = tokenizer.__class__

    class BatchSplitMessageTokenizerImpl(base_cls):  # type: ignore[valid-type, misc]
        def encode(  # type: ignore[override]
            self,
            text: str,
            truncation: bool | None = None,
            max_length: int | None = None,
            add_special_tokens: bool = True,
        ) -> list[int]:
            if not isinstance(text, str):
                return super().encode(
                    text,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )

            # Per-segment encoding repeats any per-call special tokens. If this
            # tokenizer adds BOS/EOS and the caller wants them (completion), fall
            # back to a single encode to stay lossless. No-op for the chat path
            # (add_special_tokens=False) and for Qwen/ChatML (n_special == 0).
            if add_special_tokens and n_special > 0:
                return super().encode(
                    text,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=True,
                )

            items = text.split(delim)
            ends_with_delim = text.endswith(delim)
            if items and items[-1] == "":
                items.pop()
            if not items:
                return []

            seg_ids = _encode_segments(self, items, cache, add_special_tokens)

            result: list[int] = []
            last = len(items) - 1
            for i, ids in enumerate(seg_ids):
                result.extend(ids)
                if i < last or ends_with_delim:
                    result.extend(delim_ids)

            if max_length is not None and truncation and len(result) > max_length:
                if self.truncation_side == "left":
                    result = result[-max_length:]
                else:
                    result = result[:max_length]
            return result

    BatchSplitMessageTokenizerImpl.__name__ = f"BatchSplitMessage{base_cls.__name__}"
    tokenizer.__class__ = BatchSplitMessageTokenizerImpl
    _warn_if_not_lossless(tokenizer, base_cls, delim)
    logger.info_once(
        "batch_split_message tokenizer active: delimiter=%r, segment_cache=%s, "
        "cache_max_entries=%d, cache_max_tokens=%d, specials_added_per_encode=%d",
        delim,
        segment_cache,
        cache_max_entries,
        cache_max_tokens,
        n_special,
    )
    return tokenizer


class BatchSplitMessageTokenizer(TokenizerLike):
    """Registry entry for ``tokenizer_mode='batch_split_message'``.

    ``from_pretrained`` loads a fast HF tokenizer and wraps it via
    :func:`make_batch_split_message_tokenizer`. Configuration is forwarded
    through ``--tokenizer-mode-config``.
    """

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> PreTrainedTokenizerFast:
        # Pop our config out of kwargs so the rest can flow to AutoTokenizer.
        split_delimiter = kwargs.pop("split_delimiter", None)
        segment_cache = bool(kwargs.pop("segment_cache", False))
        cache_max_entries = int(
            kwargs.pop("cache_max_entries", _DEFAULT_CACHE_MAX_ENTRIES)
        )
        cache_max_tokens = int(
            kwargs.pop("cache_max_tokens", _DEFAULT_CACHE_MAX_TOKENS)
        )

        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
            **kwargs,
        )
        tokenizer = get_cached_tokenizer(tokenizer)  # type: ignore[assignment]
        return make_batch_split_message_tokenizer(
            tokenizer,  # type: ignore[arg-type]
            split_delimiter=split_delimiter,
            segment_cache=segment_cache,
            cache_max_entries=cache_max_entries,
            cache_max_tokens=cache_max_tokens,
        )
