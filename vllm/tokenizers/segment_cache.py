# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, per-renderer encoding cache for added-token-delimited text."""

from __future__ import annotations

import json
import math
import sys
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import regex as re

if TYPE_CHECKING:
    from vllm.tokenizers.protocol import TokenizerLike


class SegmentTokenizerCache:
    """Cache exact encodings for stateless ByteLevel BPE tokenizers.

    Added tokens delimit BPE merges. NFC is supported when added tokens are
    extracted before normalization. Other normalization, Metaspace,
    prefix-space insertion and whitespace-consuming added tokens are excluded.
    Text keys prevent digest collisions, and the byte budget includes keys
    and token values. An unsupported tokenizer or encoding option falls back
    to its regular call. The tokenizer must remain immutable while serving.
    """

    def __init__(self, tokenizer: TokenizerLike, capacity_gb: float) -> None:
        if not math.isfinite(capacity_gb) or capacity_gb < 0:
            raise ValueError("tokenizer_cache_gb must be finite and nonnegative")
        self._tokenizer = tokenizer
        self.capacity = int(capacity_gb * (1 << 30))
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[tuple[int, ...], int]] = OrderedDict()
        self.bytes_used = 0
        self.hits = 0
        self.lookups = 0
        self.enabled = False
        self.disabled_reason = "zero capacity"
        self._pattern: re.Pattern[str] | None = None
        self._special_tokens_are_noop = False
        if not self.capacity:
            return
        try:
            backend = tokenizer.backend_tokenizer  # type: ignore[attr-defined]
            state = json.loads(backend.to_str())
            if state["model"].get("dropout") is not None:
                self.disabled_reason = "stochastic BPE dropout is unsupported"
                return
            normalizer = state.get("normalizer")
            if state["model"]["type"] != "BPE" or normalizer not in (
                None,
                {"type": "NFC"},
            ):
                self.disabled_reason = "requires BPE with no normalizer or NFC"
                return
            pre = state.get("pre_tokenizer")
            parts = pre.get("pretokenizers", [pre]) if pre else []
            if not parts or not any(p["type"] == "ByteLevel" for p in parts):
                self.disabled_reason = "requires a stateless ByteLevel pre-tokenizer"
                return
            if any(
                p["type"] not in ("ByteLevel", "Split")
                or p.get("add_prefix_space", False)
                for p in parts
            ):
                self.disabled_reason = "unsupported pre-tokenizer state"
                return
            added = state.get("added_tokens", [])
            if normalizer and any(t.get("normalized", True) for t in added):
                self.disabled_reason = (
                    "NFC requires added tokens extracted before normalization"
                )
                return
            if not added or any(
                t.get("lstrip") or t.get("rstrip") or t.get("single_word")
                for t in added
            ):
                self.disabled_reason = (
                    "added tokens need context-independent boundaries"
                )
                return
            tokens = sorted(
                {t["content"] for t in added if t["content"]}, key=len, reverse=True
            )
            if not tokens:
                self.disabled_reason = "no added-token boundaries"
                return
            self._pattern = re.compile("|".join(re.escape(t) for t in tokens))
            a, b = tokens[0], tokens[-1]
            probes = [
                f"hello{a}world{b}end",
                f"{a}{b}{a}",
                f"x   {a}   y\n{b}\n",
                f"你好🌏{a}e\u0301👩‍💻{b}code",
                ("def f(x): return {'x': x}\n" * 40) + a + "tail",
            ]
            for text in probes:
                ids = [i for s in self._split(text) for i in self._raw_encode(s)]
                if ids != self._raw_encode(text):
                    self.disabled_reason = (
                        "segmented encoding differs from full encoding"
                    )
                    return
            self._special_tokens_are_noop = all(
                self._raw_encode(text)
                == list(tokenizer(text, add_special_tokens=True)["input_ids"])
                for text in probes
            )
            self.enabled = True
            self.disabled_reason = ""
        except (AttributeError, KeyError, TypeError, ValueError):
            self.disabled_reason = "tokenizer does not expose a supported fast backend"

    def _raw_encode(self, text: str) -> list[int]:
        return list(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def _split(self, text: str) -> list[str]:
        assert self._pattern is not None
        result, start = [], 0
        for match in self._pattern.finditer(text):
            result.append(text[start : match.end()])
            start = match.end()
        if start < len(text):
            result.append(text[start:])
        return result

    def encode(self, text: str, **kwargs: Any) -> list[int] | None:
        """Return cached ids, or None to preserve unsupported call semantics."""
        if not self.enabled or not isinstance(text, str):
            return None
        if set(kwargs) - {"add_special_tokens", "truncation", "max_length"}:
            return None
        if kwargs.get("add_special_tokens", True) and not self._special_tokens_are_noop:
            return None
        if kwargs.get("truncation") and kwargs.get("max_length") is None:
            return None
        result: list[int] = []
        for segment in self._split(text):
            with self._lock:
                entry = self._data.get(segment)
                self.lookups += 1
                if entry is not None:
                    self._data.move_to_end(segment)
                    self.hits += 1
            if entry is None:
                ids = tuple(self._raw_encode(segment))
                # Include the OrderedDict node, entry tuple and size integer.
                size = (
                    sys.getsizeof(segment)
                    + sys.getsizeof(ids)
                    + sum(sys.getsizeof(i) for i in ids)
                    + sys.getsizeof((ids, 0))
                    + sys.getsizeof(0)
                    + sys.getsizeof(OrderedDict([(segment, None)]))
                    - sys.getsizeof(OrderedDict())
                )
                with self._lock:
                    if size <= self.capacity and segment not in self._data:
                        while self.bytes_used + size > self.capacity:
                            _, (_, removed_size) = self._data.popitem(last=False)
                            self.bytes_used -= removed_size
                        self._data[segment] = (ids, size)
                        self.bytes_used += size
            else:
                ids = entry[0]
            result.extend(ids)
        maximum = kwargs.get("max_length")
        if kwargs.get("truncation") and maximum is not None and len(result) > maximum:
            return None
        return result
