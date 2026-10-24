# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-selected token reuse for the initially audited Qwen tokenizer."""

import hashlib
import importlib.metadata
import json
import sys
import sysconfig
import threading
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import regex as re

from vllm.logger import init_logger
from vllm.utils.session_prefix import (
    PrefixContext,
    SessionPrefixStore,
    lcp,
    token_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.renderers.params import TokenizeParams
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)
_QWEN_BACKEND = "934c73f70d9dbdc5c03916ec8045c27749cc2580d14447fc90d05696101af934"
_ENCODING_METHODS = (
    "__call__",
    "_encode_plus",
    "_convert_encoding",
    "_get_padding_truncation_strategies",
    "set_truncation_and_padding",
    "_switch_to_input_mode",
    "_eventual_warn_about_too_long_sequence",
)


def _marker_tables(
    tokens: list[dict],
) -> tuple[dict[str, int], set[str], set[str]]:
    """Return scanned markers, boundary markers and right-stripping boundaries.

    Non-normalized added tokens are extracted from the raw text by
    leftmost-longest matching before other processing, so the text scan uses
    all of them. A marker may end a reused prefix only if none of its
    occurrences can overlap another marker occurrence and it needs no word
    context; other markers are still scanned but never used as boundaries.
    """
    markers = {
        t["content"]: t["id"] for t in tokens if t["content"] and not t["normalized"]
    }
    prefixes: dict[str, set[str]] = {}
    suffixes: dict[str, set[str]] = {}
    for m in markers:
        for i in range(1, len(m)):
            prefixes.setdefault(m[:i], set()).add(m)
            suffixes.setdefault(m[i:], set()).add(m)
    overlapping = set()
    for m in markers:
        for i in range(1, len(m)):
            # A proper suffix of m starts another marker, or a proper prefix
            # of m ends another marker.
            overlapping |= prefixes.get(m[i:], set()) | suffixes.get(m[:i], set())
            if m[i:] in prefixes or m[:i] in suffixes:
                overlapping.add(m)
        for i in range(len(m)):
            for j in range(i + 1, len(m) + 1):
                if (i, j) != (0, len(m)) and m[i:j] in markers:
                    overlapping |= {m, m[i:j]}
    flags = {t["content"]: t for t in tokens if t["content"] in markers}
    boundaries = {
        m for m in markers if m not in overlapping and not flags[m]["single_word"]
    }
    rstrip = {m for m in boundaries if flags[m]["rstrip"]}
    return markers, boundaries, rstrip


def _alternation(literals: "Iterable[str]") -> str:
    # Longest first: the leftmost match is then the longest, as in tokenizers.
    return "|".join(re.escape(m) for m in sorted(literals, key=len, reverse=True))


@dataclass(frozen=True)
class TokenSnapshot:
    text: str
    ids: list[int]


class SessionTokenCache:
    def __init__(self, tokenizer: "TokenizerLike", capacity: int):
        self.store: SessionPrefixStore[TokenSnapshot] = SessionPrefixStore(capacity)
        self.stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()
        self.enabled = False
        try:
            from transformers import TokenizersBackend

            if (
                not isinstance(tokenizer, TokenizersBackend)
                or sysconfig.get_config_var("Py_GIL_DISABLED")
                or importlib.metadata.version("transformers") != "5.19.0"
                or importlib.metadata.version("tokenizers") != "0.23.3"
                # Backend JSON does not describe Python encoding overrides.
                or any(
                    name in vars(tokenizer)
                    or getattr(type(tokenizer), name, None)
                    is not getattr(TokenizersBackend, name, None)
                    for name in _ENCODING_METHODS
                )
                or tokenizer.truncation_side not in ("left", "right")
                or tokenizer.split_special_tokens
            ):
                raise ValueError("unsupported tokenizer implementation or runtime")
            graph = json.loads(tokenizer.backend_tokenizer.to_str())
            digest = hashlib.sha256(
                json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if digest != _QWEN_BACKEND:
                raise ValueError("tokenizer differs from the audited Qwen3 tokenizer")
            markers, boundaries, self.rstrip_markers = _marker_tables(
                graph["added_tokens"]
            )
            if not boundaries:
                raise ValueError("no added token can end a reused prefix")
            # token_bytes() uses the native helper.
            from vllm import _session_prefix  # noqa: F401

            self.marker_ids = markers
            self.marker_set = frozenset(markers.values())
            self.pattern = re.compile(_alternation(markers))
            # Reverse search finds the last boundary marker before a position.
            self.boundary_pattern = re.compile("(?r)" + _alternation(boundaries))
            self.enabled = True
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Session token cache disabled: %s", exc)

    def context(self, session: str | None, salt: str | None) -> PrefixContext | None:
        if not self.enabled or not session or self.store.closed:
            return None
        return PrefixContext((salt, session), self.store.epoch)

    def _count(self, route: str, tokens: int = 0) -> None:
        with self._stats_lock:
            self.stats[route] += 1
            self.stats["reused_tokens"] += tokens
        logger.debug("Session token reuse: route=%s reused_tokens=%d", route, tokens)

    def _boundary(
        self, old: TokenSnapshot, new: str, common: int
    ) -> tuple[int, int] | None:
        """Return the last boundary marker end within ``common`` and its token end.

        Text and token IDs are aligned at both ends, so only the markers between
        the boundary and the nearer end are matched against the token IDs.
        """
        text, ids, end = old.text, old.ids, common
        while match := self.boundary_pattern.search(text, 0, end):
            # A right-stripping marker would absorb whitespace the new suffix
            # starts with, so the suffix could not be encoded separately.
            if (
                match.group() not in self.rstrip_markers
                or not new[match.end() : match.end() + 1].isspace()
            ):
                break
            end = match.start()
        else:
            return None
        boundary = match.end()
        if boundary > len(text) - boundary:
            expected = [
                self.marker_ids[m] for m in self.pattern.findall(text, boundary)
            ]
            expected.reverse()
            expected.append(self.marker_ids[match.group()])
            index = len(ids)
            for marker in expected:
                index -= 1
                while index >= 0 and ids[index] not in self.marker_set:
                    index -= 1
                if index < 0 or ids[index] != marker:
                    raise ValueError("source markers differ from token IDs")
        else:
            expected = [
                self.marker_ids[m] for m in self.pattern.findall(text, 0, boundary)
            ]
            index = -1
            for marker in expected:
                index += 1
                while index < len(ids) and ids[index] not in self.marker_set:
                    index += 1
                if index == len(ids) or ids[index] != marker:
                    raise ValueError("source markers differ from token IDs")
        return boundary, index + 1

    def encode(
        self, tokenizer: "TokenizerLike", text: str, params: "TokenizeParams"
    ) -> list[int] | None:
        context = params.session_prefix_context
        if (
            not self.enabled
            or type(text) is not str
            or context is None
            or not context.active
            or context.epoch != self.store.epoch
            or params.truncate_prompt_tokens is not None
            or params.pad_prompt_tokens is not None
            or params.truncation_side is not None
            or params.do_lower_case
            or params.add_special_tokens
            or params.return_token_offsets
        ):
            return None
        with self.store.borrow(context.key) as old:
            route = "miss"
            boundary = tokens = 0
            if old is not None:
                if text == old.text:
                    route = "exact"
                else:
                    try:
                        found = self._boundary(old, text, lcp(text, old.text))
                    except ValueError:
                        found = None
                        self._count("index_declined")
                    if found is not None:
                        boundary, tokens = found
                        route = "partial"
            if route == "exact":
                assert old is not None
                ids = old.ids.copy()
            elif route == "partial":
                assert old is not None
                ids = (
                    old.ids[:tokens]
                    + tokenizer(text[boundary:], **params.get_encode_kwargs())[
                        "input_ids"
                    ]
                )
            else:
                ids = tokenizer(text, **params.get_encode_kwargs())["input_ids"]
            maximum = params.max_input_tokens
            if maximum is not None and len(ids) > maximum:
                # Native encoding caps the reported length at maximum + 1.
                # Reuse its validation error without re-encoding or publishing.
                return params._token_len_check(tokenizer, ids[: maximum + 1])
            self._count(route, len(ids) if route == "exact" else tokens)
            # Reserve the snapshot size before the cache-only ID copy.
            size = (
                1024
                + sum(sys.getsizeof(part) for part in context.key)
                + sys.getsizeof(text)
                + token_bytes(ids)
            )
            reservation = self.store.reserve(context.key, size)
            if reservation is None:
                self._count("capacity_declined")
                return ids
            try:
                context.stage(reservation, TokenSnapshot(text, ids.copy()), size)
            except BaseException:
                reservation.cancel()
                raise
            return ids
