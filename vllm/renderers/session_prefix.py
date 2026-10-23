# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-selected token reuse for the initially audited Qwen tokenizer."""

import bisect
import hashlib
import importlib.metadata
import json
import sys
import sysconfig
import threading
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.session_prefix import PrefixContext, SessionPrefixStore, token_bytes

if TYPE_CHECKING:
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


@dataclass(frozen=True)
class TokenSnapshot:
    text: str
    ids: list[int]
    cuts: list[tuple[int, int]]


class SessionTokenCache:
    def __init__(self, tokenizer: "TokenizerLike", capacity: int):
        self.store: SessionPrefixStore[TokenSnapshot] = SessionPrefixStore(capacity)
        self.stats: Counter[str] = Counter()
        self._stats_lock = threading.Lock()
        self.enabled = False
        self.markers = None
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
                raise ValueError(
                    "tokenizer differs from the audited Qwen3-0.6B revision"
                )
            from vllm import _session_prefix

            self.helper = _session_prefix
            self.markers = self.helper.make_markers(
                {t["id"]: t["content"] for t in graph["added_tokens"]}
            )
            self.enabled = True
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
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
            boundary = tokens = at = 0
            route = "miss"
            if old is not None:
                if text == old.text:
                    route = "exact"
                else:
                    common = self.helper.lcp(text, old.text)
                    at = bisect.bisect_right(old.cuts, (common, sys.maxsize))
                    if at:
                        boundary, tokens = old.cuts[at - 1]
                        route = "partial"
            if route == "exact":
                assert old is not None
                ids = old.ids.copy()
                suffix_ids = None
            elif route == "partial":
                assert old is not None
                suffix_ids = tokenizer(text[boundary:], **params.get_encode_kwargs())[
                    "input_ids"
                ]
                ids = old.ids[:tokens] + suffix_ids
            else:
                ids = tokenizer(text, **params.get_encode_kwargs())["input_ids"]
                suffix_ids = None
            maximum = params.max_input_tokens
            if maximum is not None and len(ids) > maximum:
                # Native encoding caps the reported length at maximum + 1.
                # Reuse its validation error without re-encoding or publishing.
                return params._token_len_check(tokenizer, ids[: maximum + 1])
            self._count(route, len(ids) if route == "exact" else tokens)
            fixed = (
                1024
                + sum(sys.getsizeof(part) for part in context.key)
                + sys.getsizeof(text)
                + token_bytes(ids)
            )
            # Reserve before cache-only copies and boundary metadata. Include
            # transient tail/prefix lists and two Py_ssize_t-sized ints per cut.
            reserved = fixed + 256 * len(ids) + 128
            reservation = self.store.reserve(context.key, reserved)
            if reservation is not None:
                try:
                    if route == "exact":
                        assert old is not None
                        cuts = old.cuts.copy()
                    elif route == "partial":
                        assert old is not None
                        assert suffix_ids is not None
                        cuts = old.cuts[:at] + self.helper.boundaries(
                            self.markers, text[boundary:], suffix_ids, boundary, tokens
                        )
                    else:
                        cuts = self.helper.boundaries(self.markers, text, ids)
                    snapshot = TokenSnapshot(text, ids.copy(), cuts)
                    size = (
                        fixed
                        + sys.getsizeof(cuts)
                        + sum(
                            sys.getsizeof(pair) + sum(map(sys.getsizeof, pair))
                            for pair in cuts
                        )
                    )
                    context.stage(reservation, snapshot, size)
                except (ValueError, OverflowError):
                    reservation.cancel()
                    self._count("index_declined")
                except BaseException:
                    reservation.cancel()
                    raise
            else:
                self._count("capacity_declined")
            return ids
