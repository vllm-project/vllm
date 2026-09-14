# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the unified Cohere Command parser tests."""

from __future__ import annotations

import itertools

import regex as re

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser

_SPECIAL_TOKEN_RE = re.compile(r"(<\|[A-Z_]+\|>)")
REPLACEMENT_CHAR = "�"


class MockCohereTokenizer:
    """Byte-level stand-in for the Cohere tokenizer.

    Text round-trips through UTF-8 bytes so a multi-byte character can split
    across "tokens" (reproducing trailing U+FFFD buffering), while each
    ``<|...|>`` marker is a single id so token-id reasoning-end gating behaves
    as with the real tokenizer.
    """

    _SPECIAL_TOKEN_IDS = {
        tok: 256 + i
        for i, tok in enumerate(
            (
                "<|START_THINKING|>",
                "<|END_THINKING|>",
                "<|CHATBOT_TOKEN|>",
                "<|START_RESPONSE|>",
                "<|END_RESPONSE|>",
                "<|START_TEXT|>",
                "<|END_TEXT|>",
                "<|START_ACTION|>",
                "<|END_ACTION|>",
            )
        )
    }
    _ID_TO_SPECIAL_TOKEN = {v: k for k, v in _SPECIAL_TOKEN_IDS.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS.get(token, 0)

    def get_vocab(self) -> dict[str, int]:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for part in _SPECIAL_TOKEN_RE.split(text):
            if part in self._SPECIAL_TOKEN_IDS:
                ids.append(self._SPECIAL_TOKEN_IDS[part])
            else:
                ids.extend(part.encode("utf-8"))
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out: list[str] = []
        for special, run in itertools.groupby(
            ids, self._ID_TO_SPECIAL_TOKEN.__contains__
        ):
            if not special:
                out.append(bytes(run).decode("utf-8", errors="replace"))
            elif not skip_special_tokens:
                out.extend(self._ID_TO_SPECIAL_TOKEN[i] for i in run)
        return "".join(out)


def token_deltas(
    tokenizer: MockCohereTokenizer, text: str, chunk_size: int = 1
) -> list[tuple[str, list[int]]]:
    """Split ``text`` into ``(delta_text, delta_token_ids)`` steps of
    ``chunk_size`` tokens, buffering incomplete multi-byte sequences (trailing
    U+FFFD) into the next step as real streaming does."""
    ids = tokenizer.encode(text)
    deltas: list[tuple[str, list[int]]] = []
    prev = ""
    pending: list[int] = []
    for start in range(0, len(ids), chunk_size):
        end = start + chunk_size
        pending.extend(ids[start:end])
        current = tokenizer.decode(ids[:end])
        if current.endswith(REPLACEMENT_CHAR) and end < len(ids):
            continue
        deltas.append((current[len(prev) :], pending))
        prev, pending = current, []
    return deltas


def make_parser(
    tokenizer: MockCohereTokenizer,
    name: str,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
    model_config=None,
) -> Parser:
    """Resolve the unified Cohere parser through ``ParserManager`` by registry
    name (``cohere_command3`` / ``cohere_command4``)."""
    cls = ParserManager.get_parser(name, name, enable_auto_tools=True)
    assert cls is not None
    return cls(
        tokenizer,
        tools,
        model_config=model_config,
        chat_template_kwargs=chat_template_kwargs or {},
    )


def drive_parser(
    parser: Parser,
    request: ChatCompletionRequest,
    deltas: list[tuple[str, list[int]]],
) -> list[DeltaMessage]:
    """Feed ``deltas`` through ``parse_delta`` (flushing on the last one) and
    return the non-``None`` messages in order."""
    out: list[DeltaMessage] = []
    for i, (text, ids) in enumerate(deltas):
        delta = parser.parse_delta(text, ids, request, finished=i == len(deltas) - 1)
        if delta is not None:
            out.append(delta)
    return out


def stream_parser(parser, request, tokenizer, text: str, chunk_size: int = 1):
    """Stream ``text`` through ``parser`` in ``chunk_size``-token deltas."""
    return drive_parser(parser, request, token_deltas(tokenizer, text, chunk_size))
