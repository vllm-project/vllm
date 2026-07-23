# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Olmo 3 reasoning parser built on the streaming parser engine.

Olmo 3 wraps its reasoning trace in ``<think>``/``</think>`` written as
ordinary text: the markers are split across multiple vocabulary tokens by
the pre-tokenizer (e.g. ``Ġ</`` + ``think`` + ``>``), so terminals are
lexed from text rather than matched by token ID. Some Olmo 3 chat
templates hardcode ``<think>`` in the prompt, so the parser starts in
REASONING state and treats a leading ``<think>`` as optional.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"


@functools.cache
def olmo3_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="olmo3",
        initial_state=ParserState.REASONING,
        terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        },
        # No token_id_terminals: the markers are not single vocabulary
        # tokens, so they can only be lexed from text.
        transitions={
            # Absorb the optional leading <think>.
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # A </think> seen in CONTENT state has no transition, so the
            # engine passes it through as literal content text.
        },
        # Olmo 3 reasoning is reported verbatim, including trailing
        # whitespace before </think>.
        strip_trailing_reasoning_whitespace=False,
    )


class Olmo3Parser(ParserEngine):
    """Olmo 3 parser: plain-text ``<think>``/``</think>`` reasoning."""

    CONFIG_NAME = "olmo3"
    # </think> is split in 3 by the pre-tokenizer; the first split can be
    # tokenized with an optional leading space, so there are 2 possible
    # tokenizations.
    think_end_first_split: list[str] = [r"Ġ</", r"</"]
    think_end_rest_split: list[str] = [r"think", r">"]

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", olmo3_config())
        super().__init__(tokenizer, tools, **kwargs)
        vocab = self.vocab
        self.think_end_first_token_ids: list[int] = [
            vocab[token] for token in self.think_end_first_split
        ]
        self.think_end_rest_token_ids: list[int] = [
            vocab[token] for token in self.think_end_rest_split
        ]

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        rest_ids = self.think_end_rest_token_ids
        rest_len = len(rest_ids)
        for i in range(len(input_ids) - rest_len, -1, -1):
            if (
                list(input_ids[i + 1 : i + 1 + rest_len]) == rest_ids
                and input_ids[i] in self.think_end_first_token_ids
            ):
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # The multi-token </think> straddles the reasoning/content
        # boundary, so the ids of the delta that completes it are not
        # content; the streaming parse extracts the content instead.
        return []
