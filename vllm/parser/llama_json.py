# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Llama 3.x/4 JSON tool-call parser (llama3_json / llama4_json).

Format: ``[<|python_tag|>]{"name": "f", "parameters": {...}}`` -- a bare JSON
envelope with no tool-name state and no end marker. A ``{`` opens a call; it
closes when the envelope's JSON balances (``tool_call_ends_on_args_balance``).

This is the initial, minimal version: name extraction uses the engine's
default (first ``"name"`` key found anywhere) and argument extraction reuses
the shared ``vllm/parser/utils.py`` scanner Granite already relies on. Later
commits add envelope-aware name resolution, multi-shape call handling,
truncation repair, schema-coercion safety, and resumable scanning.
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
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

PYTHON_TAG = "<|python_tag|>"
# Llama 4 wraps calls in these instead of <|python_tag|>; must be consumed,
# not left as content -- skip_special_tokens=False means the detokenizer
# no longer strips them, and drop machinery only covers all_special_tokens.
PYTHON_START = "<|python_start|>"
PYTHON_END = "<|python_end|>"
# Matched by position, not this order: the first alias in the text wins.
_ARG_KEYS = ("arguments", "parameters")


def _llama_arg_converter(raw_args: str, partial: bool) -> str:
    """Carve the parameters/arguments object out of the envelope.

    Same shared utility Granite uses (``extract_json_member_object``) --
    tries ``parameters`` then ``arguments``; whichever is present wins.
    No truncation repair yet: a span cut off mid-value is returned verbatim.
    """
    for key in _ARG_KEYS:
        try:
            span = extract_json_member_object(raw_args, key)
        except ValueError:
            continue
        if span is not None:
            return span if partial else (span or "{}")
    return "" if partial else "{}"


@functools.cache
def llama_json_config(
    markers: tuple[str, ...] = (PYTHON_TAG, PYTHON_START, PYTHON_END),
) -> ParserEngineConfig:
    """Config for the Llama JSON format.

    *markers* is the subset of tool markers present in the tokenizer's
    vocab; only those become **text** terminals -- a marker the model
    can't emit as a token can only have been written as prose, and
    consuming it there deletes the client's text.
    """
    return ParserEngineConfig(
        name="llama_json",
        terminals={
            **{
                name: text
                for name, text in (
                    ("PYTHON_TAG", PYTHON_TAG),
                    ("PYTHON_START", PYTHON_START),
                    ("PYTHON_END", PYTHON_END),
                )
                if text in markers
            },
            "OPEN_BRACE": "{",
        },
        token_id_terminals={
            "PYTHON_TAG": PYTHON_TAG,
            "PYTHON_START": PYTHON_START,
            "PYTHON_END": PYTHON_END,
        },
        transitions={
            (ParserState.CONTENT, "PYTHON_TAG"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "PYTHON_START"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "PYTHON_END"): Transition(ParserState.CONTENT, ()),
            (ParserState.CONTENT, "OPEN_BRACE"): Transition(
                ParserState.TOOL_ARGS,
                # ARG_VALUE_CHUNK re-injects the consumed "{" into the
                # slot so the accumulated envelope stays valid JSON.
                (EventType.TOOL_CALL_START, EventType.ARG_VALUE_CHUNK),
            ),
        },
        initial_state=ParserState.CONTENT,
        arg_converter=_llama_arg_converter,
        stream_arg_deltas=True,
        tool_args_json=True,
        tool_call_ends_on_args_balance=True,
        validate_tool_names=False,
    )


class LlamaJsonParser(ParserEngine):
    """Llama 3.x/4 JSON tool-call parser backed by the parser engine."""

    CONFIG_NAME = "llama_json"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        if "parser_engine_config" not in kwargs:
            vocab = kwargs.get("vocab")
            if vocab is None:
                try:
                    vocab = tokenizer.get_vocab()
                except (AttributeError, NotImplementedError):
                    vocab = {}
            present = tuple(
                m for m in (PYTHON_TAG, PYTHON_START, PYTHON_END) if m in vocab
            )
            kwargs["parser_engine_config"] = llama_json_config(present)
        super().__init__(tokenizer, tools, **kwargs)
