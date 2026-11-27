# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import json
from contextlib import suppress
from typing import TYPE_CHECKING

import regex as re
from xgrammar import StructuralTag
from xgrammar.structural_tag import AnyTextFormat

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.sampling_params import StructuredOutputsParams

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_PARAMETER_RE = re.compile(
    r"<｜GCML｜parameter\b(?P<attrs>[^>]*)>(?P<body>.*?)</｜GCML｜parameter>",
    re.DOTALL,
)
_ATTRIBUTE_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')


def _gcml_arg_converter(raw_args: str, partial: bool) -> str:
    args: dict[str, object] = {}
    for match in _PARAMETER_RE.finditer(raw_args):
        attribute_text = match.group("attrs")
        attrs = dict(_ATTRIBUTE_RE.findall(attribute_text))
        name = attrs.get("name")
        if not name:
            continue
        value = match.group("body")
        if attrs.get("string", "true").lower() != "true":
            with suppress(ValueError):
                value = json.loads(value)
        args[name] = value
    return json.dumps(args, ensure_ascii=False)


@functools.cache
def gigachat4_config(thinking: bool = True) -> ParserEngineConfig:
    terminals = {
        "THINK_START": "<think>",
        "THINK_END": "</think>",
        "TOOL_START": "<｜GCML｜tool_calls>",
        "TOOL_END": "</｜GCML｜tool_calls>",
        "INVOKE_PREFIX": '<｜GCML｜invoke name="',
        "INVOKE_NAME_END": '">',
        "INVOKE_END": "</｜GCML｜invoke>",
        "PARAM_START": "<｜GCML｜parameter",
        "PARAM_CLOSE": "</｜GCML｜parameter>",
    }
    return ParserEngineConfig(
        name="gigachat4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals=terminals,
        token_id_terminals={
            name: terminals[name] for name in ("THINK_START", "THINK_END")
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(ParserState.CONTENT),
            (ParserState.REASONING, "THINK_START"): Transition(ParserState.REASONING),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(ParserState.TOOL_PREAMBLE),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            # A tool call ends the turn; discard text after the tool block.
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_gcml_arg_converter,
        arg_structural_chars=frozenset(">"),
        stream_arg_deltas=False,
        tool_args_json=False,
        strip_content_whitespace_with_tools=False,
        turn_boundary_tokens=frozenset({"<|role_sep|>\n", "<|message_sep|>\n\n"}),
    )


class GigaChat4Parser(ParserEngine):
    always_adjust_request = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        reasoning = chat_kwargs.get("reasoning")
        thinking = True if reasoning is None else bool(reasoning)
        parser_config = gigachat4_config(thinking=thinking)
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=parser_config,
            **kwargs,
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if (
            request.tool_choice != "none"
            or request.extract_structured_outputs() is not None
        ):
            return request

        tag = StructuralTag(format=AnyTextFormat(excludes=["<｜GCML｜", "</｜GCML｜"]))
        params = StructuredOutputsParams(structural_tag=tag.model_dump_json())
        if isinstance(request, ChatCompletionRequest):
            request.structured_outputs = params
        else:
            request._parser_structured_outputs = params
        request._grammar_from_parser = True
        return request

    def _check_skip_tool_parsing(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> None:
        if (
            request.tool_choice == "none"
            and not request._grammar_from_parser
            and request.extract_structured_outputs() is not None
        ):
            # GCML inside an explicitly requested format is data, not a call.
            self.skip_tool_parsing = True
        else:
            super()._check_skip_tool_parsing(request)
