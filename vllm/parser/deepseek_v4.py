# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: `<think>`/`</think>`
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import contextlib
import functools
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import regex as re
from openai.types.responses import CustomTool

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine, ToolCallSlot
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import (
    custom_tools_with_grammar,
    find_tool_properties,
    partial_tag_overlap,
    validate_custom_tool_input,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool
_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"

# Spellings variants of ``DSML_TOOL_START`` observed in production.
DSML_TOOL_START_VARIANTS: tuple[str, ...] = (
    f"<{_DSML}toolcalls>",
    f"<{_DSML}tool>",
)


def _param_patterns(
    param_start: str, param_close: str
) -> tuple[re.Pattern, re.Pattern]:
    """Complete and trailing-partial parameter regexes for one DSML dialect.

    The `string` attribute is optional: the model sometimes omits it, and
    dropping such a parameter hands the client a tool call with no arguments.
    The trailing-partial regex captures the open value up to the end of the
    text; `_match_open_param` holds back an unfinished next parameter tag
    so a streamed string value never includes it.
    """
    start, close = re.escape(param_start), re.escape(param_close)
    head = rf'{start}\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
    return (
        re.compile(rf"{head}(.*?)(?:{close}|(?={start}\s+name=))", re.DOTALL),
        re.compile(rf"{head}(.*)\Z", re.DOTALL),
    )


_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)


def _match_open_param(
    raw_args: str, pos: int, partial_param_re: re.Pattern, param_start: str
) -> tuple[str, str | None, str, bool] | None:
    """Return (name, is_str, value, held_back) for the trailing open param.

    An unfinished next parameter tag (`param_start` with no `>` after it)
    is cut from the value; `held_back` reports whether that happened.
    """
    pm = partial_param_re.search(raw_args, pos)
    if pm is None:
        return None
    value = pm.group(3)
    cut = value.find(param_start, value.rfind(">") + 1)
    if cut >= 0:
        return pm.group(1), pm.group(2), value[:cut], True
    return pm.group(1), pm.group(2), value, False


def _dsml_arg_converter(
    raw_args: str,
    partial: bool,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    param_start: str = DSML_PARAM_START,
) -> str:
    params: dict[str, object] = {}

    last_end = 0
    # A missing ``string`` attribute is treated like ``string="false"``: try
    # JSON, fall back to the literal text.
    for m in param_re.finditer(raw_args):
        name, is_str, value = m.group(1), m.group(2), m.group(3)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                params[name] = value
        last_end = m.end()

    if partial:
        open_param = _match_open_param(
            raw_args, last_end, partial_param_re, param_start
        )
        if open_param:
            name, is_str, value, _ = open_param
            if is_str == "true":
                params[name] = value
            else:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    params[name] = json.loads(value)

    return json.dumps(params, ensure_ascii=False)


_JSON_VALUE_START = frozenset('{["-0123456789tfnNI')


def _cannot_start_json(value: str) -> bool:
    stripped = value.lstrip(" \t\r\n")
    return bool(stripped) and stripped[0] not in _JSON_VALUE_START


def _dsml_arg_value_extender(
    raw_args: str,
    *,
    param_re: re.Pattern = _PARAM_RE,
    partial_param_re: re.Pattern = _PARTIAL_PARAM_RE,
    param_start: str = DSML_PARAM_START,
    param_close: str = DSML_PARAM_CLOSE,
) -> ArgValueExtender | None:
    # ``arg_structural_chars`` skips every delta without ``>``, so a long
    # string value would wait for its closing tag. Extend the open trailing
    # value instead. A non-string value that cannot start JSON never parses,
    # so the converter already drops it until its tag closes: extend it with
    # no output rather than reconverting on every ``>``. The lexer can flush
    # a held partial tag as text, so keep a tail to catch a tag split across
    # deltas.
    names = set()
    last_end = 0
    for m in param_re.finditer(raw_args):
        names.add(m.group(1))
        last_end = m.end()
    open_param = _match_open_param(raw_args, last_end, partial_param_re, param_start)
    if open_param is None:
        return None
    name, is_str, value, held_back = open_param
    if name in names or held_back:
        return None
    streams = is_str == "true"
    if not streams and not _cannot_start_json(value):
        return None

    keep = max(len(param_start), len(param_close)) - 1
    tail = raw_args[-keep:]

    def extend(delta: str) -> str | None:
        nonlocal tail
        text = tail + delta
        if param_start in text or param_close in text:
            return None
        tail = text[-keep:]
        if not streams:
            return ""
        return json.dumps(delta, ensure_ascii=False)[1:-1]

    return extend


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": (DSML_TOOL_START, *DSML_TOOL_START_VARIANTS),
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            # A tool call ends the turn: stay in TOOL_BETWEEN so any text
            # after the block is dropped.
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        arg_value_extender=_dsml_arg_value_extender,
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    parser_config = staticmethod(deepseek_v4_config)

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=self.parser_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args
        self._custom_grammar_tools = custom_tools_with_grammar(tools)
        self._custom_grammar_chunks: list[str] = []
        self._custom_grammar_validate_input = False
        self._custom_grammar_probe = ""
        self._custom_grammar_request: (
            ChatCompletionRequest | ResponsesRequest | None
        ) = None

    def _tool_start_literals(self) -> tuple[str, ...]:
        starts = self.parser_engine_config.terminals["TOOL_START"]
        return (starts,) if isinstance(starts, str) else tuple(starts)

    def _custom_grammar_call_pattern(self) -> re.Pattern:
        starts = sorted(self._tool_start_literals(), key=len, reverse=True)
        invoke_prefix = self.parser_engine_config.terminal_literal("INVOKE_PREFIX")
        name_end = self.parser_engine_config.terminal_literal("INVOKE_NAME_END")
        assert invoke_prefix is not None and name_end is not None
        return re.compile(
            rf"(?:{'|'.join(re.escape(start) for start in starts)})\s*"
            rf"{re.escape(invoke_prefix)}(?P<name>[^\"\n]+)"
            rf"{re.escape(name_end)}\s*"
        )

    def _custom_grammar_envelope(self) -> tuple[str, str]:
        param_start = self.parser_engine_config.terminal_literal("PARAM_START")
        param_close = self.parser_engine_config.terminal_literal("PARAM_CLOSE")
        invoke_end = self.parser_engine_config.terminal_literal("INVOKE_END")
        tool_end = self.parser_engine_config.terminal_literal("TOOL_END")
        assert None not in (param_start, param_close, invoke_end, tool_end)
        return (
            f'{param_start} name="input" string="true">',
            f"{param_close}\n{invoke_end}\n{tool_end}",
        )

    def _split_custom_grammar_call(
        self, text: str
    ) -> tuple[re.Match, CustomTool, str, str | None] | None:
        match = self._custom_grammar_call_pattern().search(text)
        if match is None or match["name"] not in self._custom_grammar_tools:
            return None
        tool = self._custom_grammar_tools[match["name"]]
        body = text[match.end() :]
        param_begin, envelope_end = self._custom_grammar_envelope()
        if not body.startswith(param_begin) or not body.endswith(envelope_end):
            return match, tool, text[: match.start()], None
        payload = body[len(param_begin) : -len(envelope_end)]
        return match, tool, text[: match.start()], payload

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        converter = self.parser_engine_config.arg_converter
        assert converter is not None
        result = converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)

    def _check_skip_tool_parsing(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> None:
        super()._check_skip_tool_parsing(request)
        self._custom_grammar_validate_input = bool(
            getattr(request, "_custom_tool_grammar_enforced", False)
        )

    def _reset(self, initial_state: ParserState | None = None) -> None:
        super()._reset(initial_state=initial_state)
        self._custom_grammar_chunks.clear()
        self._custom_grammar_probe = ""

    def _single_pass_parse(
        self,
        text: str,
        token_ids: Sequence[int],
        initial_state: ParserState | None = None,
    ) -> tuple[str | None, str | None, ExtractedToolCallInformation]:
        if (
            not self._custom_grammar_tools
            or self.skip_tool_parsing
            or self._suppress_tool_calls
            or not self._custom_grammar_validate_input
        ):
            return super()._single_pass_parse(text, token_ids, initial_state)

        split = self._split_custom_grammar_call(text)
        if split is None:
            return super()._single_pass_parse(text, token_ids, initial_state)

        _, tool, prefix, payload = split
        self.skip_tool_parsing = True
        try:
            reasoning, content, plain = super()._single_pass_parse(
                prefix, [], initial_state
            )
        finally:
            self.skip_tool_parsing = False

        if payload is None:
            return (
                reasoning,
                content or plain.content,
                ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=content or plain.content,
                ),
            )
        if self._custom_grammar_validate_input:
            validate_custom_tool_input(tool, payload)

        slot = ToolCallSlot()
        slot.name = tool.name
        self._ensure_tool_id(slot, tool.name)
        call = ToolCall(
            id=slot.id,
            function=FunctionCall(
                name=tool.name,
                arguments=json.dumps({"input": payload}, ensure_ascii=False),
            ),
        )
        return (
            reasoning,
            content or plain.content,
            ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=[call],
                content=content or plain.content,
            ),
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ExtractedToolCallInformation:
        if not self._custom_grammar_tools or request.tool_choice == "none":
            return super().extract_tool_calls(model_output, request)
        self._check_skip_tool_parsing(request)
        return self._single_pass_parse(model_output, [])[2]

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> DeltaMessage | None:
        if not self._custom_grammar_tools or request.tool_choice == "none":
            return super().extract_tool_calls_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
                request,
            )
        self._custom_grammar_request = request
        self.initialize_streaming(initial_state=ParserState.CONTENT)
        self._check_skip_tool_parsing(request)
        if self._custom_grammar_chunks:
            self._custom_grammar_chunks.append(delta_text)
            return None

        combined = self._custom_grammar_probe + delta_text
        starts = self._tool_start_literals()
        marker_positions = [
            position for start in starts if (position := combined.find(start)) >= 0
        ]
        if marker_positions:
            marker = min(marker_positions)
            prefix = combined[:marker]
            self._custom_grammar_probe = ""
            self._custom_grammar_chunks.append(combined[marker:])
            if not prefix:
                return None
            return super().extract_tool_calls_streaming(
                previous_text,
                prefix,
                prefix,
                previous_token_ids,
                [],
                [],
                request,
            )

        overlap = max(
            (partial_tag_overlap(combined, start) for start in starts), default=0
        )
        safe = combined[:-overlap] if overlap else combined
        self._custom_grammar_probe = combined[-overlap:] if overlap else ""
        if not safe:
            return None
        return super().extract_tool_calls_streaming(
            previous_text,
            safe,
            safe,
            previous_token_ids,
            current_token_ids,
            [],
            request,
        )

    @staticmethod
    def _merge_deltas(*deltas: DeltaMessage | None) -> DeltaMessage | None:
        present = [delta for delta in deltas if delta is not None]
        if not present:
            return None
        reasoning = "".join(delta.reasoning or "" for delta in present) or None
        content = "".join(delta.content or "" for delta in present) or None
        tool_calls = [call for delta in present for call in (delta.tool_calls or [])]
        return DeltaMessage(
            reasoning=reasoning,
            content=content,
            tool_calls=tool_calls,
        )

    def finish_streaming(self) -> DeltaMessage | None:
        if not self._custom_grammar_tools:
            return super().finish_streaming()

        if self._custom_grammar_chunks:
            text = "".join(self._custom_grammar_chunks)
            self._custom_grammar_chunks.clear()
            reasoning, content, info = self._single_pass_parse(
                text, [], initial_state=ParserState.CONTENT
            )
            calls = [
                DeltaToolCall(
                    index=index,
                    id=call.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=call.function.name,
                        arguments=call.function.arguments,
                    ),
                )
                for index, call in enumerate(info.tool_calls)
            ]
            if not reasoning and not content and not calls:
                return None
            return DeltaMessage(
                reasoning=reasoning,
                content=content,
                tool_calls=calls or None,
            )

        if self._custom_grammar_probe:
            probe, self._custom_grammar_probe = self._custom_grammar_probe, ""
            request = self._custom_grammar_request
            assert request is not None
            prefix_delta = super().extract_tool_calls_streaming(
                "", probe, probe, [], [], [], request
            )
            return self._merge_deltas(prefix_delta, super().finish_streaming())
        return super().finish_streaming()
