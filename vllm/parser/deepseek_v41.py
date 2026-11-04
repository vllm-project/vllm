# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Sequence
from dataclasses import replace

import regex as re

from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    _dsml_arg_value_extender,
    _dsml_open_value_extender,
    _param_patterns,
    _unwrap_wrapper_args,
    deepseek_v4_config,
)
from vllm.parser.engine.adapters import (
    ParserEngineReasoningAdapter,
    ParserEngineToolAdapter,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig
from vllm.parser.metrics import record_tool_parser_invocation

DSML_ATOM = "｜DSML｜"

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_RE, _PARTIAL_PARAM_RE = _param_patterns(DSML_PARAM_START, DSML_PARAM_CLOSE)
_PARAM_HEAD_RE = re.compile(
    re.escape(DSML_PARAM_START) + r'\s+name="([^"]+)"(?:\s+string="([^"]*)")?\s*>'
)


def _token_origin_boundaries(
    raw_args: str, spans: Iterable[tuple[int, int]]
) -> list[tuple[int, int, re.Match | None]]:
    """Parameter tags whose DSML atom came from the DSML token, in text order.

    Each entry is ``(start, end, head)``; ``head`` is ``None`` for a closer
    or an unfinished header, which must not extend the preceding value.
    """
    trusted = set(spans)
    boundaries = []
    for head in _PARAM_HEAD_RE.finditer(raw_args, partial=True):
        atom_start = head.start() + 1
        if (atom_start, atom_start + len(DSML_ATOM)) in trusted:
            boundaries.append(
                (head.start(), head.end(), None if head.partial else head)
            )
    for close in re.finditer(re.escape(DSML_PARAM_CLOSE), raw_args):
        atom_start = close.start() + 2
        if (atom_start, atom_start + len(DSML_ATOM)) in trusted:
            boundaries.append((close.start(), close.end(), None))
    boundaries.sort(key=lambda entry: entry[0])
    return boundaries


def _token_origin_args(
    raw_args: str, partial: bool, spans: Iterable[tuple[int, int]]
) -> str:
    """Parse only parameter boundaries carrying a genuine DSML token atom."""
    boundaries = _token_origin_boundaries(raw_args, spans)
    params = {}
    for index, (_, end, head) in enumerate(boundaries):
        if head is None:
            continue
        complete = index + 1 < len(boundaries)
        if not complete and not partial:
            continue
        value_end = boundaries[index + 1][0] if complete else len(raw_args)
        value = raw_args[end:value_end]
        name, is_str = head.group(1), head.group(2)
        if is_str == "true":
            params[name] = value
        else:
            try:
                params[name] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                if complete:
                    params[name] = value
    return json.dumps(params, ensure_ascii=False)


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    terminal_overrides = {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    return replace(
        config,
        name="deepseek_v41",
        turn_boundary_tokens=frozenset(
            {"<｜Assistant｜>", "<｜User｜>", "<｜System｜>"}
        ),
        terminals={**config.terminals, **terminal_overrides},
        token_id_terminals={
            **{
                key: terminal_overrides.get(key, value)
                for key, value in config.token_id_terminals.items()
            },
            "DSML_ATOM": DSML_ATOM,
        },
        composite_token_id_terminals=dict.fromkeys(terminal_overrides, "DSML_ATOM"),
        arg_converter=functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
            param_start=DSML_PARAM_START,
        ),
        arg_value_extender=functools.partial(
            _dsml_arg_value_extender,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
            param_start=DSML_PARAM_START,
            param_close=DSML_PARAM_CLOSE,
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        self._arg_value_extender = self._extend_open_value

    @property
    def _token_origin(self) -> bool:
        return self._engine._ever_had_token_ids and self._engine._has_composite_ids

    def _arg_slot(self, raw_args: str):
        return next(
            (slot for slot in reversed(self._tool_slots) if slot.args is raw_args), None
        )

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        if not self._token_origin:
            return super()._convert_args(raw_args, partial)
        slot = self._arg_slot(raw_args)
        spans = slot.arg_token_id_spans if slot is not None else ()
        result = _token_origin_args(raw_args, partial, spans or ())
        return _unwrap_wrapper_args(
            result, self._tools, slot.name if slot is not None else None
        )

    def _extend_open_value(self, raw_args: str):
        extender = self.parser_engine_config.arg_value_extender
        assert extender is not None
        if not self._token_origin:
            return extender(raw_args)
        # The text-based extender would trust parameter tags spelled in text.
        slot = self._arg_slot(raw_args)
        spans = slot.arg_token_id_spans if slot is not None else ()
        boundaries = _token_origin_boundaries(raw_args, spans or ())
        if not boundaries or (head := boundaries[-1][2]) is None:
            return None
        names = {entry[2].group(1) for entry in boundaries[:-1] if entry[2]}
        if head.group(2) != "true" or head.group(1) in names:
            return None
        return _dsml_open_value_extender(raw_args, DSML_PARAM_START, DSML_PARAM_CLOSE)

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self.set_prompt_token_ids(prompt_token_ids)
        self._reset()


class DeepSeekV41ServingParser(DelegatingParser):
    """Keep serving adapters and request constraints around one output engine."""

    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, **kwargs)
        reasoning_adapter, tool_adapter = self._reasoning_parser, self._tool_parser
        assert isinstance(reasoning_adapter, ParserEngineReasoningAdapter)
        assert isinstance(tool_adapter, ParserEngineToolAdapter)
        self._reasoning_adapter = reasoning_adapter
        self._unified_parser = reasoning_adapter._parser_engine
        self._unified_parser._tools = tools
        self._unified_parser._stream_state = self._stream_state
        tool_adapter._parser_engine = self._unified_parser
        self._skip_tools_override = False

    @property
    def skip_tool_parsing(self):
        return self._unified_parser.skip_tool_parsing

    @skip_tool_parsing.setter
    def skip_tool_parsing(self, value):
        self._skip_tools_override = value
        self._unified_parser.skip_tool_parsing = value

    def _prepare_tools(self, request, enable_auto_tools=True):
        self._unified_parser.skip_tool_parsing = (
            self._skip_tools_override
            or request.tool_choice == "none"
            or not request.tools
            or not enable_auto_tools
        )
        self._unified_parser._suppress_tool_calls = False

    def _sync_reasoning(self):
        self._reasoning_adapter._streaming_count_valid = True
        self._stream_state.reasoning_ended = self._unified_parser.reasoning_ended

    def set_prompt_token_ids(self, prompt_token_ids):
        self._unified_parser.set_prompt_token_ids(prompt_token_ids)
        self._reasoning_adapter._streaming_count_valid = False

    def _uses_json_tool_choice(self, request):
        return (
            self._tool_parser is not None
            and self._tool_parser.supports_required_and_named
            and request.tool_choice not in (None, "auto", "none")
        )

    def parse(
        self, model_output, request, enable_auto_tools=False, model_output_token_ids=()
    ):
        if self._uses_json_tool_choice(request):
            # Reasoning still needs token provenance before JSON tool dispatch.
            with self._reasoning_adapter._skip_tool_parsing():
                reasoning, content, _ = self._unified_parser.parse(
                    model_output, request, enable_auto_tools, model_output_token_ids
                )
            self._sync_reasoning()
            tool_calls, content = self._extract_tool_calls(
                content, request, enable_auto_tools
            )
            return reasoning, content, tool_calls
        self._prepare_tools(request, enable_auto_tools)
        is_tool_called: bool | Exception = False
        try:
            result = self._unified_parser.parse(
                model_output, request, enable_auto_tools, model_output_token_ids
            )
            is_tool_called = bool(result[2])
            self._sync_reasoning()
            return result
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called, is_streaming=False, request=request
            )

    def parse_delta(
        self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished
    ):
        if self._uses_json_tool_choice(request):
            # The required-choice JSON scanner needs accumulated output text.
            self._stream_state.engine_based = False
            return super().parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
        self._prepare_tools(request)
        is_tool_called: bool | Exception = False
        try:
            result = self._unified_parser.parse_delta(
                delta_text,
                delta_token_ids,
                request,
                prompt_token_ids,
                finished=finished,
            )
            is_tool_called = bool(result and result.tool_calls)
            self._sync_reasoning()
            return result
        except Exception as e:
            is_tool_called = e
            raise
        finally:
            record_tool_parser_invocation(
                is_tool_called=is_tool_called, is_streaming=True, request=request
            )
