# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
import json
from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

BEGIN_TOOL_REQUESTS = "<|plamo:begin_tool_requests:plamo|>"
END_TOOL_REQUESTS = "<|plamo:end_tool_requests:plamo|>"
BEGIN_TOOL_REQUEST = "<|plamo:begin_tool_request:plamo|>"
END_TOOL_REQUEST = "<|plamo:end_tool_request:plamo|>"
BEGIN_TOOL_NAME = "<|plamo:begin_tool_name:plamo|>"
END_TOOL_NAME = "<|plamo:end_tool_name:plamo|>"
BEGIN_TOOL_ARGUMENTS = (
    "<|plamo:begin_tool_arguments:plamo|><|plamo:constrain|>json<|plamo:msg|>"
)
END_TOOL_ARGUMENTS = "<|plamo:end_tool_arguments:plamo|>"
BEGIN_THINK = "<|plamo:begin_think:plamo|>"
END_THINK = "<|plamo:end_think:plamo|>"
EOT = "<|plamo:tag|>"

# PLaMo3 control markers are split into multiple tokenizer tokens.  These
# atomic pieces must not be treated as unrelated special tokens by the
# parser-engine drop-token handling.
PLAMO_MARKER_TOKENS = frozenset(
    {
        "<|plamo:begin_",
        "<|plamo:end_",
        ":plamo|>",
    }
)


@functools.cache
def plamo3_config(thinking: bool = True) -> ParserEngineConfig:
    if thinking:
        reasoning_terminals = {
            "THINK_START": BEGIN_THINK,
            "THINK_END": END_THINK,
        }
        reasoning_transitions = {
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.REASONING, "TOOL_REQUESTS_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
        }
        initial_state = ParserState.REASONING
    else:
        reasoning_terminals = {}
        reasoning_transitions = {}
        initial_state = ParserState.CONTENT

    return ParserEngineConfig(
        name="plamo3",
        initial_state=initial_state,
        terminals={
            **reasoning_terminals,
            "TOOL_REQUESTS_START": BEGIN_TOOL_REQUESTS,
            "TOOL_REQUESTS_END": END_TOOL_REQUESTS,
            "TOOL_REQUEST_START": BEGIN_TOOL_REQUEST,
            "TOOL_REQUEST_END": END_TOOL_REQUEST,
            "TOOL_NAME_START": BEGIN_TOOL_NAME,
            "TOOL_NAME_END": END_TOOL_NAME,
            "TOOL_ARGS_START": BEGIN_TOOL_ARGUMENTS,
            "TOOL_ARGS_END": END_TOOL_ARGUMENTS,
            "EOT": EOT,
        },
        transitions={
            **reasoning_transitions,
            (ParserState.CONTENT, "TOOL_REQUESTS_START"): Transition(
                ParserState.TOOL_PREAMBLE,
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_REQUEST_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_NAME_START"): Transition(
                ParserState.TOOL_NAME,
            ),
            (ParserState.TOOL_NAME, "TOOL_NAME_END"): Transition(
                ParserState.TOOL_PREAMBLE,
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_ARGS_START"): Transition(
                ParserState.TOOL_ARGS,
            ),
            (ParserState.TOOL_ARGS, "TOOL_ARGS_END"): Transition(
                ParserState.TOOL_BETWEEN,
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_REQUEST_END"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_PREAMBLE, "TOOL_REQUESTS_END"): Transition(
                ParserState.CONTENT,
            ),
            # EOT is normally consumed by generation stop handling.  Keep it
            # in the grammar as well so text-only parsing and explicitly
            # supplied stop tokens cannot leak it into user-visible output.
            (ParserState.REASONING, "EOT"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "EOT"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_PREAMBLE, "EOT"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_NAME, "EOT"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_ARGS, "EOT"): Transition(ParserState.CONTENT),
            (ParserState.TOOL_BETWEEN, "EOT"): Transition(ParserState.CONTENT),
        },
        token_id_terminals={"EOT": EOT, **reasoning_terminals},
        drop_whitespace_only_content_before_tools=True,
        stream_arg_deltas=True,
        strip_content_whitespace_with_tools=False,
        strip_trailing_reasoning_whitespace=False,
        preserve_tokens=PLAMO_MARKER_TOKENS,
        tool_args_json=True,
    )


class Plamo3Parser(ParserEngine):
    """PLaMo3 reasoning and tool-call parser backed by ParserEngine."""

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        # Configure PLaMo parsing and cache its multi-token reasoning markers.
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)
        kwargs.setdefault(
            "parser_engine_config",
            plamo3_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

        self._reasoning_start_token_ids: list[int] = list(
            tokenizer.encode(BEGIN_THINK, add_special_tokens=False)
        )
        self._reasoning_end_token_ids = list(
            tokenizer.encode(END_THINK, add_special_tokens=False)
        )
        self._partial_think_end_markers = tuple(
            tokenizer.decode(
                self._reasoning_end_token_ids[:size],
                skip_special_tokens=False,
            )
            for size in range(1, len(self._reasoning_end_token_ids))
        )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        # Detect PLaMo reasoning boundaries that span multiple token IDs.
        if not self.thinking_enabled:
            return True

        for i in range(len(input_ids) - 1, -1, -1):
            if (
                input_ids[i : i + len(self._reasoning_end_token_ids)]
                == self._reasoning_end_token_ids
            ):
                return True
            if (
                input_ids[i : i + len(self._reasoning_start_token_ids)]
                == self._reasoning_start_token_ids
            ):
                return False
            if input_ids[i] == self.vocab.get(EOT):
                return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Extract content after PLaMo's multi-token reasoning end marker.
        if not self.thinking_enabled:
            return input_ids
        end_ids = self._reasoning_end_token_ids
        for i in range(len(input_ids) - len(end_ids), -1, -1):
            if input_ids[i : i + len(end_ids)] == end_ids:
                return input_ids[i + len(end_ids) :]
        return input_ids

    def _strip_partial_think_end(self, value: str | None) -> str | None:
        for marker in reversed(self._partial_think_end_markers):
            if value and value.endswith(marker):
                return value[: -len(marker)] or None
        return value

    def _strip_unfinished_marker(
        self, value: str | None, *, prefix: str = "<|plamo:"
    ) -> str | None:
        if not value:
            return value
        start = value.rfind(prefix)
        if start >= 0:
            suffix = value[start:]
            if any(
                marker != suffix and marker.startswith(suffix)
                for marker in self.parser_engine_config.terminals.values()
            ):
                return value[:start] or None
        return value or None

    def _coalesce_finished_tool_events(
        self, events: list[SemanticEvent]
    ) -> list[SemanticEvent]:
        combined = []
        for (kind, _), group in groupby(
            events, key=lambda event: (event.type, event.tool_index)
        ):
            chunks = list(group)
            if kind not in (EventType.TOOL_NAME, EventType.ARG_VALUE_CHUNK):
                combined.extend(chunks)
                continue
            if kind == EventType.TOOL_NAME and chunks[0].tool_index < 0:
                continue
            value = "".join(chunk.value for chunk in chunks)
            if kind == EventType.TOOL_NAME:
                value = self._strip_unfinished_marker(value) or ""
            if value:
                combined.append(replace(chunks[0], value=value))
        return combined

    def _events_to_delta(
        self,
        events: list[SemanticEvent],
        finished: bool = False,
    ) -> DeltaMessage | None:
        # Coalesce flushed tool fragments before stripping truncated markers.
        if not finished:
            return super()._events_to_delta(events)

        events = self._coalesce_finished_tool_events(events)
        delta = super()._events_to_delta(events, finished=True)
        if delta is None:
            return None

        delta.reasoning = self._strip_partial_think_end(delta.reasoning)
        if not self.skip_tool_parsing:
            delta.content = self._strip_unfinished_marker(delta.content)
        if delta.reasoning is None and delta.content is None and not delta.tool_calls:
            return None
        return delta

    def _handle_arg_chunk(
        self,
        event: SemanticEvent,
        deltas: list[DeltaToolCall],
    ) -> None:
        # Strip truncated marker suffixes and preserve the first argument delta.
        idx = event.tool_index
        slot = self._tool_slots[idx]
        if (marker_pos := event.value.rfind("<|plamo:")) >= 0:
            stripped = event.value[:marker_pos]
            try:
                json.loads(slot.args + stripped)
            except ValueError:
                pass
            else:
                event = replace(event, value=stripped)
        name_sent_before = slot.name_sent
        super()._handle_arg_chunk(event, deltas)
        if event.value and not name_sent_before and slot.name_sent:
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=event.value),
                )
            )

    def _extract_args_json(self, raw_args: str, func_name: str) -> str:
        # Return the raw arguments as JSON
        # since PLaMo3 generates function names and arguments separately.
        return raw_args.strip() or "{}"

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        # Strip a truncated multi-token reasoning marker from parsed output.
        if not self.thinking_enabled:
            return None, model_output
        reasoning, content = super().extract_reasoning(model_output, request)
        return self._strip_partial_think_end(reasoning), content
