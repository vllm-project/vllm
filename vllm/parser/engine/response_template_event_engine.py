# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map `chat_parsing` region events to vLLM parser-engine events."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.parser.chat_parsing import ResponseParser
from vllm.parser.chat_parsing.content_parsers import _apply_transform
from vllm.parser.chat_parsing.response_templates import ResponseTemplate
from vllm.parser.engine.events import EventType, SemanticEvent

THINKING_FIELD = "thinking"
CONTENT_FIELD = "content"
TOOL_FIELD = "tool_calls"

logger = init_logger(__name__)


class ResponseTemplateEventEngine:
    """Adapt generic response-template regions to semantic events."""

    def __init__(
        self,
        template: ResponseTemplate,
        *,
        tools: Sequence[dict[str, Any]],
        parse_reasoning: bool,
        parse_tools: bool,
        enable_auto_tools: bool,
    ) -> None:
        self.template = template
        self.prefix = ""
        self.tools = list(tools)
        self.parse_reasoning = parse_reasoning
        self.stream_tool_names = False
        self.skip_tool_parsing = False
        self.skip_reasoning_parsing = False
        self.tool_parsing_enabled = parse_tools and enable_auto_tools
        tool_field = template.fields.get(TOOL_FIELD)
        self._tool_has_closer = (
            tool_field is not None and tool_field.close_re is not None
        )
        self.reset()

    def _tool_events(
        self,
        index: int,
        *,
        name: str | None = None,
        arguments: str | None = None,
        end: bool = False,
    ) -> list[SemanticEvent]:
        events: list[SemanticEvent] = []
        if name is not None:
            events += [
                SemanticEvent(EventType.TOOL_CALL_START, tool_index=index),
                SemanticEvent(EventType.TOOL_NAME, name, index),
                # ParserEngine releases a stored name on the next argument chunk.
                SemanticEvent(EventType.ARG_VALUE_CHUNK, "", index),
            ]
        if arguments or (arguments is not None and name is None):
            events.append(SemanticEvent(EventType.ARG_VALUE_CHUNK, arguments, index))
        if end:
            events.append(SemanticEvent(EventType.TOOL_CALL_END, tool_index=index))
        return events

    @property
    def reasoning_token_count(self) -> int:
        return 0

    @property
    def incomplete_tool_call_indices(self) -> set[int]:
        return set(self._incomplete_tool_call_indices)

    def reset(self, initial_state: Any = None) -> None:
        del initial_state
        self.parser = ResponseParser(
            self.template,
            prefix=self.prefix,
            tools=self.tools,
        )
        self._prefix_end = len(self.parser.input_text)
        self.initial_events = list(self.parser.initial_events)
        self.next_tool_index = 0
        self.pending_tool_index: int | None = None
        self.pending_tool_name: str | None = None
        self.finalized = False
        self._incomplete_tool_call_indices: set[int] = set()

    def feed(
        self,
        text: str,
        token_ids: Sequence[int] = (),
    ) -> list[SemanticEvent]:
        del token_ids
        if self.finalized:
            return []
        return self._route(self.parser.feed(text))

    def finish(self) -> list[SemanticEvent]:
        if self.finalized:
            return []
        self.finalized = True
        if len(self.parser.input_text) == self._prefix_end:
            self.initial_events = []
            return []
        _, events = self.parser.finalize()
        return self._route(events)

    def _initial_tool_open(self) -> list[dict[str, Any]]:
        """A tool region left open by the prompt prefix, replayed once."""
        events, self.initial_events = self.initial_events, []
        for event in reversed(events):
            if event.get("field") == TOOL_FIELD and event["type"] != "region_chunk":
                return [event] if event["type"] == "region_open" else []
        return []

    def _generated_text(self, event: dict[str, Any]) -> str:
        if event["type"] == "region_chunk":
            return event["text"]
        start = max(event["start"], self._prefix_end)
        return self.parser.input_text[start : event["end"]]

    def _is_unusable_tool_end(self, event: dict[str, Any]) -> bool:
        """Whether a tool region ended malformed, or at end of stream without its
        closer because the call was cut off."""
        return event["type"] == "region_malformed" or (
            event["type"] == "region_close"
            and self._tool_has_closer
            and event["start"] == len(self.parser.input_text)
        )

    def _open_tool_name(self, captures: dict | None) -> str | None:
        """Name already fixed by the opener, before the region body exists."""
        field = self.template.fields.get(TOOL_FIELD)
        transform = None if field is None or field.transform_each else field.transform
        function = transform.get("function") if isinstance(transform, dict) else None
        if not isinstance(function, dict):
            return None
        try:
            name = _apply_transform(function.get("name"), captures or {})
        except (KeyError, ValueError):
            return None
        return name if isinstance(name, str) else None

    def _to_tool_calls(self, value: Any) -> list[tuple[str, str]]:
        values = value if isinstance(value, list) else [value]
        calls: list[tuple[str, str]] = []
        for item in values:
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                return []
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or arguments is None:
                return []
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append((name, arguments))
        return calls

    def _route(self, events: Sequence[dict[str, Any]]) -> list[SemanticEvent]:
        events = self._initial_tool_open() + list(events)
        unusable_opens: set[int] = set()
        open_index = None
        for index, event in enumerate(events):
            if event.get("field") == TOOL_FIELD:
                if event["type"] == "region_open":
                    open_index = index
                elif self._is_unusable_tool_end(event):
                    unusable_opens.add(open_index)
        output: list[SemanticEvent] = []
        for index, event in enumerate(events):
            field = event.get("field")
            if field == TOOL_FIELD:
                output.extend(
                    self._route_tool_event(event, suppress_open=index in unusable_opens)
                )
            elif event["type"] == "region_chunk" and field in (
                THINKING_FIELD,
                CONTENT_FIELD,
            ):
                reasoning = (
                    field == THINKING_FIELD
                    and self.parse_reasoning
                    and not self.skip_reasoning_parsing
                )
                event_type = (
                    EventType.REASONING_CHUNK if reasoning else EventType.TEXT_CHUNK
                )
                output.append(SemanticEvent(event_type, event["text"]))
        return output

    def _route_tool_event(
        self,
        event: dict[str, Any],
        *,
        suppress_open: bool,
    ) -> list[SemanticEvent]:
        """Emit a tool call only when it ends with its closer and parses.
        Otherwise drop it, leaving a streamed name incomplete.
        With tool parsing disabled, the region passes through as content."""
        if not self.tool_parsing_enabled or self.skip_tool_parsing:
            text = self._generated_text(event)
            return [SemanticEvent(EventType.TEXT_CHUNK, text)] if text else []
        event_type = event["type"]
        if event_type == "region_open":
            return self._open_tool(event, suppress=suppress_open)
        if event_type == "region_chunk":
            return []

        calls = []
        if not self._is_unusable_tool_end(event):
            calls = self._to_tool_calls(event.get("value"))
        if self.pending_tool_index is not None and (
            len(calls) != 1 or calls[0][0] != self.pending_tool_name
        ):
            calls = []
        if calls:
            output = self._complete_tools(calls)
        else:
            logger.warning("response_template: dropping malformed or cut-off tool call")
            if self.pending_tool_index is not None:
                self._incomplete_tool_call_indices.add(self.pending_tool_index)
            output = []
        self.pending_tool_index = None
        self.pending_tool_name = None
        return output

    def _open_tool(
        self,
        event: dict[str, Any],
        *,
        suppress: bool,
    ) -> list[SemanticEvent]:
        if suppress or not self.stream_tool_names:
            return []
        name = self._open_tool_name(event.get("captures"))
        if name is None:
            return []
        index = self.next_tool_index
        self.next_tool_index += 1
        self.pending_tool_index = index
        self.pending_tool_name = name
        return self._tool_events(index, name=name)

    def _complete_tools(
        self,
        calls: list[tuple[str, str]],
    ) -> list[SemanticEvent]:
        if self.pending_tool_index is not None:
            return self._tool_events(
                self.pending_tool_index, arguments=calls[0][1], end=True
            )
        output: list[SemanticEvent] = []
        for name, arguments in calls:
            index = self.next_tool_index
            self.next_tool_index += 1
            output.extend(
                self._tool_events(index, name=name, arguments=arguments, end=True)
            )
        return output
