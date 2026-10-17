# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified vLLM parser backed by checkpoint `response_template` metadata."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Sequence
from math import inf
from typing import TYPE_CHECKING, Any, cast

from openai.types.responses import ToolChoiceCustom, ToolChoiceFunction

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.parser.abstract_parser import Parser
from vllm.parser.chat_parsing import ResponseParser
from vllm.parser.chat_parsing.response_templates import (
    ResponseTemplate,
    load_response_template,
)
from vllm.parser.response_template_grammar import message_start_anchors
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import iter_response_function_tool_dicts

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

logger = init_logger(__name__)

THINKING_FIELD = "thinking"
CONTENT_FIELD = "content"
TOOL_FIELD = "tool_calls"
_SUPPORTED_FIELDS = frozenset({THINKING_FIELD, CONTENT_FIELD, TOOL_FIELD})


def _tool_dict(tool: Any) -> dict[str, Any]:
    if hasattr(tool, "model_dump"):
        tool = tool.model_dump(exclude_none=True)
    return tool if isinstance(tool, dict) else {}


def _as_tool_dicts(tools: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Responses function tools with flattened namespaces, plus chat tools."""
    tools = list(tools or ())
    result = iter_response_function_tool_dicts(tools)
    result.extend(
        tool_dict
        for tool_dict in map(_tool_dict, tools)
        if isinstance(tool_dict.get("function"), dict)
    )
    return result


def _tool_is_strict(tool: Any) -> bool:
    tool_dict = _tool_dict(tool)
    function = tool_dict.get("function")
    return bool(
        tool_dict.get("strict")
        or (isinstance(function, dict) and function.get("strict"))
    )


def _decode(tokenizer: TokenizerLike, token_ids: Sequence[int]) -> str:
    try:
        return cast(Any, tokenizer.decode)(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def resolve_response_template(
    tokenizer: Any | None,
    template: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return `template` if given, otherwise the tokenizer's metadata."""
    if template is not None or tokenizer is None:
        return template
    template = getattr(tokenizer, "response_template", None)
    if not isinstance(template, dict):
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        template = init_kwargs.get("response_template")
    return template if isinstance(template, dict) else None


def validate_response_template_for_serving(
    template: dict[str, Any],
) -> ResponseTemplate:
    """Validate the template and reject semantic fields serving cannot route."""
    loaded = load_response_template(template)
    unsupported = (set(loaded.fields) - _SUPPORTED_FIELDS) | (
        set(loaded.defaults) - {"role"}
    )
    if unsupported:
        raise ValueError(
            "response_template contains unsupported semantic fields: "
            f"{sorted(unsupported)}. Supported fields are: "
            f"{sorted(_SUPPORTED_FIELDS)}"
        )
    for name in (THINKING_FIELD, CONTENT_FIELD):
        field = loaded.fields.get(name)
        if field is None:
            continue
        if (
            field.content != "text"
            or field.transform is not None
            or field.join is not None
        ):
            raise ValueError(
                f"response_template field {name!r} uses semantics that cannot "
                "be streamed by the OpenAI serving adapter"
            )
    return loaded


def validate_tokenizer_response_template(
    tokenizer: Any,
    *,
    reasoning_parser: str | None,
    tool_parser: str | None,
) -> None:
    """Fail at startup if the tokenizer cannot serve the selected parsers,
    named as given on the command line."""
    template = resolve_response_template(tokenizer)
    if template is None:
        selected = " ".join(
            f"{flag} {name}"
            for flag, name in (
                ("--reasoning-parser", reasoning_parser),
                ("--tool-call-parser", tool_parser),
            )
            if name
        )
        raise TypeError(
            f"{selected} requires `response_template` metadata "
            "in the tokenizer configuration"
        )
    try:
        fields = validate_response_template_for_serving(template).fields
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid response_template metadata: {exc}") from exc
    if reasoning_parser and THINKING_FIELD not in fields:
        raise TypeError(
            f"--reasoning-parser {reasoning_parser} requires a response_template "
            "with a thinking field"
        )
    if tool_parser and TOOL_FIELD not in fields:
        raise TypeError(
            f"--tool-call-parser {tool_parser} requires a response_template "
            "with a tool_calls field"
        )


def _streaming_template(template: dict[str, Any]) -> ResponseTemplate:
    result = copy.deepcopy(template)
    fields = result["fields"]
    if CONTENT_FIELD not in fields and all(
        "open" in field or "open_pattern" in field for field in fields.values()
    ):
        fields[CONTENT_FIELD] = {}
    for field in fields.values():
        field["optional"] = True
    return load_response_template(result)


def _is_named_tool_choice(request: ChatCompletionRequest | ResponsesRequest) -> bool:
    return isinstance(
        request.tool_choice,
        (ToolChoiceFunction, ToolChoiceCustom, ChatCompletionNamedToolChoiceParam),
    )


def _tool_parsing_enabled(
    request: ChatCompletionRequest | ResponsesRequest,
    *,
    enable_auto_tools: bool,
) -> bool:
    if request.tool_choice == "none":
        return False
    if request.tool_choice == "required" or _is_named_tool_choice(request):
        return True
    return enable_auto_tools and request.tool_choice in (None, "auto")


def _unsupported_tool_guarantee(
    request: ChatCompletionRequest | ResponsesRequest,
) -> tuple[str, str] | None:
    """The first requested tool guarantee this parser cannot enforce, with the
    request parameter that asked for it."""
    if any(_tool_is_strict(tool) for tool in request.tools or ()):
        return "strict tools", "tools"
    if request.tool_choice == "required":
        return "required tool choice", "tool_choice"
    if _is_named_tool_choice(request):
        return "named tool choice", "tool_choice"
    if getattr(request, "parallel_tool_calls", True) is False:
        return "parallel_tool_calls=False", "parallel_tool_calls"
    return None


def _reasoning_has_ended(
    parser: ResponseParser,
    template: ResponseTemplate,
    *,
    thinking_disabled: bool = False,
) -> bool:
    thinking = template.fields.get(THINKING_FIELD)
    if thinking is None:
        return True
    if thinking.repeats:
        return False

    active_field: str | None = None
    thinking_closed = False
    for event in parser.initial_events:
        if event["type"] == "region_open":
            active_field = event["field"]
        elif event["type"] in {"region_close", "region_malformed"}:
            thinking_closed |= event["field"] == THINKING_FIELD
            active_field = None
    if thinking_closed:
        return True
    if active_field in (CONTENT_FIELD, TOOL_FIELD):
        return True

    # Resolve delimiters held at the prompt boundary. A NUL cannot occur in
    # the model wire format and any implicit chunk containing it is a probe,
    # not evidence that content generation has started.
    for event in parser.feed("\0"):
        if (
            event["type"] == "region_open"
            and event["field"] in (CONTENT_FIELD, TOOL_FIELD)
            and event["end"] > event["start"]
        ):
            return True
        if event["field"] == THINKING_FIELD:
            if event["type"] == "region_open":
                return False
            if event["type"] in {"region_close", "region_malformed"}:
                return True
    if active_field == THINKING_FIELD:
        return False
    return thinking_disabled


def _literal_ids(
    tokenizer: TokenizerLike, vocab: dict[str, int], literals: Sequence[str]
) -> list[list[int]] | None:
    """Token ids of each literal, or `None` when one encodes to nothing."""
    ids = [
        [vocab[literal]]
        if literal in vocab
        else tokenizer.encode(literal, add_special_tokens=False)
        for literal in literals
    ]
    return ids if all(ids) else None


def _reasoning_end_ids(
    tokenizer: TokenizerLike, vocab: dict[str, int], template: ResponseTemplate
) -> list[list[int]] | None:
    """Token ids of each thinking closer, or `None` when they are not all
    literal."""
    thinking = template.fields.get(THINKING_FIELD)
    if thinking is None or thinking.repeats or not thinking.close_literals:
        return None
    return _literal_ids(tokenizer, vocab, thinking.close_literals)


def _ends_within(token_ids: Sequence[int], end_ids: list[int], num_new: int) -> bool:
    """Whether `end_ids` completes within the last `num_new` of `token_ids`."""
    window = list(token_ids[max(0, len(token_ids) - num_new - len(end_ids) + 1) :])
    return any(
        window[index : index + len(end_ids)] == end_ids
        for index in range(len(window) - len(end_ids) + 1)
    )


def _serving_template(
    tokenizer: Any,
    template: dict[str, Any] | None,
) -> ResponseTemplate:
    """The validated streaming form of `template` or the tokenizer's metadata."""
    template = resolve_response_template(tokenizer, template)
    if template is None:
        raise ValueError(
            "Parsing with a response template requires `response_template` "
            "metadata in the tokenizer configuration"
        )
    validate_response_template_for_serving(template)
    return _streaming_template(template)


def _to_tool_calls(value: Any) -> list[tuple[str, str]]:
    """`(name, arguments)` of each call in a tool region value, or nothing if
    any call is incomplete."""
    values = value if isinstance(value, list) else [value]
    calls: list[tuple[str, str]] = []
    for item in values:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            return []
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name or arguments is None:
            return []
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append((name, arguments))
    return calls


class ResponseTemplateReasoningParser(ReasoningParser):
    """Reasoning end detection for a response-template parser.

    `ResponseTemplateParser` parses the output; this parser tells it and the
    structured-output engine where reasoning ends.
    """

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        """The unified parser that selecting this reasoning parser stands for."""
        return ResponseTemplateParser

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *args,
        response_template: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self.response_template = _serving_template(tokenizer, response_template)
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_disabled = not chat_template_kwargs.get("enable_thinking", True)
        self._reasoning_end_ids = _reasoning_end_ids(
            tokenizer, self.vocab, self.response_template
        )

    @property
    def reasoning_start_str(self) -> str | None:
        thinking = self.response_template.fields.get(THINKING_FIELD)
        literals = thinking.open_literals if thinking else None
        return literals[0] if literals else None

    @property
    def reasoning_end_str(self) -> str | None:
        thinking = self.response_template.fields.get(THINKING_FIELD)
        literals = thinking.close_literals if thinking else None
        return literals[0] if literals else None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        try:
            parser = ResponseParser(
                self.response_template,
                prefix=_decode(self.model_tokenizer, input_ids),
            )
        except Exception:
            return False
        return _reasoning_has_ended(
            parser,
            self.response_template,
            thinking_disabled=self._thinking_disabled,
        )

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._reasoning_end_ids is None:
            return self.is_reasoning_end(input_ids)
        num_new = len(list(delta_ids))
        return any(
            _ends_within(input_ids, end_ids, num_new)
            for end_ids in self._reasoning_end_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        raise NotImplementedError("ResponseTemplateParser extracts content")

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError("ResponseTemplateParser extracts reasoning")

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        raise NotImplementedError("ResponseTemplateParser extracts reasoning")


class ResponseTemplateToolParser(ToolParser):
    """Selects a response-template parser as the tool parser;
    `ResponseTemplateParser` extracts the tool calls."""

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        """The unified parser that selecting this tool parser stands for."""
        return ResponseTemplateParser


def response_template_parser_cls(
    split_parser_cls: type[ReasoningParser] | type[ToolParser],
) -> type[ResponseTemplateParser] | None:
    """The response-template parser that a registered reasoning or tool parser
    stands for, or `None` for any other parser."""
    if issubclass(
        split_parser_cls,
        (ResponseTemplateReasoningParser, ResponseTemplateToolParser),
    ):
        return split_parser_cls.unified_parser_cls()
    return None


class _DeltaBuilder:
    """Accumulates one `DeltaMessage`."""

    def __init__(self) -> None:
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.tool_calls: list[DeltaToolCall] = []

    def build(self) -> DeltaMessage | None:
        fields: dict[str, Any] = {}
        if reasoning := "".join(self.reasoning):
            fields["reasoning"] = reasoning
        if content := "".join(self.content):
            fields["content"] = content
        if self.tool_calls:
            fields["tool_calls"] = self.tool_calls
        return DeltaMessage(**fields) if fields else None


class _ResponseStream:
    """One generation fed through `ResponseParser`, routed to message deltas."""

    def __init__(
        self,
        template: ResponseTemplate,
        *,
        prefix: str,
        tools: list[dict[str, Any]],
        parse_reasoning: bool,
        include_reasoning: bool,
        parse_tools: bool,
        new_tool_call_id: Callable[[str], str],
    ) -> None:
        self.parse_reasoning = parse_reasoning
        self.include_reasoning = include_reasoning
        self.parse_tools = parse_tools
        self.new_tool_call_id = new_tool_call_id
        self.template = template
        self.parser = ResponseParser(template, prefix=prefix, tools=tools)
        self._prefix_end = len(self.parser.input_text)
        self._next_tool_index = 0
        self._finalized = False
        self._open_region: dict[str, Any] | None = None
        self._track(self.parser.initial_events)

    def feed(self, text: str, *, finished: bool) -> DeltaMessage | None:
        if self._finalized:
            return None
        delta = _DeltaBuilder()
        self._route(self._track(self.parser.feed(text)), delta)
        if finished:
            self._finalized = True
            if len(self.parser.input_text) > self._prefix_end:
                events = self.parser.feed(self._closer_rest())
                self._route(events + self.parser.finalize()[1], delta)
        return delta.build()

    def _track(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for event in events:
            if event["type"] == "region_open":
                self._open_region = event
            elif event["type"] in ("region_close", "region_malformed"):
                self._open_region = None
        return events

    def _closer_rest(self) -> str:
        """The rest of a closer the output stopped in after ending its region,
        when only the next message's start anchor is missing."""
        if self._open_region is None:
            return ""
        field = self.template.fields[self._open_region["field"]]
        text = self.parser.input_text[self._open_region["end"] :]
        for closer in field.close_literals or ():
            for anchor in message_start_anchors(self.template):
                if not anchor or not closer.endswith(anchor) or closer == anchor:
                    continue
                for cut in range(len(closer) - 1, len(closer) - len(anchor) - 1, -1):
                    if text.endswith(closer[:cut]):
                        return closer[cut:]
        return ""

    def _route(self, events: Sequence[dict[str, Any]], delta: _DeltaBuilder) -> None:
        for event in events:
            field = event.get("field")
            if field == TOOL_FIELD:
                self._route_tool_event(event, delta)
            elif event["type"] == "region_chunk":
                if field == THINKING_FIELD and self.parse_reasoning:
                    if self.include_reasoning:
                        delta.reasoning.append(event["text"])
                elif field in (THINKING_FIELD, CONTENT_FIELD):
                    delta.content.append(event["text"])

    def _route_tool_event(self, event: dict[str, Any], delta: _DeltaBuilder) -> None:
        """Emit the calls of a tool region once it ends and parses; a region cut
        off at the end of the stream is parsed as generated. With tool parsing
        disabled, the region passes through as content."""
        if not self.parse_tools:
            if event["type"] == "region_chunk":
                delta.content.append(event["text"])
            else:
                start = max(event["start"], self._prefix_end)
                delta.content.append(self.parser.input_text[start : event["end"]])
            return
        if event["type"] == "region_malformed":
            logger.warning("response_template: dropping a malformed tool call")
        elif event["type"] == "region_close":
            calls = _to_tool_calls(event.get("value"))
            if not calls:
                logger.warning("response_template: dropping an incomplete tool call")
            for name, arguments in calls:
                delta.tool_calls.append(
                    DeltaToolCall(
                        index=self._next_tool_index,
                        id=self.new_tool_call_id(name),
                        type="function",
                        function=DeltaFunctionCall(name=name, arguments=arguments),
                    )
                )
                self._next_tool_index += 1


class _ReasoningTokenCounter:
    """Counts the reasoning tokens of one growing generation, parsing each
    token once."""

    def __init__(
        self, template: ResponseTemplate, prefix: str, tokenizer: TokenizerLike
    ) -> None:
        self.tokenizer = tokenizer
        self.parser = ResponseParser(template, prefix=prefix)
        self.start = len(self.parser.input_text)
        self.token_ids: list[int] = []
        self.token_ends: list[int] = []
        # Input spans between the delimiters of thinking regions.
        self.spans: list[tuple[int, float]] = []
        self.open: int | None = None
        self._track(self.parser.initial_events)

    def continues(self, token_ids: Sequence[int]) -> bool:
        return list(token_ids[: len(self.token_ids)]) == self.token_ids

    def count(self, token_ids: Sequence[int]) -> int:
        new_ids = list(token_ids[len(self.token_ids) :])
        texts = [_decode(self.tokenizer, [token_id]) for token_id in new_ids]
        end = self.token_ends[-1] if self.token_ends else self.start
        for text in texts:
            end += len(text)
            self.token_ends.append(end)
        self.token_ids.extend(new_ids)
        self._track(self.parser.feed("".join(texts)))
        # Text held back as a possible closer counts until a closer arrives.
        spans = self.spans if self.open is None else [*self.spans, (self.open, inf)]
        count, start = 0, self.start
        for end in self.token_ends:
            count += any(begin < end and start < close for begin, close in spans)
            start = end
        return count

    def _track(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            if event["field"] != THINKING_FIELD or event["type"] == "region_chunk":
                continue
            if event["type"] == "region_open":
                self.open = event["end"]
            elif self.open is not None:
                self.spans.append((self.open, event["start"]))
                self.open = None


class ResponseTemplateParser(Parser):
    """Parse reasoning, content, and tool calls with one response template."""

    reasoning_parser_cls: type[ReasoningParser] | None = ResponseTemplateReasoningParser
    tool_parser_cls: type[ToolParser] | None = ResponseTemplateToolParser
    always_adjust_request = True
    _enable_auto_tools = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        response_template: dict[str, Any] | None = None,
        enable_auto_tools: bool | None = None,
        **kwargs,
    ) -> None:
        self.response_template = _serving_template(tokenizer, response_template)
        super().__init__(
            tokenizer, tools, response_template=response_template, **kwargs
        )
        if enable_auto_tools is not None:
            self._enable_auto_tools = enable_auto_tools
        self._tools = _as_tool_dicts(tools)
        self._prompt_token_ids: list[int] | None = None
        self._prefix = ""
        self._stream: _ResponseStream | None = None
        self._reasoning_counter: _ReasoningTokenCounter | None = None

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        prompt_token_ids = list(prompt_token_ids)
        if prompt_token_ids == self._prompt_token_ids:
            return
        self._prompt_token_ids = prompt_token_ids
        self._prefix = _decode(self.model_tokenizer, prompt_token_ids)
        self._stream = None
        self._reasoning_counter = None

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        if getattr(request, "chat_template", None) is not None:
            logger.warning_once(
                "A request chat template is set; response_template parsing still "
                "expects the checkpoint's output format."
            )
        request.skip_special_tokens = False
        if hasattr(request, "spaces_between_special_tokens"):
            request.spaces_between_special_tokens = False
        return self._constrain_request(request)

    def _constrain_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Constrain `request` to the tool guarantees it asks for, or reject
        those this parser cannot enforce."""
        if (
            self._tool_parser is not None
            and request.tools
            and request.tool_choice != "none"
            and (unsupported := _unsupported_tool_guarantee(request)) is not None
        ):
            guarantee, parameter = unsupported
            raise VLLMValidationError(
                f"response_template parsing cannot guarantee {guarantee} "
                "without a format-compatible structural grammar",
                parameter=parameter,
            )
        return request

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return (
            self._reasoning_parser is None
            or self._reasoning_parser.is_reasoning_end(input_ids)
        )

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count the generated tokens whose text the template parses as
        reasoning, continuing reasoning left open by the prompt."""
        if (
            self._reasoning_parser is None
            or THINKING_FIELD not in self.response_template.fields
        ):
            return 0
        counter = self._reasoning_counter
        if counter is None or not counter.continues(token_ids):
            counter = self._reasoning_counter = _ReasoningTokenCounter(
                self.response_template, self._prefix, self.model_tokenizer
            )
        return counter.count(token_ids)

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        self._initialize_history_tool_call_cnt(request)
        stream = self._new_stream(
            request,
            enable_auto_tools=enable_auto_tools,
            include_reasoning=True,
        )
        delta = stream.feed(model_output, finished=True)
        if delta is None:
            return None, None, None
        tool_calls = [
            FunctionCall(
                id=call.id,
                name=call.function.name or "",
                arguments=call.function.arguments or "{}",
            )
            for call in delta.tool_calls
            if call.function is not None
        ]
        return delta.reasoning, delta.content, tool_calls or None

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        self._initialize_history_tool_call_cnt(request)
        if self._prompt_token_ids is None and prompt_token_ids is not None:
            self.set_prompt_token_ids(prompt_token_ids)
        if self._stream is None:
            self._stream = self._new_stream(
                request,
                enable_auto_tools=self._enable_auto_tools,
                include_reasoning=request.include_reasoning,
            )
        return self._stream.feed(delta_text, finished=finished)

    def _new_stream(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        enable_auto_tools: bool,
        include_reasoning: bool,
    ) -> _ResponseStream:
        return _ResponseStream(
            self.response_template,
            prefix=self._prefix,
            tools=self._tools,
            parse_reasoning=self._reasoning_parser is not None,
            include_reasoning=include_reasoning,
            parse_tools=self._tool_parser is not None
            and _tool_parsing_enabled(request, enable_auto_tools=enable_auto_tools),
            new_tool_call_id=self._new_tool_call_id,
        )

    def _new_tool_call_id(self, name: str) -> str:
        state = self._stream_state
        tool_call_id = make_tool_call_id(
            id_type=state.tool_call_id_type,
            func_name=name,
            idx=state.history_tool_call_cnt,
        )
        state.history_tool_call_cnt += 1
        return tool_call_id
