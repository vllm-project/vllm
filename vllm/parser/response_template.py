# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified vLLM parser backed by checkpoint `response_template` metadata."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.generate.base.protocol import (
    DeltaMessage,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.parser.chat_parsing import ResponseParser
from vllm.parser.chat_parsing.response_templates import (
    ResponseTemplate,
    load_response_template,
)
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserEngineConfig, ParserState
from vllm.parser.engine.response_template_event_engine import (
    ResponseTemplateEventEngine,
)
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
    reasoning: bool,
    tools: bool,
) -> None:
    """Fail at startup if the tokenizer cannot serve the requested parsing."""
    template = resolve_response_template(tokenizer)
    if template is None:
        raise TypeError(
            "The response_template parser requires `response_template` metadata "
            "in the tokenizer configuration"
        )
    try:
        fields = validate_response_template_for_serving(template).fields
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid response_template metadata: {exc}") from exc
    if reasoning and THINKING_FIELD not in fields:
        raise TypeError(
            "--reasoning-parser response_template requires a response_template "
            "with a thinking field"
        )
    if tools and TOOL_FIELD not in fields:
        raise TypeError(
            "--tool-call-parser response_template requires a response_template "
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
        request.tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam)
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


def _response_template_engine_config(
    template: ResponseTemplate,
) -> ParserEngineConfig:
    terminals: dict[str, str] = {}
    thinking = template.fields.get(THINKING_FIELD)
    if thinking is not None and thinking.open_literals:
        terminals["THINK_START"] = thinking.open_literals[0]
    if thinking is not None and thinking.close_literals:
        terminals["THINK_END"] = thinking.close_literals[0]
    return ParserEngineConfig(
        name="response_template",
        terminals=terminals,
        initial_state=ParserState.CONTENT,
        strip_trailing_reasoning_whitespace=False,
        drop_whitespace_only_content_before_tools=False,
        strip_content_whitespace_with_tools=False,
        defer_content_after_tools=False,
    )


class ResponseTemplateParser(ParserEngine):
    """Parse reasoning, content, and tool calls with one response template."""

    always_adjust_request = True
    _parse_reasoning = True
    _parse_tools = True
    _enable_auto_tools = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *,
        response_template: dict[str, Any] | None = None,
        parse_reasoning: bool | None = None,
        parse_tools: bool | None = None,
        enable_auto_tools: bool | None = None,
        **kwargs,
    ) -> None:
        template = resolve_response_template(tokenizer, response_template)
        if template is None:
            raise ValueError(
                "The response_template parser requires `response_template` "
                "metadata in the tokenizer configuration"
            )
        validate_response_template_for_serving(template)
        self.response_template = _streaming_template(template)
        self._chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._parse_reasoning_enabled = (
            self._parse_reasoning if parse_reasoning is None else parse_reasoning
        )
        self._parse_tools_enabled = (
            self._parse_tools if parse_tools is None else parse_tools
        )
        self._auto_tools_enabled = (
            self._enable_auto_tools if enable_auto_tools is None else enable_auto_tools
        )
        event_engine = ResponseTemplateEventEngine(
            self.response_template,
            tools=_as_tool_dicts(tools),
            parse_reasoning=self._parse_reasoning_enabled,
            parse_tools=self._parse_tools_enabled,
            enable_auto_tools=self._auto_tools_enabled,
        )
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=_response_template_engine_config(
                self.response_template
            ),
            streaming_engine=event_engine,
            **kwargs,
        )
        self._reasoning_parser = (
            cast(Any, self)
            if self._parse_reasoning_enabled
            and THINKING_FIELD in self.response_template.fields
            else None
        )
        self._tool_parser = None
        tool_field = self.response_template.fields.get(TOOL_FIELD)
        close_literals = (tool_field.close_literals if tool_field else None) or []
        self._tool_closers = {
            token_id: literal
            for literal in close_literals
            if (token_id := self.vocab.get(literal)) is not None
        }
        self._prompt_initialized = False
        self._stream_reasoning: list[str] | None = None
        self._stream_content: list[str] | None = None
        self._active_auto_tools = self._auto_tools_enabled

    @property
    def _response_engine(self) -> ResponseTemplateEventEngine:
        return cast(ResponseTemplateEventEngine, self._engine)

    def prepare_structured_tag(self, original_tag, tool_server):
        return original_tag

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        prefix = _decode(self.model_tokenizer, prompt_token_ids)
        if self._prompt_initialized and prefix == self._response_engine.prefix:
            return
        self._prompt_initialized = True
        self._response_engine.prefix = prefix
        self._reset()
        self._stream_reasoning = None
        self._stream_content = None

    def adjust_initial_state_from_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> None:
        self.set_prompt_token_ids(prompt_token_ids)

    def _preprocess_feed(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
    ) -> tuple[str, Sequence[int]]:
        # A tool closer that stops generation is dropped from the text but kept in
        # the token ids; restore it so the call is not mistaken for a cut-off one.
        closer = (
            self._tool_closers.get(delta_token_ids[-1]) if delta_token_ids else None
        )
        if closer is not None and not delta_text.endswith(closer):
            delta_text += closer
        return delta_text, delta_token_ids

    def _check_skip_tool_parsing(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> None:
        self._suppress_tool_calls = False
        self._response_engine.tool_parsing_enabled = (
            self._parse_tools_enabled
            and _tool_parsing_enabled(
                request, enable_auto_tools=self._active_auto_tools
            )
        )

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        del func_name
        return args_json

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
        if (
            self._parse_tools_enabled
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
        if not self._parse_reasoning_enabled:
            return True
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
            thinking_disabled=not self._chat_template_kwargs.get(
                "enable_thinking", True
            ),
        )

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        # A finished stream has consumed the engine state, so return what it streamed.
        if self._stream_content is not None and self._response_engine.finalized:
            tool_call_info = self._build_extracted_result()
            tool_calls = [
                FunctionCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in tool_call_info.tool_calls
            ]
            return (
                "".join(self._stream_reasoning or ()) or None,
                "".join(self._stream_content) or None,
                tool_calls or None,
            )
        self._active_auto_tools = enable_auto_tools
        self._response_engine.stream_tool_names = False
        return super().parse(
            model_output,
            request,
            enable_auto_tools=enable_auto_tools,
            model_output_token_ids=model_output_token_ids,
        )

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        if not self._prompt_initialized and prompt_token_ids is not None:
            self.set_prompt_token_ids(prompt_token_ids)
        self._prompt_streaming_prepared = True
        self._active_auto_tools = self._auto_tools_enabled
        self._response_engine.stream_tool_names = True
        if self._stream_content is None:
            self._stream_reasoning = []
            self._stream_content = []

        delta = super().parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=None,
            finished=finished,
        )

        if delta is not None:
            if delta.reasoning:
                assert self._stream_reasoning is not None
                self._stream_reasoning.append(delta.reasoning)
            if delta.content:
                self._stream_content.append(delta.content)
        return delta

    @property
    def incomplete_tool_call_indices(self) -> set[int]:
        return self._response_engine.incomplete_tool_call_indices


ResponseTemplateReasoningParser, ResponseTemplateToolParser = make_adapters(
    ResponseTemplateParser
)
ResponseTemplateToolParser.supports_required_and_named = False
