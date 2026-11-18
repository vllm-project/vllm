# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parser for Beam checkpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from openai.types.responses import ToolChoiceFunction

from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.parser.abstract_parser import (
    _xgrammar_supports,
    structured_outputs_to_format,
)
from vllm.parser.chat_parsing import ResponseParser
from vllm.parser.chat_parsing.response_templates import ResponseTemplate
from vllm.parser.response_template import (
    CONTENT_FIELD,
    THINKING_FIELD,
    TOOL_FIELD,
    ResponseTemplateParser,
    ResponseTemplateReasoningParser,
    ResponseTemplateToolParser,
    _as_tool_dicts,
    _is_named_tool_choice,
    _tool_parsing_enabled,
)
from vllm.parser.response_template_grammar import (
    _openers,
    answer_rest_structural_tag,
    message_start_anchors,
    tool_call_structural_tag,
)
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.structural_tag_registry import resolve_tool_strictness

if TYPE_CHECKING:
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


def _named_tool(request: ChatCompletionRequest | ResponsesRequest) -> str:
    choice = request.tool_choice
    if isinstance(choice, ToolChoiceFunction):
        return choice.name
    return cast(ChatCompletionNamedToolChoiceParam, choice).function.name


def _starts_message(template: ResponseTemplate, prefix: str) -> bool:
    """Whether output after `prefix` starts the assistant message."""
    ends = [match.end() for match in template.start_anchor_re.finditer(prefix)]
    return bool(ends) and ends[-1] == len(prefix)


def _open_field(template: ResponseTemplate, prefix: str) -> str | None:
    """The field that `prefix` leaves open with its explicit opener, if any."""
    field = None
    for event in ResponseParser(template, prefix=prefix).initial_events:
        if event["type"] == "region_open" and event["end"] > event["start"]:
            field = event["field"]
        elif event["type"] in {"region_close", "region_malformed"}:
            field = None
    return field


class BeamReasoningParser(ResponseTemplateReasoningParser):
    """Selects `BeamParser` as the reasoning parser."""

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        return BeamParser

    @property
    def reasoning_start_str(self) -> str | None:
        if literal := super().reasoning_start_str:
            return literal
        thinking = self.response_template.fields.get(THINKING_FIELD)
        if thinking is None:
            return None
        openers = _openers(thinking, message_start_anchors(self.response_template))
        if not openers:
            return None
        # Budget tracking needs one marker for both a prompt continuation
        # and a complete assistant message in the checkpoint's pattern.
        marker = min(openers, key=len)
        return marker if all(opener.endswith(marker) for opener in openers) else None


class BeamToolParser(ResponseTemplateToolParser):
    """Selects `BeamParser` as the tool parser."""

    @classmethod
    def unified_parser_cls(cls) -> type[ResponseTemplateParser]:
        return BeamParser


class BeamParser(ResponseTemplateParser):
    """Parse Beam output with the checkpoint's `response_template` metadata,
    and constrain it with a grammar derived from that template."""

    reasoning_parser_cls = BeamReasoningParser
    tool_parser_cls = BeamToolParser

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or {}
        # From the message start, the grammar opens the message with non-empty
        # reasoning, unless the request turns thinking off.
        self._requires_reasoning = bool(
            chat_template_kwargs.get("enable_thinking", True)
        )

    def _constrain_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        message_start = _starts_message(self.response_template, self._prefix)
        tag = self._template_tag(request, message_start=message_start)
        if tag is not None:
            grammar = StructuredOutputsParams(structural_tag=json.dumps(tag))
            if isinstance(request, ChatCompletionRequest):
                request.structured_outputs = grammar
                request.response_format = None
            else:
                request._parser_structured_outputs = grammar
            # From the message start, the tag also covers the reasoning.
            request._grammar_from_parser = message_start
        return request

    def _template_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        message_start: bool,
    ) -> dict[str, Any] | None:
        """The template grammar for `request`, or `None` to leave its output
        to the request's own constraint, if any."""
        fields = self.response_template.fields
        named = _is_named_tool_choice(request)
        tools_enabled = (
            self._tool_parser is not None
            and bool(request.tools)
            and _tool_parsing_enabled(
                request, enable_auto_tools=self._enable_auto_tools
            )
        )
        forced = tools_enabled and (request.tool_choice == "required" or named)
        # The prompt may still list the tools, so only the grammar keeps the
        # model from calling them.
        tools_forbidden = (
            bool(request.tools)
            and request.tool_choice == "none"
            and TOOL_FIELD in fields
        )
        constrained_tools = None
        if tools_enabled and (forced or envs.VLLM_ENFORCE_STRICT_TOOL_CALLING):
            assert self._tool_parser is not None
            strict_level = self.tool_strict_level
            if strict_level is None:
                strict_level = self._tool_parser.default_tool_strict_level
            constrained_tools = resolve_tool_strictness(
                request.tools or [], request.tool_choice, strict_level
            )
        output = request.extract_structured_outputs()
        if forced and output is not None:
            logger.warning_once(
                "structured outputs are ignored because tool_choice forces tool call."
            )
            output = None
        answer = fields.get(CONTENT_FIELD)
        if output is not None and (
            output.structural_tag is not None
            or answer is None
            or answer.open_re is None
            or not _xgrammar_supports(output)
        ):
            if constrained_tools is not None:
                logger.warning_once(
                    "response_template parsing cannot enforce strict tools "
                    "alongside this structured output."
                )
            return None
        # Without a tool or structured output constraint, the output still
        # follows the template.
        always_constrain = envs.VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT and not forced
        if (
            constrained_tools is None
            and output is None
            and not tools_forbidden
            and not always_constrain
        ):
            return None

        # Inside an open answer, only the always-on grammar applies, finishing a
        # free answer. Inside open reasoning, the grammar needs the reasoning
        # gate to defer it to the reasoning's end.
        open_field = _open_field(self.response_template, self._prefix)
        reasoning_gated = (
            self._reasoning_parser is not None and request.include_reasoning
        )
        tag = None
        if open_field == CONTENT_FIELD:
            if always_constrain and output is None:
                tag = answer_rest_structural_tag(fields[CONTENT_FIELD])
        elif open_field != THINKING_FIELD or reasoning_gated:
            functions = []
            if tools_enabled:
                tools = _as_tool_dicts(constrained_tools or request.tools)
                functions = [tool.get("function", tool) for tool in tools]
            if named:
                name = _named_tool(request)
                functions = [f for f in functions if f.get("name") == name]
            answer_format = (
                None if output is None else structured_outputs_to_format(output)
            )
            try:
                tag = tool_call_structural_tag(
                    fields.get(TOOL_FIELD),
                    functions,
                    parallel_tool_calls=(
                        not named and request.parallel_tool_calls is not False
                    ),
                    thinking=fields.get(THINKING_FIELD),
                    answer=None if forced else answer,
                    answer_format=(
                        None if answer_format is None else answer_format.model_dump()
                    ),
                    message_start=(
                        message_start_anchors(self.response_template)
                        if message_start
                        else None
                    ),
                    requires_reasoning=self._requires_reasoning,
                    # Beam drifts from schemas the grammar does not enforce,
                    # and also writes arguments sorted by name.
                    sorted_arguments=True,
                    strict_arguments=True,
                )
            except ValueError as exc:
                raise VLLMValidationError(str(exc), parameter="tools") from exc
        if tag is None and forced:
            kind = "named" if named else "required"
            raise VLLMValidationError(
                f"response_template parsing cannot guarantee {kind} tool choice "
                "for this request",
                parameter="tool_choice",
            )
        if tag is None and constrained_tools is not None:
            logger.warning_once(
                "response_template parsing cannot enforce strict tools for this "
                "request."
            )
        return tag
