# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import (
    ResponseFormatTextJSONSchemaConfig,
    ResponseTextConfig,
    ToolChoiceFunction,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import LlamaJsonParserToolAdapter
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.utils import get_json_schema_from_tools


def _empty_object_schema() -> dict[str, Any]:
    """Schema for a tool that declares no parameters.

    ``additionalProperties: False`` is load-bearing, not decoration.  Without
    it the schema forbids nothing, so the decoding grammar accepts the
    ``{"name": ..., "parameters": ...}`` envelope the chat template asks for
    as readily as the bare ``{}`` -- and the parser cannot tell which it was
    handed.  Forbidding extra keys makes ``{}`` the only valid output on
    every backend, which is also exactly what "no parameters" means.

    Built fresh per call: a shared literal is mutated in place downstream
    (``_get_tool_schema_defs`` pops ``$defs``, the guidance backend writes
    ``additionalProperties``), so one request's edits would otherwise reach
    every parameterless tool served afterwards.
    """
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _named_choice_tool_name(
    request: ChatCompletionRequest | ResponsesRequest,
) -> str | None:
    tool_choice = getattr(request, "tool_choice", None)
    if isinstance(tool_choice, ToolChoiceFunction):
        return tool_choice.name
    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.function.name
    return None


class Llama3JsonToolParser(LlamaJsonParserToolAdapter):  # type: ignore[valid-type, misc]
    """Llama 3.x/4 JSON tool parser backed by the declarative parser
    engine (see vllm/parser/llama_json.py).

    Used when --enable-auto-tool-choice --tool-call-parser llama3_json or
    llama4_json are set.
    """

    structural_tag_model = "llama"
    # Engine-based streaming feeds one delta at a time, while the generic
    # required/named helpers need the cumulative document -- so route those
    # tool choices through this parser instead, as the other engine-based
    # parsers do.  Guided decoding is unaffected: the tool schema is applied
    # from the request's tool_choice, independent of this flag.
    supports_required_and_named = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Give a parameterless tool an empty-object schema for named choice.

        ``get_json_schema_from_tools`` returns the selected function's
        ``parameters``, which is ``None`` for a tool that declares none. No
        schema is then applied, so with no llama structural tag either
        (``VLLM_ENFORCE_STRICT_TOOL_CALLING=0``) nothing constrains
        generation and the parser cannot tell bare parameters from a full
        envelope -- it required the envelope and returned no tool call at
        all. An empty object says exactly what "no parameters" says and
        keeps the named-choice path deterministic.
        """
        request = super().adjust_request(request)
        name = _named_choice_tool_name(request)
        if name is not None:
            for tool in request.tools or ():
                # Chat completions nest the definition under .function;
                # the Responses API carries name/parameters on the tool.
                function: Any = getattr(tool, "function", tool)
                if (
                    getattr(function, "name", None) == name
                    and getattr(function, "parameters", None) is None
                ):
                    function.parameters = _empty_object_schema()
        return self._install_tool_schema(request)

    def _install_tool_schema(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Install the tool-choice schema directly, bypassing the base
        class's ``supports_required_and_named`` gate.

        The shared ``ToolParser.adjust_request`` only installs a
        structured-outputs schema when ``supports_required_and_named`` is
        True, since parsers that *don't* support it are expected to read
        required/named choice from their own native format instead. This
        parser is a special case: it sets ``supports_required_and_named =
        False`` for a different reason (streaming gets one delta at a time,
        so the generic cumulative-document helpers for required/named
        choice don't apply), but it still needs the schema installed so
        guided decoding constrains generation to bare parameters. Inlined
        here rather than relying on the base method's gate.
        """
        if not request.tools:
            return request
        structured_outputs = getattr(request, "structured_outputs", None)
        if (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        ):
            return request
        json_schema_from_tool = get_json_schema_from_tools(
            tool_choice=request.tool_choice,
            tools=request.tools,
            parallel_tool_calls=request.parallel_tool_calls,
        )
        if json_schema_from_tool is not None:
            if isinstance(request, ChatCompletionRequest):
                request.structured_outputs = StructuredOutputsParams(
                    json=json_schema_from_tool  # type: ignore[call-arg]
                )
                request.response_format = None
            if isinstance(request, ResponsesRequest):
                request.text = ResponseTextConfig(
                    format=ResponseFormatTextJSONSchemaConfig(
                        type="json_schema",
                        name="tool_calling_response",
                        schema=json_schema_from_tool,
                        strict=True,
                    )
                )
        return request
