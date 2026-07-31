# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import LlamaJsonParserToolAdapter

_EMPTY_OBJECT_SCHEMA = {"type": "object", "properties": {}}


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
                    function.parameters = dict(_EMPTY_OBJECT_SCHEMA)
        return super().adjust_request(request)
