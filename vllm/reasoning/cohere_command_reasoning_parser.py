# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, TypedDict, TypeGuard

import regex as re
import xgrammar as xgr

try:
    from cohere_melody import PyFilter, PyFilterOptions
except ImportError as e:
    raise ImportError(
        "The Cohere reasoning parser requires the `cohere_melody` "
        "package, which is not installed. Install it with:\n"
        "    pip install cohere_melody"
    ) from e


from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    AnyResponseFormat,
    Citation,
    CitationSource,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers import TokenizerLike

REPLACEMENT_CHAR = "\ufffd"


class CohereTagRegistry(NamedTuple):
    """A single ``structural_tag`` trigger / end pair (``begin`` uses ``trigger``)."""

    trigger: str
    end: str


class CohereTagStyle(NamedTuple):
    """The structural tags style for a given model architecture.

    ``json_tags`` lists every JSON-schema wrapper the model may emit (MOE uses
    both response and text delimiters). ``tools`` is the tool-call wrapper.
    """

    json_tags: tuple[CohereTagRegistry, ...]
    tools: CohereTagRegistry


class CohereNormalizedTool(TypedDict):
    """A tool definition normalized to the shape ``collect_tool_schema`` expects.

    ``parameters`` is a JSON Schema object (possibly empty) describing the tool's
    call signature.
    """

    name: str
    parameters: dict[str, Any]


COMMAND_A_TOOLS_TAG = CohereTagRegistry(
    trigger="<|START_ACTION|>",
    end="<|END_ACTION|>",
)
COMMAND_A_JSON_TAG = CohereTagRegistry(
    trigger="<|START_RESPONSE|>",
    end="<|END_RESPONSE|>",
)
COMMAND_A_PLUS_JSON_TAG = CohereTagRegistry(
    trigger="<|START_TEXT|>",
    end="<|END_TEXT|>",
)

MODEL_TO_TAG_STYLE: dict[str, CohereTagStyle] = {
    "Cohere2ForCausalLM": CohereTagStyle(
        json_tags=(COMMAND_A_JSON_TAG,),
        tools=COMMAND_A_TOOLS_TAG,
    ),
    "Cohere2VisionForConditionalGeneration": CohereTagStyle(
        json_tags=(COMMAND_A_JSON_TAG, COMMAND_A_PLUS_JSON_TAG),
        tools=COMMAND_A_TOOLS_TAG,
    ),
    "Cohere2MoeForCausalLM": CohereTagStyle(
        json_tags=(COMMAND_A_JSON_TAG, COMMAND_A_PLUS_JSON_TAG),
        tools=COMMAND_A_TOOLS_TAG,
    ),
}


def collect_tool_schema(tool_schema: list[CohereNormalizedTool]) -> str:
    """Build an xgrammar EBNF grammar that matches a JSON array of tool calls.

    The grammar shape is architecture-independent; callers are responsible for
    wrapping it in the correct structural tag (see ``CohereTagStyle.tools``).
    """
    tool_dictionary: dict[str, str] = {}
    for tool in tool_schema:
        tool_name = tool["name"]
        tool_parameters = json.dumps(tool["parameters"])
        json_schema = f"""{{
                        "type": "object",
                        "properties": {{
                            "tool_call_id": {{
                                "type": "string",
                                "pattern": "^[0-9]+$"
                            }},
                            "tool_name": {{
                                "type": "string",
                                "const": "{tool_name}"
                            }},
                            "parameters": {tool_parameters}
                            }}
                            }}"""
        tool_grammar = str(xgr.Grammar.from_json_schema(json_schema))
        for match in re.findall(r"\b(\w+)\s*::=", tool_grammar):
            tool_grammar = re.sub(
                rf"\b{re.escape(match)}\b", tool_name + match, tool_grammar
            )
        tool_dictionary[tool_name] = f"{tool_name} ::= {tool_name}root\n{tool_grammar}"
    # Emitted grammar shape:
    #   root  ::= tools
    #   tools ::= ws "[" ws tool ws ("," ws tool)* ws "]" ws
    #   ws    ::= (" " | "\t" | "\n")*
    #   tool  ::= <tool_a> | <tool_b> | ...         (one alternative per input)
    #   <tool_x>     ::= <tool_x>root               (per-tool xgrammar rules)
    #   <tool_x>root ::= ...                        (from xgr.Grammar.from_json_schema)
    tool_alternatives = "tool ::= " + " | ".join(tool_dictionary.keys())
    tool_rules = "\n    ".join(tool_dictionary.values())
    grammar = f"""root ::= tools
    tools ::= ws "[" ws tool ws ("," ws tool)*  ws "]" ws
    ws    ::= (" " | "\\t" | "\\n")*
    {tool_alternatives}
    {tool_rules}
    """
    return grammar


def _tool_definitions_to_schema_list(
    tools: str | list[Any],
) -> list[CohereNormalizedTool]:
    """
    Build the list of ``CohereNormalizedTool`` dicts expected by
    ``collect_tool_schema``.

    Accepts:
    - JSON string
    - list of dicts with top-level ``name`` / ``parameters``
    - list of Chat Completions-style ``{"type": "function", "function": {...}}``
    - list of Pydantic models with ``model_dump()``
    """
    if isinstance(tools, str):
        try:
            parsed = json.loads(tools)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
    else:
        parsed = list(tools)

    out: list[CohereNormalizedTool] = []
    for raw in parsed:
        t = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if not isinstance(t, dict):
            continue
        # Unwrap Chat Completions' ``{"type": "function", "function": {...}}``
        # shape; otherwise take the dict as-is.
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            t = t["function"]
        name = t.get("name")
        if not isinstance(name, str):
            continue
        params = t.get("parameters")
        out.append(
            CohereNormalizedTool(
                name=name,
                parameters=params if isinstance(params, dict) else {},
            )
        )
    return out


def _has_effective_tools(
    tools: str | list[Any] | None,
) -> TypeGuard[str | list[Any]]:
    """
    True when ``tools`` contains at least one tool definition to convert.

    ``ResponsesRequest`` defaults ``tools`` to ``[]``; ``ChatCompletionRequest``
    uses ``None``. Both mean "no tools" here. Strings (e.g. a JSON blob) are
    treated as effective only when non-blank.
    """
    if tools is None:
        return False
    if isinstance(tools, str):
        return bool(tools.strip())
    return len(tools) > 0


# Builder: produces vLLM response_format in xgrammar's canonical format.
# See xgrammar docs: type "structural_tag" with "format" = triggered_tags
# and tag content type = json_schema | grammar.
def convert_schema_to_structural_tags(
    schema: dict | None = None,
    tools: str | list[Any] | None = None,
    model_architecture: str | None = None,
) -> str | None:
    """
    Returns a response_format string accepted by xgrammar's structural tag format.
    Uses the canonical shape: {"type": "structural_tag", "format": {...}} with
    format.type "triggered_tags" and tag content type "json_schema" or "grammar".

    Callers that are not on an engine path (e.g. the reasoning parser) must pass
    ``model_architecture`` explicitly.
    """
    if model_architecture is None or model_architecture not in MODEL_TO_TAG_STYLE:
        return None
    style = MODEL_TO_TAG_STYLE[model_architecture]

    tags: list[dict] = []
    triggers: list[str] = []

    def _add_tag(tag: CohereTagRegistry, content: dict) -> None:
        tags.append({"begin": tag.trigger, "content": content, "end": tag.end})
        triggers.append(tag.trigger)

    if schema is not None:
        # One structural tag per JSON wrapper (e.g. MOE: response + text).
        # Same for schema-only and "tools plus JSON mode" (North: schema when
        # the model does not call tools).
        for jt in style.json_tags:
            _add_tag(jt, {"type": "json_schema", "json_schema": schema})

    if _has_effective_tools(tools):
        # ``tools`` may be a JSON string (poseidon / RESPONSE_FORMAT_TOOL_DEFINITIONS)
        # or a list (Chat Completions ``request.tools`` as Pydantic models or dicts).
        tool_schema_list = _tool_definitions_to_schema_list(tools)
        if not tool_schema_list:
            raise ValueError(
                "No valid tool definitions could be parsed from the request for "
                "structural tag conversion."
            )
        tool_grammar = collect_tool_schema(tool_schema_list)
        _add_tag(style.tools, {"type": "grammar", "grammar": tool_grammar})

    if not tags:
        return None
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
            },
        }
    )


def _response_format_type(
    response_format: AnyResponseFormat | dict | None,
) -> str | None:
    if response_format is None:
        return None
    if isinstance(response_format, dict):
        t = response_format.get("type")
        return t if isinstance(t, str) else None
    return response_format.type


def _maybe_parse_json_dict(value: Any) -> dict | None:
    """If value is a JSON string, parse to dict; otherwise require dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _unwrap_nested_schema(candidate: Any) -> dict | None:
    """Return ``candidate`` as a dict, unwrapping a nested ``schema`` if present.

    Returns ``None`` if ``candidate`` is not (and cannot be parsed into) a dict.
    """
    cand = _maybe_parse_json_dict(candidate)
    if not isinstance(cand, dict):
        return None
    nested = cand.get("schema")
    return nested if isinstance(nested, dict) else cand


def _schema_from_json_schema_field(js_wr: Any) -> dict | None:
    """
    Extract the JSON Schema object from Chat Completions ``json_schema`` payload.

    Accepts:
    - ``JsonSchemaResponseFormat`` (Pydantic) with ``schema`` / ``json_schema`` field
    - dict in OpenAI shape ``{"name": ..., "schema": {...}}``
    - dict with ``json_schema`` key holding either the schema or a nested wrapper
    - dict that is already a JSON Schema document (some clients omit the wrapper)
    - JSON strings for any of the above
    """
    if js_wr is None:
        return None

    parsed_wr = _maybe_parse_json_dict(js_wr)
    if parsed_wr is not None:
        js_wr = parsed_wr

    if hasattr(js_wr, "model_dump"):
        for by_alias in (True, False):
            try:
                data = js_wr.model_dump(by_alias=by_alias, exclude_none=False)
            except TypeError:
                data = js_wr.model_dump(by_alias=by_alias)
            out = _unwrap_nested_schema(data.get("schema") or data.get("json_schema"))
            if out is not None:
                return out
        inner_attr = getattr(js_wr, "json_schema", None)
        return inner_attr if isinstance(inner_attr, dict) else None

    if isinstance(js_wr, dict):
        for key in ("schema", "json_schema"):
            out = _unwrap_nested_schema(js_wr.get(key))
            if out is not None:
                return out
        return js_wr

    return None


def _schema_dict_from_chat_response_format(
    rf: AnyResponseFormat | dict | None,
) -> dict | None:
    """JSON schema dict from Chat Completions ``request.response_format`` only."""
    if rf is None:
        return None
    rf_type = _response_format_type(rf)
    if rf_type == "json_object":
        return {"type": "object"}
    if rf_type != "json_schema":
        return None
    js_wr = (
        rf.get("json_schema")
        if isinstance(rf, dict)
        else getattr(rf, "json_schema", None)
    )
    return _schema_from_json_schema_field(js_wr)


def _schema_dict_from_structured_outputs(
    so: StructuredOutputsParams | None,
) -> dict | None:
    """Schema dict from ``structured_outputs`` (``json`` / ``json_object``).

    Same unwrapping as ``json_schema``. ``json`` is expected to be ``str`` or
    ``dict`` (enforced by ``StructuredOutputsParams`` / request models); other
    types raise ``ValueError`` only if a caller bypasses that validation.
    """
    if so is None:
        return None
    if so.json_object:
        return {"type": "object"}
    raw: Any = so.json
    if raw is None:
        return None

    if hasattr(raw, "model_dump"):
        out = _schema_from_json_schema_field(raw)
        if out is None:
            raise ValueError(
                "structured_outputs.json model has no extractable JSON Schema."
            )
        return out

    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError("structured_outputs.json cannot be empty.")
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError("structured_outputs.json must be valid JSON.") from e
        if not isinstance(raw, dict):
            raise ValueError("structured_outputs.json must decode to a JSON object.")

    if isinstance(raw, Mapping):
        body = raw if isinstance(raw, dict) else dict(raw)
        return _schema_from_json_schema_field(body) or body

    raise ValueError(
        f"structured_outputs.json has unsupported type {type(raw).__name__}."
    )


def _melody_sources_to_vllm(raw_sources: Any) -> list[CitationSource]:
    """Convert melody's ``Source`` objects into :class:`CitationSource`.

    melody's ``Source`` shape is ``{tool_call_index, tool_result_indices,
    document_ids}``. ``document_ids`` may not be set; if it is empty and
    we have no resolvable identifier then we fall back to a generic
    ``tool``-style source carrying the tool-call index for visibility.
    """
    out: list[CitationSource] = []
    for s in raw_sources or []:
        # TODO Verify the tool vs doc logic
        doc_ids: list[str] = list(getattr(s, "document_ids", None) or [])
        if doc_ids:
            for did in doc_ids:
                if did:
                    out.append(CitationSource(type="document", id=did))
            continue
        tool_call_index = getattr(s, "tool_call_index", None)
        out.append(
            CitationSource(
                type="tool",
                id=(str(tool_call_index) if tool_call_index is not None else None),
            )
        )
    return out


def _melody_citations_to_vllm(raw_citations: Any) -> list[Citation] | None:
    """Convert melody's ``FilterCitation`` objects into :class:`Citation`."""
    if not raw_citations:
        return None
    out: list[Citation] = []
    for c in raw_citations:
        out.append(
            Citation(
                start=getattr(c, "start_index", None),
                end=getattr(c, "end_index", None),
                text=getattr(c, "text", None),
                sources=_melody_sources_to_vllm(getattr(c, "sources", None)),
                type=(
                    "THINKING_CONTENT"
                    if getattr(c, "is_thinking", False)
                    else "TEXT_CONTENT"
                ),
            )
        )
    return out


class BaseCohereCommandReasoningParser(ReasoningParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        *args,
        streaming_opts: PyFilterOptions,
        unary_opts: PyFilterOptions,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, **kwargs)
        self.start_token_id = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")
        self.chatbot_token_id = tokenizer.convert_tokens_to_ids("<|CHATBOT_TOKEN|>")
        self.unary_opts = unary_opts
        self.melody_unary = PyFilter(unary_opts)
        self.melody_streaming = PyFilter(streaming_opts)
        # Citations extracted by the most recent ``extract_reasoning`` call.
        # The non-streaming chat-completion path reads this back from the
        # parser instance (which is constructed per-request) and attaches
        # the result to ``ChatMessage.citations`` so grounded surfaces
        # (e.g. ``/cohere/v2/chat``) can surface them. ``None`` when the
        # last parse produced no citations.
        self.last_unary_citations: list[Citation] | None = None

    @property
    def reasoning_start_str(self) -> str | None:
        return "<|START_THINKING|>"

    @property
    def reasoning_end_str(self) -> str | None:
        return "<|END_THINKING|>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        r = self.melody_streaming.write_decoded(delta_text)
        citations = _melody_citations_to_vllm(getattr(r, "citations", None))
        if (
            r.content is None
            and r.reasoning is None
            and not r.tool_calls
            and not citations
        ):
            return None
        msg = DeltaMessage()
        if r.content is not None:
            msg.content = r.content
        if r.reasoning is not None:
            msg.reasoning = r.reasoning
        if r.tool_calls:
            msg.tool_calls = [
                DeltaToolCall(
                    id=tc.id,
                    index=tc.index,
                    type="function",
                    function=DeltaFunctionCall(name=tc.name, arguments=tc.arguments),
                )
                for tc in r.tool_calls
            ]
        if citations:
            msg.citations = citations
        return msg

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        result = self.melody_unary.process_full_text(model_output)
        # Cache citations so the non-streaming chat-completion path can
        # surface them on ``ChatMessage.citations`` (the ``parse`` return
        # tuple is locked to ``(reasoning, content, tool_calls)`` across
        # all parsers, so we ferry citations via parser-instance state --
        # safe because the parser is constructed per-request).
        self.last_unary_citations = _melody_citations_to_vllm(
            getattr(result, "citations", None)
        )
        return result.reasoning, result.content

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        token_buf: list[int] = []
        content_ids: list[int] = []
        content_filter = PyFilter(self.unary_opts)
        for t in input_ids:
            token_buf.append(t)
            s = self.model_tokenizer.decode(token_buf, skip_special_tokens=False)
            if s.endswith(REPLACEMENT_CHAR):
                continue
            r = content_filter.write_decoded(s)
            if r.content is not None:
                content_ids.extend(token_buf)
            token_buf = []
        return content_ids

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        chatbot = self.chatbot_token_id
        start = self.start_token_id
        end = self.end_token_id
        has_end_token = False

        for i in reversed(range(len(input_ids))):
            tid = input_ids[i]
            if tid == start:
                return has_end_token
            if tid == chatbot:
                return False
            if tid == end:
                has_end_token = True

        return has_end_token

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        so = request.structured_outputs
        if so is not None and so.structural_tag:
            return request
        # Schema: prefer ``response_format`` (OpenAI Chat Completions), then
        # ``structured_outputs.json`` / ``json_object`` (vLLM direct). Tools stay
        # on ``request.tools``.
        rf = (
            request.response_format
            if isinstance(request, ChatCompletionRequest)
            else None
        )
        if rf is not None and _response_format_type(rf) == "structural_tag":
            return request
        model_architecture = (
            self._model_config.architecture if self._model_config is not None else None
        )
        tools = request.tools
        # ``response_format`` wins if both it and ``structured_outputs`` supply JSON.
        schema = _schema_dict_from_chat_response_format(rf)
        if schema is None:
            schema = _schema_dict_from_structured_outputs(so)
        if schema is None and not _has_effective_tools(tools):
            return request
        if model_architecture is None:
            return request
        result = convert_schema_to_structural_tags(
            schema=schema,
            tools=tools,
            model_architecture=model_architecture,
        )
        if result is None:
            # Unsupported architectures are not in ``MODEL_TO_TAG_STYLE``.
            raise ValueError(
                "Failed to build structural_tag guided decoding constraints from "
                "this request's JSON schema and/or tools. The configured model "
                f"architecture ({model_architecture!r}) does not support Cohere "
                "command structural tags, or the schema cannot be expressed in "
                "that format."
            )
        request.structured_outputs = StructuredOutputsParams(structural_tag=result)
        # Folded JSON constraints into ``structural_tag``; drop ``response_format``
        # when it was the source so ``to_sampling_params`` does not also set ``json`` /
        # ``json_object`` (mutually exclusive in ``StructuredOutputsParams``).
        if isinstance(request, ChatCompletionRequest) and rf is not None:
            rf_type = _response_format_type(rf)
            if rf_type in ("json_schema", "json_object"):
                request.response_format = None
        return request


# melody's streaming filter only buffers a partial ``<co: ...>`` citation
# across ``write_decoded`` calls when ``stream_non_grounded_answer`` is
# set: otherwise, the moment an opening ``<co`` is seen without a
# closing ``</co: ...>`` in the same delta, the filter emits the partial
# marker bytes verbatim as plain content. In vLLM's streaming path the
# parser is fed one token (1-4 chars) per call, so an unbuffered filter
# will leak ``<co: 0>``-style markers into ``delta.content`` and never
# emit a ``FilterCitation`` for them. Enabling the flag flips the
# partial-match branch in melody's ``parse_citations`` (see
# ``src/parsing/citations_filter.rs``) to ``return (None, 0)`` -- i.e.
# keep buffering -- which lets a full citation eventually resolve.
# Non-streaming (unary) parsing receives the whole output in one call so
# the flag is a no-op there and we leave ``unary_opts`` alone.
class CohereCommand3ReasoningParser(BaseCohereCommandReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(
            tokenizer,
            *args,
            streaming_opts=PyFilterOptions().cmd3().stream_non_grounded_answer(),
            unary_opts=PyFilterOptions().cmd3().no_tools(),
            **kwargs,
        )


class CohereCommand4ReasoningParser(BaseCohereCommandReasoningParser):
    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(
            tokenizer,
            *args,
            streaming_opts=PyFilterOptions().cmd4().stream_non_grounded_answer(),
            unary_opts=PyFilterOptions().cmd4().no_tools(),
            **kwargs,
        )
