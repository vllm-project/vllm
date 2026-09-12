# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cohere Command parser.

Uses ``cohere_melody`` as a stateful parser filter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, TypeGuard

import regex as re
import xgrammar as xgr

try:
    from cohere_melody import (
        AccumulatedToolCall,
        FilterAggregatedResult,
        PyFilter,
        PyFilterOptions,
    )
except ImportError as e:
    raise ImportError(
        "The Cohere Command parser requires the `cohere_melody` package. "
        "Install it with:\n    pip install 'cohere-melody>=0.14.0'"
    ) from e

from vllm.entrypoints.cohere.cohere_chat_message import (
    Citation,
    CitationSource,
    CohereDeltaMessage,
)
from vllm.entrypoints.generate.base.protocol import (
    AnyResponseFormat,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.cohere_command_reasoning_parser import (
    BaseCohereCommandReasoningParser,
)
from vllm.renderers.cohere import POSITION_TO_SOURCE_KEY
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.cohere_command_tool_parser import BaseCohereCommandToolParser

if TYPE_CHECKING:
    # Declared in melody's stub only; not exported at runtime.
    from cohere_melody import FilterCitation, Source

    from vllm.config import ModelConfig
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.utils import Tool

PositionToSource = Mapping[tuple[int, int], CitationSource]


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
    #   tools ::= ws "[" ws tool (ws "," ws tool)* ws "]" ws
    #   ws    ::= (" " | "\t" | "\n")*
    #   tool  ::= <tool_a> | <tool_b> | ...         (one alternative per input)
    #   <tool_x>     ::= <tool_x>root               (per-tool xgrammar rules)
    #   <tool_x>root ::= ...                        (from xgr.Grammar.from_json_schema)
    tool_alternatives = "tool ::= " + " | ".join(tool_dictionary.keys())
    tool_rules = "\n    ".join(tool_dictionary.values())
    grammar = f"""root ::= tools
    tools ::= ws "[" ws tool (ws "," ws tool)* ws "]" ws
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


def _melody_sources_to_vllm(
    sources: list[Source], position_to_source: PositionToSource | None
) -> list[CitationSource]:
    """Resolve melody ``Source`` addresses into :class:`CitationSource` objects.

    Melody reports each source as ``(tool_call_index, tool_result_indices)``.
    ``position_to_source`` is built by
    :meth:`vllm.entrypoints.cohere.serving.CohereServingChatV2._build_position_to_source`
    and forwarded through ``chat_template_kwargs``; each index fans out to one
    resolved source, and unresolved positions are skipped. Without a map (a
    request that did not come through the Cohere v2 handler) every source is
    dropped and the serving layer discards the citation at wire-coercion time.
    """
    if not position_to_source:
        return []
    out: list[CitationSource] = []
    for source in sources:
        for idx in source.tool_result_indices:
            info = position_to_source.get((source.tool_call_index, idx))
            if info is not None:
                out.append(info)
    return out


def _melody_citations_to_vllm(
    citations: list[FilterCitation], position_to_source: PositionToSource | None
) -> list[Citation] | None:
    """Convert melody ``FilterCitation`` objects into :class:`Citation`.

    Citations whose sources all fail to resolve are still returned with
    ``sources=[]``; the serving layer drops them so the fail-closed policy
    lives in one place.
    """
    if not citations:
        return None
    return [
        Citation(
            start=c.start_index,
            end=c.end_index,
            text=c.text,
            sources=_melody_sources_to_vllm(c.sources, position_to_source),
            type="THINKING_CONTENT" if c.is_thinking else "TEXT_CONTENT",
        )
        for c in citations
    ]


class CohereCommandParser(DelegatingParser):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        model_config: ModelConfig | None = None,
        **kwargs,
    ):
        super().__init__(tokenizer, tools, *args, model_config=model_config, **kwargs)
        self._model_config = model_config
        self.start_token_id = tokenizer.convert_tokens_to_ids("<|START_THINKING|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|END_THINKING|>")
        shim = self._reasoning_parser or self._tool_parser
        assert isinstance(
            shim, (BaseCohereCommandReasoningParser, BaseCohereCommandToolParser)
        )
        options = getattr(PyFilterOptions(), shim.melody_preset)()
        self._streaming = PyFilter(options)
        self._unary = PyFilter(options)
        # Citations from the most recent ``parse`` call, surfaced on
        # ``CohereChatMessage`` by ``CohereServingChatV2``.
        self.last_unary_citations: list[Citation] | None = None
        # Melody may emit a tool-call id before the function name; hold it until
        # the first name delta so clients receive both together.
        self._pending_tool_ids: dict[int, str] = {}
        ctk = kwargs.get("chat_template_kwargs") or {}
        raw_map = ctk.get(POSITION_TO_SOURCE_KEY)
        self._position_to_source: PositionToSource | None = (
            raw_map if isinstance(raw_map, Mapping) else None
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        # keep special tokens to ensure melody can correctly identify boundaries.
        request.skip_special_tokens = False
        request = self._apply_structural_tags(request)
        return super().adjust_request(request)

    def _apply_structural_tags(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Fold ``tools`` and any JSON schema into a Cohere ``structural_tag``."""
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

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == self.start_token_id:
                depth += 1
                continue
            if token_id == self.end_token_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count

    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        result = self._unary.process_full_text(model_output)
        self.last_unary_citations = _melody_citations_to_vllm(
            result.citations, self._position_to_source
        )
        tool_calls: list[FunctionCall] | None = None
        if self._tools_allowed(request):
            tool_calls = [
                FunctionCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                for tc in result.tool_calls
            ] or None
        return result.reasoning, result.content, tool_calls

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        results = [self._streaming.write_decoded(delta_text)]
        if finished:
            results.append(self._streaming.flush_partials())
        return self._to_delta(results, request)

    def _tools_allowed(self, request: ChatCompletionRequest | ResponsesRequest) -> bool:
        return self._tool_parser is not None and request.tool_choice != "none"

    def _to_delta(
        self,
        results: list[FilterAggregatedResult],
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> CohereDeltaMessage | None:
        content = self._join(r.content for r in results)
        reasoning = (
            self._join(r.reasoning for r in results)
            if request.include_reasoning
            else None
        )
        tool_calls: list[DeltaToolCall] = []
        citations: list[Citation] = []
        for result in results:
            if result.tool_calls and self._tools_allowed(request):
                tool_calls.extend(self._tool_call_deltas(result.tool_calls))
            citations.extend(
                _melody_citations_to_vllm(result.citations, self._position_to_source)
                or []
            )
        if not (content or reasoning or tool_calls or citations):
            return None
        # ``CohereDeltaMessage`` carries citations to the wire via
        # ``SerializeAsAny[DeltaMessage]``; it drops the field when unset so the
        # shape matches a plain ``DeltaMessage``.
        message = CohereDeltaMessage()
        if content is not None:
            message.content = content
        if reasoning is not None:
            message.reasoning = reasoning
        if tool_calls:
            message.tool_calls = tool_calls
        if citations:
            message.citations = citations
        return message

    @staticmethod
    def _join(parts: Any) -> str | None:
        present = [p for p in parts if p is not None]
        return "".join(present) if present else None

    def _tool_call_deltas(
        self, melody_tool_calls: list[AccumulatedToolCall]
    ) -> list[DeltaToolCall]:
        tool_calls: list[DeltaToolCall] = []
        for tc in melody_tool_calls:
            name = tc.name or None
            arguments = tc.arguments or None
            if tc.id:
                self._pending_tool_ids[tc.index] = tc.id
            # Empty strings are melody streaming placeholders; skip fragments
            # that carry neither a name nor argument text.
            if name is None and arguments is None:
                continue
            function = DeltaFunctionCall()
            if name is not None:
                function.name = name
            if arguments is not None:
                function.arguments = arguments
            delta = DeltaToolCall(index=tc.index, function=function)
            if name is not None:
                resolved_id = tc.id or self._pending_tool_ids.pop(tc.index, None)
                if resolved_id is not None:
                    delta.id = resolved_id
                delta.type = "function"
            tool_calls.append(delta)
        return tool_calls
