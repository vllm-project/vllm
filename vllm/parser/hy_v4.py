# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 reasoning and XML tool calls, including checkpoint-specific tokens."""

from __future__ import annotations

import ast
import copy
import json
from typing import Any, TypedDict
from urllib.parse import unquote

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.logger import init_logger
from vllm.parser.engine.events import EventType, SemanticEvent
from vllm.parser.engine.incremental_lexer import LexToken
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ArgValueExtender,
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import get_function_tools

logger = init_logger(__name__)
DEFAULT_REASONING_EFFORT = "high"


class ToolSchema(TypedDict):
    name: str
    parameters: dict[str, Any] | None


def detect_token_suffix(tokenizer: TokenizerLike) -> str:
    """Detect the per-checkpoint suffix used by Hunyuan structural tokens.

    Args:
        tokenizer: Tokenizer of the served checkpoint.

    Returns:
        The suffix including its leading colon (e.g. `":6124c78e"`), or `""`
        when the checkpoint uses unsuffixed tokens.

    Raises:
        RuntimeError: The tokenizer declares the structural tokens through
            `model_specific_special_tokens`, which transformers 5 no longer
            round-trips.

    """
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    think_begin_as_special = init_kwargs.get("model_specific_special_tokens", {}).get(
        "think_begin_token", ""
    )
    if think_begin_as_special:
        raise RuntimeError(
            "This checkpoint declares HYV4 structural tokens (think_begin_token"
            "/toolcalls_begin_token/argkey_begin_token) in "
            "tokenizer_config.json, which transformers 5 no longer supports. "
            "Remove those fields and keep the tokens in the tokenizer's own "
            "token definitions so the suffix can be read from the vocab."
        )

    structural_token_re = re.compile(
        r"<(?:think|tool_calls|tool_call|arg_key|arg_value)(:[^\s>]+)?>"
    )
    for token in tokenizer.get_vocab():
        match = structural_token_re.fullmatch(token)
        if match:
            return match.group(1) or ""

    return ""


_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "bool": "boolean",
    "binary": "boolean",
    "int": "integer",
    "float": "number",
    "double": "number",
    "list": "array",
    "dict": "object",
    "map": "object",
}

# Prefix-based wildcard matching for non-standard type names.
# Following the same approach as qwen3coder_tool_parser._convert_param_value
# which uses param_type.startswith("int"), startswith("uint"), etc.
_INTEGER_PREFIXES: tuple[str, ...] = ("int", "uint", "long", "short", "unsigned")
_NUMBER_PREFIXES: tuple[str, ...] = ("num", "float")
_SCHEMA_TYPE_KEYS: tuple[str, ...] = ("type", "anyOf", "oneOf")
_MAX_REF_HOPS = 3
_MAX_ANCHOR_DEPTH = 8


def _normalize_type(raw_type: str) -> str:
    """Map non-standard type aliases to JSON Schema standard names.

    First performs exact lookup in _TYPE_ALIASES. On miss, falls back to
    prefix-based matching using startswith()
      - int*/uint*/long*/short*/unsigned* → "integer"
      - num*/float* → "number"
    """
    exact = _TYPE_ALIASES.get(raw_type)
    if exact is not None:
        return exact
    lower = raw_type.lower()
    if any(lower.startswith(p) for p in _INTEGER_PREFIXES):
        return "integer"
    if any(lower.startswith(p) for p in _NUMBER_PREFIXES):
        return "number"
    return raw_type


def _is_ref_type(schema: Any) -> bool:
    """Return whether a schema expresses its type only through `$ref`."""
    return (
        isinstance(schema, dict)
        and isinstance(schema.get("$ref"), str)
        and not any(schema.get(key) for key in _SCHEMA_TYPE_KEYS)
    )


def _resolve_json_pointer(root_schema: Any, pointer: str) -> Any:
    """Resolve a local JSON Pointer against the parameter schema."""
    if pointer == "":
        return root_schema
    if not pointer.startswith("/"):
        return None

    target = root_schema
    for raw_token in pointer[1:].split("/"):
        token = unquote(raw_token).replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict):
            if token not in target:
                return None
            target = target[token]
        elif isinstance(target, list):
            if not token.isdigit() or int(token) >= len(target):
                return None
            target = target[int(token)]
        else:
            return None
    return target


def _find_schema_anchor(
    root_schema: Any,
    anchor: str,
) -> dict[str, Any] | None:
    """Find a local JSON Schema anchor with bounded traversal."""
    stack: list[tuple[Any, int]] = [(root_schema, 0)]
    visited: set[int] = set()
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_ANCHOR_DEPTH or id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, dict):
            if node.get("$anchor") == anchor:
                return node
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        else:
            continue
        stack.extend((child, depth + 1) for child in children)
    return None


def _resolve_local_schema_ref(
    root_schema: dict[str, Any],
    ref: str,
) -> dict[str, Any] | None:
    """Resolve a same-document pointer, root reference, or anchor."""
    base, separator, fragment = ref.partition("#")
    # Never fetch external schema documents from the parsing path.
    if not separator or base:
        return None
    if fragment == "" or fragment.startswith("/"):
        target = _resolve_json_pointer(root_schema, fragment)
    else:
        target = _find_schema_anchor(root_schema, unquote(fragment))
    return target if isinstance(target, dict) else None


def _resolve_ref_type(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    seen_refs: frozenset[str] = frozenset(),
    ref_hops: int = 0,
) -> dict[str, Any]:
    """Resolve a reference to type information, without mutating the schema."""
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return {}
    if ref in seen_refs:
        logger.warning_once(
            "Cyclic JSON Schema $ref '%s'; treating it as untyped.", ref
        )
        return {}
    if ref_hops >= _MAX_REF_HOPS:
        logger.warning_once(
            "JSON Schema $ref chain exceeded %d hops at '%s'; treating it as untyped.",
            _MAX_REF_HOPS,
            ref,
        )
        return {}

    target = _resolve_local_schema_ref(root_schema, ref)
    if target is None:
        logger.warning_once(
            "Could not resolve local JSON Schema $ref '%s'; treating it as untyped.",
            ref,
        )
        return {}

    type_schema = {
        key: copy.deepcopy(target[key]) for key in _SCHEMA_TYPE_KEYS if target.get(key)
    }
    if type_schema:
        return (
            _resolve_union_ref_types(
                type_schema, root_schema, seen_refs | {ref}, ref_hops + 1
            )
            or type_schema
        )
    if _is_ref_type(target):
        return _resolve_ref_type(target, root_schema, seen_refs | {ref}, ref_hops + 1)
    return {}


def _resolve_union_ref_types(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    seen_refs: frozenset[str] = frozenset(),
    ref_hops: int = 0,
) -> dict[str, Any] | None:
    """Resolve reference-only branches of `anyOf` and `oneOf`."""
    resolved_schema: dict[str, Any] | None = None
    for union_key in ("anyOf", "oneOf"):
        branches = schema.get(union_key)
        if not isinstance(branches, list):
            continue
        if not any(_is_ref_type(branch) for branch in branches):
            continue
        if resolved_schema is None:
            resolved_schema = copy.deepcopy(schema)
        for index, branch in enumerate(resolved_schema[union_key]):
            if not _is_ref_type(branch):
                continue
            resolved_branch = _resolve_ref_type(
                branch, root_schema, seen_refs, ref_hops
            )
            if resolved_branch:
                resolved_schema[union_key][index] = resolved_branch
    return resolved_schema


def _get_arg_schema(
    function_name: str,
    arg_key: str,
    tools: list[ToolSchema] | None,
) -> dict[str, Any]:
    """Look up an argument and resolve its local schema type references."""
    if tools is None:
        return {}
    for tool in tools:
        if tool.get("name") != function_name:
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            return {}
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return {}
        arg_schema = properties.get(arg_key, {})
        if not isinstance(arg_schema, dict):
            return {}
        try:
            if _is_ref_type(arg_schema):
                return _resolve_ref_type(arg_schema, parameters) or arg_schema
            return _resolve_union_ref_types(arg_schema, parameters) or arg_schema
        except Exception as exc:
            logger.warning_once(
                "Failed to resolve the schema type for argument '%s' (%s: %s); "
                "treating it as untyped.",
                arg_key,
                type(exc).__name__,
                exc,
            )
            return {}
    logger.warning_once("No tool named '%s'.", function_name)
    return {}


def _get_schema_options(arg_schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize any property schema into a list of sub-schemas.
    - has type (single type) → return [arg_schema]
    - anyOf / oneOf → recursively expand union branches
    - fallback → [{"type": "string"}]

    Note: single `type` has the highest priority.
    """
    if "type" in arg_schema:
        type_val = arg_schema["type"]
        # JSON Schema allows "type" to be an array to represent union types,
        # e.g. "type": ["string", "object"].
        # Expand it into an anyOf-equivalent format:
        #   [{"type": "string"}, {"type": "object"}]
        # so that _get_types / _parse_value can handle it uniformly later.
        if isinstance(type_val, list):
            return [{"type": t} for t in type_val]
        return [arg_schema]
    for union_key in ("anyOf", "oneOf"):
        if union_key in arg_schema:
            # A resolved reference may itself expose a union or type array.
            return [
                option
                for branch in arg_schema[union_key]
                for option in _get_schema_options(branch)
            ]

    return [{"type": "string"}]


def _get_types(arg_schema: dict[str, Any]) -> set[str]:
    """Extract normalized, non-null type set from a property schema."""
    schemas = _get_schema_options(arg_schema)
    return {_normalize_type(s.get("type", "string")) for s in schemas} - {"null"}


def _allows_null(arg_schema: dict[str, Any]) -> bool:
    """Return whether the property schema admits a null value."""
    return any(
        _normalize_type(schema.get("type", "string")) == "null"
        for schema in _get_schema_options(arg_schema)
    )


def _is_only_string_type(
    function_name: str,
    arg_key: str,
    tools: list[ToolSchema] | None,
) -> bool:
    """Stream partial values only for non-nullable pure strings.

    A quoted prefix cannot grow into an unquoted null, array, or object.
    """
    arg_schema = _get_arg_schema(function_name, arg_key, tools)
    return _get_types(arg_schema) == {"string"} and not _allows_null(arg_schema)


def _try_parse_bool(value: str) -> bool | None:
    """Try to parse a string as bool; return None on failure."""
    lower = value.lower()
    if lower == "true":
        return True
    elif lower == "false":
        return False
    return None


def _try_parse_int(value: str) -> int | None:
    """Try to parse a string as int; return None on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _try_parse_wildcard_number(value: str) -> int | float | None:
    """Try to parse a string as a number (int or float).

    Decision rule: if the string contains '.' or 'e'/'E' (scientific notation),
    parse as float; otherwise parse as int.

    Examples: "5" → 5, "5.0" → 5.0, "1e3" → 1000.0, "-3" → -3.
    Return None on failure.
    """
    try:
        if "." in value or "e" in value or "E" in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return None


def _deserialize(value: str) -> Any:
    """Deserialize a string value using json.loads then ast.literal_eval."""
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _parse_value(
    value: str,
    function_name: str,
    arg_key: str,
    tools: list[ToolSchema] | None,
) -> Any:
    """Unified argument value parser with anyOf/oneOf support.

    Fallthrough chain:
        null → bool → int → number(wildcard_number)
        → json.loads for array/object → string → _deserialize
    """
    arg_schema = _get_arg_schema(function_name, arg_key, tools)
    types = _get_types(arg_schema)
    if value == "null" and _allows_null(arg_schema):
        return None

    # 1. Try bool
    if "boolean" in types:
        result_bool = _try_parse_bool(value)
        if result_bool is not None:
            return result_bool

    # 2. Try int
    if "integer" in types:
        result_int = _try_parse_int(value)
        if result_int is not None:
            return result_int

    # 3. Try number (wildcard_number: int if no '.'/e/E, float otherwise)
    if "number" in types:
        result_number = _try_parse_wildcard_number(value)
        if result_number is not None:
            return result_number

    # 4. Try json.loads (covers array/object and other unlisted types)
    if types - {"string", "boolean", "integer", "number"}:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

    # 5. String fallback
    if "string" in types:
        return value

    # 6. Final fallback
    return _deserialize(value)


class _HYV4Arguments:
    def __init__(self, suffix: str):
        key_start, key_end, value_start, value_end = (
            re.escape(f"<{slash}{name}{suffix}>")
            for name in ("arg_key", "arg_value")
            for slash in ("", "/")
        )
        self.pairs = re.compile(
            f"{key_start}(.*?){key_end}{value_start}(.*?){value_end}", re.DOTALL
        )
        self.open_pair = re.compile(
            f"{key_start}(.*?){key_end}{value_start}(.*)$", re.DOTALL
        )
        self.value_end = f"</arg_value{suffix}>"
        self.calls_start = f"<tool_calls{suffix}>"
        self.calls_end = f"</tool_calls{suffix}>"
        self.call_start = f"<tool_call{suffix}>"
        self.call_end = f"</tool_call{suffix}>"
        self.key_start = f"<arg_key{suffix}>"
        self.calls = re.compile(
            re.escape(self.call_start) + "(.*?)" + re.escape(self.call_end), re.DOTALL
        )

    def validate(self, text: str) -> bool | None:
        start, bare = text.find(self.calls_start), text.find(self.call_start)
        if bare != -1 and (start == -1 or bare < start):
            block = text[bare:].split(self.calls_end)[0]
        elif start != -1:
            end = text.find(self.calls_end, start)
            if end == -1:
                return False
            block = text[start + len(self.calls_start) : end]
        else:
            return None
        calls = list(self.calls.finditer(block))
        if not calls or not (
            block.count(self.call_start) == block.count(self.call_end) == len(calls)
        ):
            return False
        for call in calls:
            name, marker, arguments = call.group(1).partition(self.key_start)
            if not name.strip():
                return False
            if marker and self.pairs.sub("", marker + arguments).strip():
                return False
        return True

    def __call__(self, raw: str, partial: bool) -> str:
        pairs = []
        end = 0
        for match in self.pairs.finditer(raw):
            pairs.append(match.groups())
            end = match.end()
        if partial and (match := self.open_pair.search(raw[end:])):
            pairs.append(match.groups())
        # Duplicate keys must retain their positions in already-emitted JSON.
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: "
                f"{json.dumps(value, ensure_ascii=False)}"
                for key, value in pairs
            )
            + "}"
        )

    def extender(self, raw: str) -> ArgValueExtender | None:
        end = 0
        for match in self.pairs.finditer(raw):
            end = match.end()
        match = self.open_pair.search(raw[end:])
        if match is None:
            return None
        window = match.group(2)[-len(self.value_end) :]

        def extend(delta: str) -> str | None:
            nonlocal window
            text = window + delta
            if self.value_end in text:
                return None
            window = text[-len(self.value_end) :]
            return json.dumps(delta, ensure_ascii=False)[1:-1]

        return extend


def hy_v4_config(suffix: str, thinking: bool = True) -> ParserEngineConfig:
    terminals = {
        f"{name.upper()}_{edge}": f"<{slash}{name}{suffix}>"
        for name in ("think", "tool_calls", "tool_call", "arg_key", "arg_value")
        for edge, slash in (("START", ""), ("END", "/"))
    }
    transitions = {
        (ParserState.CONTENT, "TOOL_CALLS_START"): Transition(
            ParserState.TOOL_PREAMBLE
        ),
        (ParserState.TOOL_NAME, "ARG_KEY_START"): Transition(
            ParserState.TOOL_ARGS, (EventType.ARG_VALUE_CHUNK,)
        ),
        (ParserState.TOOL_NAME, "TOOL_CALL_END"): Transition(
            ParserState.TOOL_BETWEEN, (EventType.TOOL_CALL_END,)
        ),
        (ParserState.TOOL_ARGS, "TOOL_CALL_END"): Transition(
            ParserState.TOOL_BETWEEN, (EventType.TOOL_CALL_END,)
        ),
        (ParserState.TOOL_BETWEEN, "TOOL_CALLS_END"): Transition(
            ParserState.TOOL_PREAMBLE
        ),
    }
    for state in (
        ParserState.CONTENT,
        ParserState.TOOL_PREAMBLE,
        ParserState.TOOL_BETWEEN,
    ):
        transitions[state, "TOOL_CALL_START"] = Transition(
            ParserState.TOOL_NAME, (EventType.TOOL_CALL_START,)
        )
    if thinking:
        transitions[ParserState.REASONING, "THINK_START"] = Transition(
            ParserState.REASONING
        )
        transitions[ParserState.REASONING, "THINK_END"] = Transition(
            ParserState.CONTENT, (EventType.REASONING_END,)
        )
    arguments = _HYV4Arguments(suffix)
    return ParserEngineConfig(
        name="hy_v4",
        terminals=terminals,
        token_id_terminals=terminals,
        transitions=transitions,
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        wait_for_reasoning=thinking,
        arg_converter=arguments,
        arg_value_extender=arguments.extender,
        tool_args_json=False,
        strip_trailing_reasoning_whitespace=False,
        drop_whitespace_only_content_before_tools=False,
        strip_content_whitespace_with_tools=False,
    )


class _HYV4StreamingEngine(StreamingParserEngine):
    guided_tools = False

    def _process_lex_tokens(self, tokens: list[LexToken]) -> list[SemanticEvent]:
        if not self.guided_tools:
            return super()._process_lex_tokens(tokens)
        # Structural-tag decoding can spell tool markers using ordinary tokens.
        events = []
        for token in tokens:
            if token.terminal in self._tool_terminals:
                events.extend(
                    self._on_terminal(token.terminal, token.value, token.token_count)
                )
            else:
                events.extend(super()._process_lex_tokens([token]))
        return events


class HYV4Parser(ParserEngine):
    _streaming_engine_cls = _HYV4StreamingEngine
    _engine: _HYV4StreamingEngine

    def __init__(
        self, tokenizer: TokenizerLike, tools=None, *, thinking=None, **kwargs
    ):
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        effort = (
            chat_kwargs.get("reasoning_effort")
            or kwargs.get("reasoning_effort")
            or DEFAULT_REASONING_EFFORT
        )
        self.thinking = effort != "no_think" if thinking is None else thinking
        self.token_suffix = detect_token_suffix(tokenizer)
        self._arguments = _HYV4Arguments(self.token_suffix)
        self._plain_tools_source: list | None = None
        self._plain_tools: list[ToolSchema] | None = None
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=hy_v4_config(self.token_suffix, self.thinking),
            **kwargs,
        )
        if self.thinking and (
            self._reasoning_start_token_id is None
            or self._reasoning_end_token_id is None
        ):
            raise RuntimeError("HYV4 parser could not locate think start/end tokens!")

    def _check_skip_tool_parsing(self, request) -> None:
        super()._check_skip_tool_parsing(request)
        structured = getattr(request, "structured_outputs", None)
        self._engine.guided_tools = (
            structured is not None and structured.structural_tag is not None
        )

    def _single_pass_parse(self, text, token_ids, initial_state=None):
        content = text
        if self.thinking and initial_state is not ParserState.CONTENT:
            content = text.partition(self.reasoning_end_str)[2]
        saved = self.skip_tool_parsing
        if not saved and self._arguments.validate(content) is False:
            self.skip_tool_parsing = True
        try:
            return super()._single_pass_parse(text, token_ids, initial_state)
        finally:
            self.skip_tool_parsing = saved

    def is_reasoning_end_streaming(self, input_ids, delta_ids) -> bool:
        return not self.thinking or self._reasoning_end_token_id in delta_ids

    @property
    def reasoning_ended(self) -> bool:
        return not self.thinking or super().reasoning_ended

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking:
            return input_ids
        if self._reasoning_end_token_id not in input_ids:
            return []
        return input_ids[input_ids.index(self._reasoning_end_token_id) + 1 :]

    def _fix_arg_types(self, args_json: str, func_name: str) -> str:
        tools = self._get_plain_tools()
        pairs = json.loads(args_json, object_pairs_hook=list)
        converted = [
            (key, _parse_value(value, func_name.strip(), key, tools))
            for key, value in pairs
        ]
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: "
                f"{json.dumps(value, ensure_ascii=False)}"
                for key, value in converted
            )
            + "}"
        )

    def _get_plain_tools(self):
        if self._plain_tools_source is not self._tools:
            self._plain_tools_source = self._tools
            self._plain_tools = []
            for tool in get_function_tools(self._tools or []):
                function = (
                    tool.function
                    if isinstance(tool, ChatCompletionToolsParam)
                    else tool
                )
                self._plain_tools.append(
                    ToolSchema(name=function.name, parameters=function.parameters)
                )
        return self._plain_tools

    def _emit_name_delta(self, idx, deltas, name) -> None:
        name = name.strip() if name else name
        super()._emit_name_delta(idx, deltas, name)
        for tool in self._get_plain_tools() or []:
            if tool["name"] == name:
                properties = (tool["parameters"] or {}).get("properties", {})
                self._tool_slots[idx].string_keys = {
                    json.dumps(key, ensure_ascii=False)[1:-1]
                    for key in properties
                    if _is_only_string_type(name, key, self._plain_tools)
                }
                break

    def _handle_tool_end(self, event: SemanticEvent, deltas) -> None:
        if event.tool_index < len(self._tool_slots):
            self._tool_slots[event.tool_index].name = self._tool_slots[
                event.tool_index
            ].name.strip()
        super()._handle_tool_end(event, deltas)
