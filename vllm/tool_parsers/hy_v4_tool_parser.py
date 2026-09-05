# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import regex as re

import vllm.envs as envs
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

if TYPE_CHECKING:
    from xgrammar import StructuralTag

logger = init_logger(__name__)

# =============================================================================
# Plain-dict contract with the vLLM-side wrapper
# =============================================================================


class ToolSchema(TypedDict):
    """A tool as seen by this parser (built by the wrapper from ``request.tools``)."""

    name: str
    parameters: dict[str, Any] | None  # JSON Schema, or None


class ToolCallDict(TypedDict):
    """One extracted tool call. ``arguments`` is a JSON-encoded string."""

    name: str
    arguments: str


class ExtractResult(TypedDict):
    """Return of ``extract_tool_calls`` (non-streaming)."""

    tools_called: bool
    content: str | None
    tool_calls: list[ToolCallDict]


class StreamToolCall(TypedDict):
    """One streaming tool-call delta.

    ``new`` is True only on the chunk that first emits this tool's name -- the
    wrapper mints an id + sets type on that chunk only. ``arguments`` is the
    incremental JSON diff (or None).
    """

    index: int
    new: bool
    name: str | None
    arguments: str | None


class StreamDelta(TypedDict):
    """Return of ``extract_tool_calls_streaming``.

    A single engine delta can carry content plus every tool call that was
    drained from the buffer (batched decode), so ``tool_calls`` is not limited
    to one entry.
    """

    content: str | None
    tool_calls: list[StreamToolCall]


# NOTE: mirrored in ``vllm.reasoning.hy_v4_reasoning_parser`` (same pattern as
# ``gemma4_utils``) so neither package depends on the other.
def detect_token_suffix(tokenizer: TokenizerLike) -> str:
    """Detect the per-checkpoint suffix used by Hunyuan structural tokens.

    Args:
        tokenizer: Tokenizer of the served checkpoint.

    Returns:
        The suffix including its leading colon (e.g. ``":6124c78e"``), or ``""``
        when the checkpoint uses unsuffixed tokens.

    Raises:
        RuntimeError: The tokenizer declares the structural tokens through
            ``model_specific_special_tokens``, which transformers 5 no longer
            round-trips.
    """

    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        think_begin_as_special = init_kwargs.get(
            "model_specific_special_tokens", {}
        ).get("think_begin_token", "")
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


# =============================================================================
# Argument-value parsing utilities (stateless, pure functions)
#
# These map the model's raw ``<arg_value>`` text to typed Python values using
# the tool's JSON Schema. They do not touch any parser state, so they live at
# module level instead of tangling up HYV4ToolExtractor.
# =============================================================================

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


def _get_arg_schema(
    function_name: str,
    arg_key: str,
    tools: list[ToolSchema] | None,
) -> dict[str, Any]:
    """Look up a specific argument's property schema from the tools list."""
    if tools is None:
        return {}
    for tool in tools:
        if tool.get("name") == function_name:
            parameters = tool.get("parameters")
            if parameters is None:
                return {}
            return parameters.get("properties", {}).get(arg_key, {})
    logger.warning_once("No tool named '%s'.", function_name)
    return {}


def _get_schema_options(arg_schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize any property schema into a list of sub-schemas.
    - has type (single type) → return [arg_schema]
    - anyOf  → return the anyOf list
    - oneOf  → return the oneOf list
    - fallback → [{"type": "string"}]

    Note: single ``type`` has the highest priority.
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
    if "anyOf" in arg_schema:
        return arg_schema["anyOf"]
    if "oneOf" in arg_schema:
        return arg_schema["oneOf"]

    return [{"type": "string"}]


def _get_types(arg_schema: dict[str, Any]) -> set[str]:
    """Extract normalized, non-null type set from a property schema."""
    schemas = _get_schema_options(arg_schema)
    return {_normalize_type(s.get("type", "string")) for s in schemas} - {"null"}


def _is_only_string_type(
    function_name: str,
    arg_key: str,
    tools: list[ToolSchema] | None,
) -> bool:
    """Return True if the parameter's type set is exactly {"string"}.

    Only pure string types get partial value streaming; compound types like
    anyOf(string | array) do not, since the partial value might end up being a
    JSON array or object.
    """
    types = _get_types(_get_arg_schema(function_name, arg_key, tools))
    return types == {"string"}


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
        bool → int → number(wildcard_number)
        → json.loads for array/object → string → _deserialize
    """
    types = _get_types(_get_arg_schema(function_name, arg_key, tools))

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


# =============================================================================
# Stateful tool-call parser
# =============================================================================


class HYV4ToolExtractor:
    """Pure tool-call parsing logic for HYV4. Returns plain dicts; no vLLM types.

    Holds only parser state: the structural token strings/ids, the compiled
    regexes, and the streaming incremental state. Stateless argument parsing is
    delegated to the module-level utilities above.

    Streaming has two entry paths sharing one incremental argument parser,
    selected by the ``guided`` flag of :meth:`extract_tool_calls_streaming`:
    normal / auto tool choice detects markers on atomic token ids so ordinary
    ``<`` in content is never withheld, while structural-tag guided decoding
    detects them on the string level because the grammar may split a marker
    across sub-word tokens.
    """

    def __init__(self, vocab: dict[str, int], token_suffix: str, strict: bool):
        self._strict = strict

        # Kept so the wrapper can build a matching structural tag for
        # constrained decoding without re-scanning the tokenizer vocab.
        self.token_suffix: str = token_suffix

        # Running state read by the vLLM serving layer (mirrored by the wrapper).
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        # Streaming state: send tool name first, then return arguments at once
        self._streaming_tool_name: str | None = None  # tool name being streamed

        # State fields for incremental argument streaming
        self._completed_args: dict = {}  # closed {key: parsed_value}
        self._current_arg_key: str | None = None  # key being collected
        self._current_arg_is_string: bool = False  # is current arg pure string?
        self._streamed_json_len: int = 0  # bytes of JSON already sent

        self.tool_calls_start_token: str = f"<tool_calls{token_suffix}>"
        self.tool_calls_end_token: str = f"</tool_calls{token_suffix}>"

        self.tool_call_start_token: str = f"<tool_call{token_suffix}>"
        self.tool_call_end_token: str = f"</tool_call{token_suffix}>"

        self.arg_key_start_token: str = f"<arg_key{token_suffix}>"
        self.arg_key_end_token: str = f"</arg_key{token_suffix}>"

        self.arg_value_start_token: str = f"<arg_value{token_suffix}>"
        self.arg_value_end_token: str = f"</arg_value{token_suffix}>"

        logger.debug(
            "HYV4ToolExtractor structural tokens: %s ... %s, suffix=%r, strict=%s",
            self.tool_calls_start_token,
            self.tool_calls_end_token,
            token_suffix,
            strict,
        )

        # The suffix comes from the tokenizer vocab, so it must not be treated
        # as a pattern.
        self.tool_call_regex = re.compile(
            rf"{re.escape(self.tool_call_start_token)}(.*?)"
            rf"{re.escape(self.tool_call_end_token)}",
            re.DOTALL,
        )

        self.func_args_regex = re.compile(
            rf"{re.escape(self.arg_key_start_token)}(.*?)"
            rf"{re.escape(self.arg_key_end_token)}"
            rf"{re.escape(self.arg_value_start_token)}(.*?)"
            rf"{re.escape(self.arg_value_end_token)}",
            re.DOTALL,
        )

        self.tool_calls_start_token_id = vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = vocab.get(self.tool_call_end_token)
        self._buffer = ""

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "HYV4 tool extractor could not locate tool call "
                "start/end tokens in the tokenizer!"
            )

    def _extract_tool_calls(
        self,
        model_output: str,
        tools: list[ToolSchema] | None,
    ) -> list[ToolCallDict]:
        try:
            tool_calls: list[ToolCallDict] = []
            function_calls = self.tool_call_regex.findall(model_output)
            for tool_call_body in function_calls:
                arg_start = tool_call_body.find(self.arg_key_start_token)
                if arg_start == -1:
                    function_name = tool_call_body.strip()
                    function_args = ""
                else:
                    function_name = tool_call_body[:arg_start].strip()
                    function_args = tool_call_body[arg_start:].strip()

                arg_pairs = self.func_args_regex.findall(function_args)
                arg_dict = {}
                for key, value in arg_pairs:
                    arg_dict[key] = _parse_value(value, function_name, key, tools)
                tool_calls.append(
                    ToolCallDict(
                        name=function_name,
                        arguments=json.dumps(arg_dict, ensure_ascii=False),
                    )
                )
            return tool_calls
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return []

    def _extract_tool_calls_strict(
        self,
        model_output: str,
        tools: list[ToolSchema] | None,
    ) -> list[ToolCallDict]:
        start_idx = model_output.find(self.tool_calls_start_token)
        end_idx = model_output.find(self.tool_calls_end_token)
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            raise ValueError("Malformed tool block wrapper.")

        tool_block = model_output[
            start_idx + len(self.tool_calls_start_token) : end_idx
        ]
        if tool_block.count(self.tool_call_start_token) != tool_block.count(
            self.tool_call_end_token
        ):
            raise ValueError("Malformed tool_call wrapper count.")

        tool_calls: list[ToolCallDict] = []
        function_calls = self.tool_call_regex.findall(tool_block)
        if tool_block.count(self.tool_call_start_token) != len(function_calls):
            raise ValueError("Malformed tool_call body.")

        for tool_call_body in function_calls:
            arg_start = tool_call_body.find(self.arg_key_start_token)
            if arg_start == -1:
                function_name = tool_call_body.strip()
                function_args = ""
            else:
                function_name = tool_call_body[:arg_start].strip()
                function_args = tool_call_body[arg_start:].strip()

            if not function_name:
                raise ValueError("Empty function name in tool call.")

            if function_args.count(self.arg_key_start_token) != function_args.count(
                self.arg_key_end_token
            ) or function_args.count(self.arg_value_start_token) != function_args.count(
                self.arg_value_end_token
            ):
                raise ValueError("Malformed argument tag count.")

            arg_pairs = self.func_args_regex.findall(function_args)
            remainder = self.func_args_regex.sub("", function_args)
            if remainder.strip():
                raise ValueError("Unparsed argument payload remains.")

            arg_dict = {}
            for key, value in arg_pairs:
                arg_dict[key] = _parse_value(value, function_name, key, tools)

            tool_calls.append(
                ToolCallDict(
                    name=function_name,
                    arguments=json.dumps(arg_dict, ensure_ascii=False),
                )
            )

        if not tool_calls:
            raise ValueError("No valid tool calls extracted.")

        return tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        tools: list[ToolSchema] | None,
    ) -> ExtractResult:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractResult(
                tools_called=False, content=model_output, tool_calls=[]
            )
        else:
            try:
                if self._strict:
                    tool_calls = self._extract_tool_calls_strict(model_output, tools)
                else:
                    tool_calls = self._extract_tool_calls(model_output, tools)
                if not tool_calls:
                    raise ValueError("No valid tool calls extracted.")

                s_index = model_output.find(self.tool_calls_start_token)
                content = model_output[:s_index] if s_index != -1 else model_output
                return ExtractResult(
                    tools_called=True,
                    content=content if content else None,
                    tool_calls=tool_calls,
                )

            except Exception:
                logger.warning(
                    "HYV4ToolExtractor detected malformed tool output; "
                    "falling back to raw content."
                )
                return ExtractResult(
                    tools_called=False,
                    content=model_output,
                    tool_calls=[],
                )

    def _reset_streaming_tool_state(self):
        """Reset the streaming state for a single tool call."""
        self._streaming_tool_name = None
        self._completed_args = {}
        self._current_arg_key = None
        self._current_arg_is_string = False
        self._streamed_json_len = 0

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        tools: list[ToolSchema] | None,
        *,
        guided: bool = False,
    ) -> StreamDelta | None:
        """Consume one engine delta and emit content / tool-call deltas.

        Two entry paths share the same incremental argument parser:
        - ``guided=False`` (normal / auto tool choice): the structural markers
          arrive as atomic tokenizer tokens, so detection is done on token ids
          and ordinary ``<`` in model content is never withheld.
        - ``guided=True`` (structural-tag constrained decoding): the grammar
          only constrains strings, so ``<tool_calls...>`` may be split across
          sub-word tokens and must be detected on the string level.

        Args:
            previous_text: Text decoded before this delta.
            current_text: Full text decoded so far.
            delta_text: Text decoded in this delta.
            previous_token_ids: Token ids before this delta.
            current_token_ids: Token ids decoded so far.
            delta_token_ids: Token ids in this delta.
            tools: Plain tool schemas used for argument typing.
            guided: Whether guided decoding may split markers into sub-word
                tokens.

        Returns:
            A streaming delta carrying content and/or the tool calls drained
            from the buffer, or None when nothing can be emitted yet.
        """
        content_delta: str | None = None
        tool_calls: list[StreamToolCall] = []

        if guided:
            # Guided path: a structural-tag grammar constrains strings, not
            # token ids, so the atomic ``tool_calls_start_token_id`` never
            # appears in the token stream and an id-based check would
            # misclassify every tool delta as plain content.
            marker = self.tool_calls_start_token
            marker_pos = current_text.find(marker)

            if marker_pos == -1:
                # Marker not seen yet. The tail of ``current_text`` might still
                # be a partial prefix of the marker split across deltas; hold
                # that tail back and only stream the safe portion as content.
                safe_end = len(current_text) - partial_tag_overlap(current_text, marker)
                # Emit only the newly-arrived, safe content (delta relative to
                # what was previously streamed).
                content = current_text[len(previous_text) : safe_end]
                return StreamDelta(
                    content=content if content else None,
                    tool_calls=[],
                )

            # Marker present. Stream any content that precedes it (only the
            # part we have not already emitted), then buffer the tool payload.
            #
            # The payload is everything after the marker in the *full* text. We
            # keep the buffer in sync with it by only appending the portion that
            # arrived in this delta, mirroring the incremental ``+=`` semantics
            # while remaining correct when the marker itself was split across
            # sub-word tokens.
            content_end = marker_pos
            content = current_text[len(previous_text) : content_end]

            payload_start = marker_pos + len(marker)
            # How much of the payload had already been buffered before this
            # delta.
            prev_payload_len = max(len(previous_text) - payload_start, 0)
            new_payload = current_text[payload_start + prev_payload_len :]
            self._buffer += new_payload

            if content:
                content_delta = content
        else:
            # Atomic-token path: normal / auto generation emits the HYV4
            # structural markers as single tokenizer tokens. Pass ordinary
            # content through exactly as decoded, including comparison
            # operators such as ``<``.
            if self.tool_calls_start_token_id not in current_token_ids:
                return StreamDelta(content=delta_text, tool_calls=[])

            if self.tool_calls_start_token in delta_text:
                text_parts = delta_text.split(self.tool_calls_start_token)
                self._buffer += text_parts[-1]
                if text_parts[0]:
                    content_delta = text_parts[0]
            else:
                self._buffer += delta_text

        # Encountered finish, extract valid arguments
        if (
            current_text.find(self.tool_call_end_token + self.tool_calls_end_token)
            != -1
            and self._buffer.find(self.tool_call_end_token) == -1
        ):
            self._buffer += self.tool_call_end_token + self.tool_calls_end_token

        # Drain every complete tool call that is already buffered so a single
        # engine delta can carry content + N tool calls (batched decode).
        while True:
            cur_text = self._buffer

            # Haven't encountered tool_call start tag yet; drop the inter-tag
            # padding but keep any tail that may be a partial tool_call start
            # tag (guided decoding can split it across deltas).
            start_idx = cur_text.find(self.tool_call_start_token)
            if start_idx == -1 and self._streaming_tool_name is None:
                keep = partial_tag_overlap(cur_text, self.tool_call_start_token)
                self._buffer = cur_text[len(cur_text) - keep :] if keep else ""
                break

            # === Phase 1: Detect tool name (send when first arg_key is seen) =
            # ``name_new`` mirrors the old ``name_delta is not None``: it marks
            # the chunk that first emits this tool's name, which is the only
            # chunk the wrapper mints an id + type for.
            name_new = False
            pending_name: str | None = None
            if self._streaming_tool_name is None:
                arg_idx = cur_text.find(self.arg_key_start_token, start_idx)
                end_idx = cur_text.find(self.tool_call_end_token, start_idx)
                if arg_idx == -1 and end_idx == -1:
                    # tool name not yet complete; keep buffering from
                    # tool_call_start
                    self._buffer = cur_text[start_idx:]
                    break

                name_start = start_idx + len(self.tool_call_start_token)
                name_end = arg_idx if arg_idx != -1 else end_idx

                tool_name = cur_text[name_start:name_end].strip()
                self._streaming_tool_name = tool_name

                if arg_idx != -1:
                    self._buffer = cur_text[arg_idx:]
                else:
                    self._buffer = cur_text[end_idx:]

                # Increment tool_id and mark that a name chunk should be emitted
                self.current_tool_id += 1
                name_new = True
                pending_name = tool_name

                # Check if buffer already has complete arguments
                # (all-in-one-delta); otherwise only the name is ready now.
                if self.tool_call_end_token not in self._buffer:
                    tool_calls.append(
                        StreamToolCall(
                            index=self.current_tool_id,
                            new=True,
                            name=tool_name,
                            arguments=None,
                        )
                    )
                    break
                # Buffer already has a complete tool call; fall through to
                # phase 2 below.

            # === Phase 2: Incremental argument streaming ===
            result = self._extract_streaming_incremental(name_new, pending_name, tools)
            if result:
                tool_calls.extend(result["tool_calls"])

            # Continue only after the current tool call has completed and the
            # remaining structural payload starts with another tool call.
            # A literal ``<tool_call>`` may legitimately appear inside an open
            # string argument; in that case the current tool name is still set
            # and the buffer has not been consumed, so continuing would spin on
            # the same buffer forever.
            if self._streaming_tool_name is None and self._buffer.startswith(
                self.tool_call_start_token
            ):
                continue
            break

        if content_delta is not None or tool_calls:
            return StreamDelta(content=content_delta, tool_calls=tool_calls)
        return None

    def _make_args_delta(self, argument_diff: str) -> StreamDelta:
        """Build an args-only streaming delta (no id/type -> new=False)."""
        return StreamDelta(
            content=None,
            tool_calls=[
                StreamToolCall(
                    index=self.current_tool_id,
                    new=False,
                    name=None,
                    arguments=argument_diff,
                )
            ],
        )

    def _extract_streaming_incremental(
        self,
        name_new: bool,
        pending_name: str | None,
        tools: list[ToolSchema] | None,
    ) -> StreamDelta | None:
        """Incremental phase-2: scan tags in buffer, emit JSON diffs.

        Strategy:
        - Track completed args and emit each one as a JSON fragment.
        - For string-typed args, stream the value character-by-character.
        - Withhold the closing ``}`` until ``</tool_call>`` is seen.

        We build JSON manually via fragments rather than using json.dumps
        with a cursor, because json.dumps of partial-vs-full string values
        produces incompatible prefixes (e.g. ``""}`` vs ``"Hello"}``).
        """
        buf = self._buffer
        is_complete = self.tool_call_end_token in buf

        if is_complete:
            end_idx = buf.find(self.tool_call_end_token)
            args_text = buf[:end_idx]
            remaining = buf[end_idx + len(self.tool_call_end_token) :]
        else:
            args_text = buf
            remaining = ""

        # --- scan all fully closed kv pairs ---
        arg_pairs = self.func_args_regex.findall(args_text)
        for key, value in arg_pairs:
            key = key.strip()
            if key not in self._completed_args:
                self._completed_args[key] = _parse_value(
                    value, self._streaming_tool_name or "", key, tools
                )

        # --- detect partial (unclosed) kv at the tail ---
        last_closed_end = 0
        for m in self.func_args_regex.finditer(args_text):
            last_closed_end = m.end()
        tail = args_text[last_closed_end:]

        partial_key: str | None = None
        partial_value: str | None = None

        ak_start = tail.find(self.arg_key_start_token)
        if ak_start != -1:
            ak_end = tail.find(
                self.arg_key_end_token,
                ak_start + len(self.arg_key_start_token),
            )
            if ak_end != -1:
                partial_key = tail[
                    ak_start + len(self.arg_key_start_token) : ak_end
                ].strip()
                self._current_arg_key = partial_key
                self._current_arg_is_string = _is_only_string_type(
                    self._streaming_tool_name or "",
                    partial_key,
                    tools,
                )

                av_start = tail.find(self.arg_value_start_token, ak_end)
                if av_start != -1:
                    val_content_start = av_start + len(self.arg_value_start_token)
                    if self._current_arg_is_string:
                        raw_value = tail[val_content_start:]
                        # Guided decoding can split ``</arg_value>`` across
                        # deltas; hold back a partial closing tag so it is not
                        # streamed as part of the value.
                        keep = len(raw_value) - partial_tag_overlap(
                            raw_value, self.arg_value_end_token
                        )
                        partial_value = raw_value[:keep]
            else:
                # key not yet closed
                self._current_arg_key = None
                self._current_arg_is_string = False

        # --- build the current JSON snapshot as a string ---
        # We construct JSON manually so we can precisely control
        # what gets sent incrementally.
        snapshot_parts: list[str] = []
        for k, v in self._completed_args.items():
            k_json = json.dumps(k, ensure_ascii=False)
            v_json = json.dumps(v, ensure_ascii=False)
            snapshot_parts.append(f"{k_json}: {v_json}")

        if partial_key is not None and partial_value is not None:
            k_json = json.dumps(partial_key, ensure_ascii=False)
            # For string partial value, we build the JSON string
            # WITHOUT the closing quote, so the prefix stays stable
            # as the value grows.  The closing `"` and `}` will be
            # sent when the value or tool_call closes.
            escaped_val = json.dumps(partial_value, ensure_ascii=False)[1:-1]
            # Note: no closing " here - it's appended only on close
            snapshot_parts.append(f'{k_json}: "{escaped_val}')

        snapshot = "{" + ", ".join(snapshot_parts) + "}"

        # --- compute diff ---
        argument_diff: str | None = None

        if is_complete:
            # Tool call finished - send everything remaining.
            # Build final snapshot with proper JSON (all values closed).
            final_args = dict(self._completed_args)
            final_json = json.dumps(final_args, ensure_ascii=False)
            if self._streamed_json_len < len(final_json):
                argument_diff = final_json[self._streamed_json_len :]
            self._streamed_json_len = len(final_json)

            # Record into prev_tool_call_arr
            self.prev_tool_call_arr.append(
                {
                    "name": self._streaming_tool_name,
                    "arguments": final_args,
                }
            )
            self.streamed_args_for_tool.append(final_json)

            self._reset_streaming_tool_state()
            self._buffer = remaining
        else:
            # Still in progress - withhold the tail.
            # For open strings: snapshot ends with ...partial_val}
            #   we withhold "}" (1 char) - the missing closing " will
            #   be sent when the value closes.
            # For no open string: snapshot ends with ...value"}
            #   we withhold "}" (1 char).
            end = len(snapshot) - 1  # exclude trailing "}"
            if end > self._streamed_json_len:
                argument_diff = snapshot[self._streamed_json_len : end]
                self._streamed_json_len = end

        # --- construct return dict ---
        if name_new and argument_diff:
            return StreamDelta(
                content=None,
                tool_calls=[
                    StreamToolCall(
                        index=self.current_tool_id,
                        new=True,
                        name=pending_name,
                        arguments=argument_diff,
                    )
                ],
            )
        elif name_new:
            return StreamDelta(
                content=None,
                tool_calls=[
                    StreamToolCall(
                        index=self.current_tool_id,
                        new=True,
                        name=pending_name,
                        arguments=None,
                    )
                ],
            )
        elif argument_diff:
            return self._make_args_delta(argument_diff)
        else:
            return None


def build_tool_extractor(
    tokenizer: TokenizerLike, *, strict: bool
) -> HYV4ToolExtractor:
    return HYV4ToolExtractor(
        tokenizer.get_vocab(), detect_token_suffix(tokenizer), strict
    )


def _tools_to_plain(tools) -> list[ToolSchema] | None:
    """Convert tool-schema objects into plain dicts for the extractor."""
    if not tools:
        return None
    return [
        ToolSchema(name=t.function.name, parameters=t.function.parameters)
        for t in tools
    ]


class HYV4ToolParser(ToolParser):
    """vLLM adapter around :class:`HYV4ToolExtractor`.

    Converts the extractor's plain dicts to vLLM protocol objects, mints
    tool-call ids, and mirrors ``prev_tool_call_arr`` /
    ``streamed_args_for_tool`` so the serving layer sees live state.
    """

    # HYV4 emits its own <tool_calls>/<tool_call>/<arg_key>/<arg_value>
    # structural tokens, not a plain JSON array. The generic required/named
    # JSON streaming path (streaming.extract_required_tool_call_streaming)
    # cannot parse this format and produces misaligned multi-tool deltas in
    # streaming mode. Fall back to our own (auto-path) parser for
    # required/named too, so streaming and non-streaming stay consistent.
    supports_required_and_named: bool = False
    structural_tag_model: str = "hy_v4"

    def __init__(self, tokenizer: TokenizerLike, tools=None):
        super().__init__(tokenizer, tools)

        # Tools are supplied at construction (``tool_parser(tokenizer,
        # request.tools)``); keep a plain-dict copy for the extractor.
        self._plain_tools = _tools_to_plain(tools)
        # Malformed tag structure is rejected and returned as raw content
        # instead of a half-parsed tool call.
        self._extractor = build_tool_extractor(tokenizer, strict=True)
        self._mirror_state()

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
    ) -> StructuralTag | None:
        """Build a structural tag matching HYV4's tool tokens.

        Overridden only to pass the per-checkpoint ``token_suffix`` through to
        the builder; everything else follows the base implementation.

        Named (forced) tool choice is constrained with the same structural tag:
        because ``supports_required_and_named`` is False, required *and* named
        requests are parsed by the HYV4 extractor, which only recognizes the HYV4
        structural tokens -- the default JSON guided-decoding path would make
        the model emit JSON the extractor silently drops.

        Args:
            request: The request being adjusted.
            reasoning: Whether the grammar also covers the reasoning phase.

        Returns:
            The structural tag, or None when structural tagging does not apply.
        """
        if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
            return None

        # Only constrain required / forced tool choice with the structural tag.
        # For "auto", tool calling is optional and most requests never emit a
        # tool call, so applying the grammar to every auto request would make
        # the whole stream pay the output inflation and the per-token
        # bitmask/accept overhead for a minority benefit. Skip it here so auto
        # falls through to free generation; the HYV4 parser still reads back
        # any tool calls the model emits.
        tool_choice = request.tool_choice
        if (
            tool_choice is None
            or tool_choice == "auto"
            or getattr(tool_choice, "mode", None) == "auto"
        ):
            return None

        from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

        try:
            return get_model_structural_tag(
                model=self.structural_tag_model,
                tools=request.tools,
                tool_choice=request.tool_choice,
                reasoning=reasoning,
                token_suffix=self._extractor.token_suffix,
            )
        except Exception:
            logger.warning(
                "HYV4ToolParser failed to build a structural tag; falling back "
                "to the default guided-decoding path.",
                exc_info=True,
            )
            return None

    def _mirror_state(self) -> None:
        # Alias the SAME list objects the extractor appends to, so the serving
        # layer reads live state. ``current_tool_id`` is an int, so re-read it.
        self.prev_tool_call_arr = self._extractor.prev_tool_call_arr
        self.streamed_args_for_tool = self._extractor.streamed_args_for_tool
        self.current_tool_id = self._extractor.current_tool_id

    @staticmethod
    def _is_guided(request: ChatCompletionRequest) -> bool:
        """Whether a structural tag constrains this request's decoding.

        ``get_structural_tag`` only applies to required / named tool choice and
        the serving layer records the result on the request. A structural tag
        constrains strings rather than token ids, so the ``<tool_calls>`` marker
        can arrive split across sub-word tokens and must be detected on the
        string level.

        Args:
            request: The request being streamed.

        Returns:
            True when the streaming parser must use the string-marker path.
        """
        structured_outputs = getattr(request, "structured_outputs", None)
        return (
            structured_outputs is not None
            and structured_outputs.structural_tag is not None
        )

    def _tools(self, request: ChatCompletionRequest) -> list[ToolSchema] | None:
        # Prefer the tools passed at construction; fall back to the request.
        if self._plain_tools is not None:
            return self._plain_tools
        return _tools_to_plain(getattr(request, "tools", None))

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        result = self._extractor.extract_tool_calls(model_output, self._tools(request))
        self._mirror_state()
        return ExtractedToolCallInformation(
            tools_called=result["tools_called"],
            content=result["content"],
            tool_calls=[
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                    ),
                )
                for tc in result["tool_calls"]
            ],
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        delta = self._extractor.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            self._tools(request),
            guided=self._is_guided(request),
        )
        self._mirror_state()
        if delta is None:
            return None

        tool_calls: list[DeltaToolCall] = []
        for tc in delta.get("tool_calls", []):
            # Only set the fields that are present, leaving the rest UNSET (not
            # None). Streaming serialization uses exclude_unset, so an explicit
            # ``arguments=None`` would emit ``"arguments": null`` and break
            # clients that concatenate argument deltas. Matches hy_v3.
            fn_kwargs: dict[str, str] = {}
            name = tc.get("name")
            if name is not None:
                fn_kwargs["name"] = name
            arguments = tc.get("arguments")
            if arguments is not None:
                fn_kwargs["arguments"] = arguments
            function = DeltaFunctionCall(**fn_kwargs)
            if tc.get("new"):
                # First chunk for this tool: mint an id + set type here only.
                tool_calls.append(
                    DeltaToolCall(
                        index=tc["index"],
                        id=make_tool_call_id(),
                        type="function",
                        function=function,
                    )
                )
            else:
                tool_calls.append(DeltaToolCall(index=tc["index"], function=function))

        # A single engine delta can carry content plus N drained tool calls.
        # Only set the fields that are present so streaming serialization
        # (exclude_unset) never emits ``content: null`` on a tool-call chunk.
        content = delta.get("content")
        kwargs: dict[str, Any] = {}
        if content is not None:
            kwargs["content"] = content
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        if not kwargs:
            return None
        return DeltaMessage(**kwargs)
