# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)


class HYV3ToolParser(ToolParser):
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
    # Following the same approach as
    # qwen3coder_tool_parser._convert_param_value which uses
    # param_type.startswith("int"), startswith("uint"), etc.
    _INTEGER_PREFIXES: tuple[str, ...] = (
        "int",
        "uint",
        "long",
        "short",
        "unsigned",
    )
    _NUMBER_PREFIXES: tuple[str, ...] = ("num", "float")

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        """Map non-standard type aliases to JSON Schema standard names.

        First performs exact lookup in _TYPE_ALIASES. On miss, falls back
        to prefix-based matching using startswith()
          - int*/uint*/long*/short*/unsigned* → "integer"
          - num*/float* → "number"
        """
        exact = HYV3ToolParser._TYPE_ALIASES.get(raw_type)
        if exact is not None:
            return exact
        lower = raw_type.lower()
        if any(lower.startswith(p) for p in HYV3ToolParser._INTEGER_PREFIXES):
            return "integer"
        if any(lower.startswith(p) for p in HYV3ToolParser._NUMBER_PREFIXES):
            return "number"
        return raw_type

    @staticmethod
    def _get_arg_schema(
        function_name: str,
        arg_key: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> dict:
        """Look up a specific argument's property schema from the tools list."""
        if tools is None:
            return {}
        for tool in tools:
            if tool.function.name == function_name:
                if tool.function.parameters is None:
                    return {}
                return tool.function.parameters.get("properties", {}).get(arg_key, {})
        logger.warning("No tool named '%s'.", function_name)
        return {}

    @staticmethod
    def _get_schema_options(arg_schema: dict) -> list[dict]:
        """Normalize any property schema into a list of sub-schemas.
        - has type (single type) → return [arg_schema]
        - anyOf  → return the anyOf list
        - oneOf  → return the oneOf list
        - fallback → [{"type": "string"}]

        Note: single ``type`` has the highest priority.
        """
        if "type" in arg_schema:
            return [arg_schema]
        if "anyOf" in arg_schema:
            return arg_schema["anyOf"]
        if "oneOf" in arg_schema:
            return arg_schema["oneOf"]

        return [{"type": "string"}]

    @staticmethod
    def _get_types(arg_schema: dict) -> set[str]:
        """Extract normalized, non-null type set from a property schema."""
        schemas = HYV3ToolParser._get_schema_options(arg_schema)
        return {
            HYV3ToolParser._normalize_type(s.get("type", "string")) for s in schemas
        } - {"null"}

    @staticmethod
    def _is_only_string_type(
        function_name: str,
        arg_key: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> bool:
        """Return True if the parameter's type set is exactly {"string"}.

        Only pure string types get partial value streaming; compound types
        like anyOf(string | array) do not, since the partial value might
        end up being a JSON array or object.
        """
        arg_schema = HYV3ToolParser._get_arg_schema(function_name, arg_key, tools)
        types = HYV3ToolParser._get_types(arg_schema)
        return types == {"string"}

    @staticmethod
    def _try_parse_bool(value: str) -> bool | None:
        """Try to parse a string as bool; return None on failure."""
        lower = value.lower()
        if lower == "true":
            return True
        elif lower == "false":
            return False
        return None

    @staticmethod
    def _try_parse_int(value: str) -> int | None:
        """Try to parse a string as int; return None on failure."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _try_parse_wildcard_number(value: str) -> int | float | None:
        """Try to parse a string as a number (int or float).

        Decision rule: if the string contains '.' or 'e'/'E' (scientific
        notation), parse as float; otherwise parse as int.

        Examples:
            "5"    → int(5)
            "5.0"  → float(5.0)
            "5.3"  → float(5.3)
            "1e3"  → float(1000.0)
            "-3"   → int(-3)

        Return None on failure.
        """
        try:
            if "." in value or "e" in value or "E" in value:
                return float(value)
            return int(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
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

    @staticmethod
    def _parse_value(
        value: str,
        function_name: str,
        arg_key: str,
        tools: list[ChatCompletionToolsParam] | None,
    ) -> Any:
        """Unified argument value parser with anyOf/oneOf support.

        Fallthrough chain:
            bool → int → number(wildcard_number)
            → json.loads for array/object
            → string → _deserialize
        """
        arg_schema = HYV3ToolParser._get_arg_schema(function_name, arg_key, tools)
        types = HYV3ToolParser._get_types(arg_schema)

        # 1. Try bool
        if "boolean" in types:
            result_bool = HYV3ToolParser._try_parse_bool(value)
            if result_bool is not None:
                return result_bool

        # 2. Try int
        if "integer" in types:
            result_int = HYV3ToolParser._try_parse_int(value)
            if result_int is not None:
                return result_int

        # 3. Try number (wildcard_number: int if no '.'/e/E, float otherwise)
        if "number" in types:
            result_number = HYV3ToolParser._try_parse_wildcard_number(value)
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
        return HYV3ToolParser._deserialize(value)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        # Streaming state: send tool name first, then return arguments at once
        self._streaming_tool_name: str | None = None  # tool name being streamed

        # State fields for incremental argument streaming
        self._completed_args: dict = {}  # closed {key: parsed_value}
        self._current_arg_key: str | None = None  # key being collected
        self._current_arg_is_string: bool = False  # is current arg pure string?
        self._streamed_json_len: int = 0  # bytes of JSON already sent

        self.tool_calls_start_token: str = "<tool_calls>"
        self.tool_calls_end_token: str = "</tool_calls>"

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

        self.tool_sep_token: str = "<tool_sep>"

        self.arg_key_start_token: str = "<arg_key>"
        self.arg_key_end_token: str = "</arg_key>"

        self.arg_value_start_token: str = "<arg_value>"
        self.arg_value_end_token: str = "</arg_value>"

        self.tool_call_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}"
            rf"(.*?){self.tool_call_end_token}",
            re.DOTALL,
        )

        self.tool_call_portion_regex = re.compile(
            rf"{self.tool_call_start_token}(.*?){self.tool_sep_token}(.*)", re.DOTALL
        )

        self.func_args_regex = re.compile(
            rf"{self.arg_key_start_token}(.*?){self.arg_key_end_token}\s*"
            rf"{self.arg_value_start_token}(.*?){self.arg_value_end_token}",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)
        self._buffer = ""

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "HYV3 Tool parser could not locate tool call "
                "start/end tokens in the tokenizer!"
            )

    def _extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> list[ToolCall]:
        try:
            function_call_tuples = []
            # start_token{name}sep_token{args}end_token...
            function_calls = self.tool_call_regex.findall(model_output)
            if function_calls:
                function_call_tuples.extend(function_calls)
                remaining = model_output.split(self.tool_call_end_token)[-1]
                function_calls = self.tool_call_portion_regex.findall(remaining)
                function_call_tuples += function_calls
            else:
                function_calls = self.tool_call_portion_regex.findall(model_output)
                if function_calls:
                    function_call_tuples.extend(function_calls)
            tool_calls = []
            for match in function_call_tuples:
                function_name, function_args = match
                function_name = function_name.strip()
                function_args = function_args.strip()

                arg_pairs = self.func_args_regex.findall(function_args)
                arg_dict = {}
                for key, value in arg_pairs:
                    parsed_value = HYV3ToolParser._parse_value(
                        value, function_name, key, request.tools
                    )
                    arg_dict[key] = parsed_value
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_name,
                            arguments=json.dumps(arg_dict, ensure_ascii=False),
                        ),
                    )
                )
            return tool_calls
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return []

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # sanity check; avoid unnecessary processing
        if self.tool_calls_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            try:
                tool_calls = self._extract_tool_calls(model_output, request)

                s_index = model_output.find(self.tool_calls_start_token)
                content = model_output[:s_index] if s_index != -1 else model_output
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
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
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Check whether current tokens contain the tool_calls start token
        if self.tool_calls_start_token_id not in current_token_ids:
            return DeltaMessage(content=delta_text)

        # Encountered tool_calls start tag; extract preceding content and buffer
        if self.tool_calls_start_token in delta_text:
            text_parts = delta_text.split(self.tool_calls_start_token)
            self._buffer += text_parts[-1]
            if text_parts[0]:
                return DeltaMessage(content=text_parts[0])
            # Don't return None; continue processing buffer for complete content
        else:
            self._buffer += delta_text

        # Encountered finish, extract valid arguments
        if (
            current_text.find(self.tool_call_end_token + self.tool_calls_end_token)
            != -1
            and self._buffer.find(self.tool_call_end_token) == -1
        ):
            self._buffer += self.tool_call_end_token + self.tool_calls_end_token

        cur_text = self._buffer

        # Haven't encountered tool_call start tag yet; keep buffering
        start_idx = cur_text.find(self.tool_call_start_token)
        if start_idx == -1 and self._streaming_tool_name is None:
            self._buffer = ""
            return None

        # === Phase 1: Detect tool name (send when tool_sep_token is seen) ===
        name_delta: DeltaMessage | None = None
        if self._streaming_tool_name is None:
            sep_idx = cur_text.find(self.tool_sep_token)
            if sep_idx == -1:
                # tool_sep not yet seen; keep buffering from tool_call_start
                self._buffer = cur_text[start_idx:]
                return None

            # Extract tool name: between tool_call_start_token and tool_sep_token
            name_start = start_idx + len(self.tool_call_start_token)
            tool_name = cur_text[name_start:sep_idx].strip()
            self._streaming_tool_name = tool_name

            # Update buffer: keep only content after tool_sep (i.e. the args portion)
            self._buffer = cur_text[sep_idx + len(self.tool_sep_token) :]

            # Increment tool_id and send a chunk containing only the name
            self.current_tool_id += 1
            self._current_tool_call_id = make_tool_call_id()
            name_delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=self._current_tool_call_id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_name,
                        ),
                    )
                ]
            )

            # Check if buffer already has complete arguments (all-in-one-delta)
            if self.tool_call_end_token not in self._buffer:
                return name_delta
            # Buffer already has a complete tool call; continue to phase 2 below

        # === Phase 2: Incremental argument streaming ===
        return self._extract_streaming_incremental(name_delta, request)

    def _make_args_delta(self, argument_diff: str) -> DeltaMessage:
        """Build a DeltaMessage containing only an arguments diff."""
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(arguments=argument_diff),
                )
            ]
        )

    def _extract_streaming_incremental(
        self,
        name_delta: DeltaMessage | None,
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
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
                parsed_value = HYV3ToolParser._parse_value(
                    value, self._streaming_tool_name or "", key, request.tools
                )
                self._completed_args[key] = parsed_value

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
                self._current_arg_is_string = HYV3ToolParser._is_only_string_type(
                    self._streaming_tool_name or "",
                    partial_key,
                    request.tools,
                )

                av_start = tail.find(self.arg_value_start_token, ak_end)
                if av_start != -1:
                    val_content_start = av_start + len(self.arg_value_start_token)
                    if self._current_arg_is_string:
                        partial_value = tail[val_content_start:]
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
            escaped_val = (
                partial_value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            # Note: no closing " here – it's appended only on close
            snapshot_parts.append(f'{k_json}: "{escaped_val}')

        snapshot = "{" + ", ".join(snapshot_parts) + "}"

        # --- compute diff ---
        argument_diff: str | None = None

        if is_complete:
            # Tool call finished – send everything remaining.
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
            # Still in progress – withhold the tail.
            # For open strings: snapshot ends with ...partial_val}
            #   we withhold "}" (1 char) – the missing closing " will
            #   be sent when the value closes.
            # For no open string: snapshot ends with ...value"}
            #   we withhold "}" (1 char).
            end = len(snapshot) - 1  # exclude trailing "}"
            if end > self._streamed_json_len:
                argument_diff = snapshot[self._streamed_json_len : end]
                self._streamed_json_len = end

        # --- construct return DeltaMessage ---
        if name_delta is not None and argument_diff:
            nd_func = name_delta.tool_calls[0].function
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=self._current_tool_call_id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=nd_func.name if nd_func else None,
                            arguments=argument_diff,
                        ),
                    )
                ]
            )
        elif name_delta is not None:
            return name_delta
        elif argument_diff:
            return self._make_args_delta(argument_diff)
        else:
            return None
