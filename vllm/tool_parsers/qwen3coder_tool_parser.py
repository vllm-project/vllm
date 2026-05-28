# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import contextlib
import json
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.envs import VLLM_ENFORCE_STRICT_TOOL_CALLING
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import find_tool_properties, partial_tag_overlap

logger = init_logger(__name__)


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        self._sent_content_idx = 0

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if param_name not in param_config:
            if param_config != {}:
                logger.debug(
                    "Parsed parameter '%s' is not defined in the tool "
                    "parameters for tool '%s', directly returning the "
                    "string value.",
                    param_name,
                    func_name,
                )
            return param_value

        # ``allows_null`` is True when the schema explicitly admits a
        # null value (either via ``"type": "null"`` or in an ``anyOf``
        # union).  A nullable parameter must convert the literal
        # ``"null"`` / ``"None"`` to JSON null even when the primary
        # type is ``string`` — otherwise a Qwen3.5-trained model that
        # emits the Python ``None`` literal leaves the client with the
        # string ``"None"`` for a nullable optional.
        allows_null = False
        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
            allows_null = param_type == "null"
        elif (
            isinstance(param_config[param_name], dict)
            and "anyOf" in param_config[param_name]
        ):
            # Extract the first non-null type from the anyOf list so that
            # nullable schemas like {"anyOf": [{"type": "string"},
            # {"type": "null"}]} behave as "string", not "object".
            param_type = "string"
            picked = False
            for option in param_config[param_name]["anyOf"]:
                if isinstance(option, dict) and "type" in option:
                    opt_type = str(option["type"]).strip().lower()
                    if opt_type == "null":
                        allows_null = True
                    elif not picked:
                        param_type = opt_type
                        picked = True
        else:
            param_type = "string"
        # Nullable schemas: recognise "null" / "None" up front so a
        # string-typed nullable still maps to JSON null.
        if allows_null and param_value.lower() in ("null", "none"):
            return None
        # String type takes precedence: preserve the raw value (including
        # the literal "null") rather than converting it to Python None.
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        # For non-string types, "null" maps to JSON null.  Also accept
        # the Python literal "None" so that Qwen3.5-trained models — whose
        # chat template renders null args via ``| string`` (yielding the
        # literal "None" in the prompt) — round-trip nullable values
        # correctly.
        if param_value.lower() in ("null", "none"):
            return None
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not an "
                    "integer in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.debug(
                    "Parsed value '%s' of parameter '%s' is not a boolean "
                    "(`true` or `false`) in tool '%s', degenerating to "
                    "false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value == "true"
        else:
            is_container_type = (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            )
            if is_container_type:
                try:
                    parsed = json.loads(param_value)
                    # A model trained with a buggy template
                    # (json.dumps(str(dict))) may output a JSON-encoded
                    # Python repr like "{'k': 'v'}". json.loads returns a
                    # string in that case — try one more parse.
                    if isinstance(parsed, str):
                        with contextlib.suppress(ValueError, SyntaxError, TypeError):
                            parsed = ast.literal_eval(parsed)
                    return parsed
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug(
                        "Parsed value '%s' of parameter '%s' cannot be "
                        "parsed with json.loads in tool '%s', will try "
                        "other methods to parse it.",
                        param_value,
                        param_name,
                        func_name,
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
                # Same double-decode for container types whose raw text
                # had no JSON outer layer (e.g. bare Python repr
                # "{'k': 'v'}").
                if is_container_type and isinstance(param_value, str):
                    with contextlib.suppress(ValueError, SyntaxError, TypeError):
                        param_value = ast.literal_eval(param_value)
            except (ValueError, SyntaxError, TypeError):
                logger.debug(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "converted via Python `ast.literal_eval()` in tool "
                    "'%s', degenerating to string.",
                    param_value,
                    param_name,
                    func_name,
                )
            return param_value

    def _next_structural_param_start(
        self,
        text: str,
        start_pos: int = 0,
        valid_param_names: set[str] | None = None,
    ) -> int:
        """Return index of next structural ``<parameter=NAME>`` from
        start_pos.  Structural means preceded by ``\\n`` or at position 0.
        If valid_param_names is given, NAME must also be in that set.
        Returns -1 if none found.
        """
        ni = start_pos
        prefix_len = len(self.parameter_prefix)
        while True:
            ni = text.find(self.parameter_prefix, ni)
            if ni == -1:
                return -1
            if ni == 0 or text[ni - 1] == "\n":
                if valid_param_names is not None:
                    name_end = text.find(">", ni + prefix_len)
                    if (
                        name_end != -1
                        and text[ni + prefix_len : name_end] in valid_param_names
                    ):
                        return ni
                    ni += 1
                    continue
                return ni
            ni += 1

    def _find_true_function_end(self, text: str) -> int:
        """Return the index of the real structural ``</function>`` in text
        (followed with optional whitespace by ``</tool_call>`` or end of
        string), or -1 if none found.  Skips ``</function>`` that appears
        as literal text inside a parameter value.
        """
        search_pos = 0
        while True:
            idx = text.find(self.function_end_token, search_pos)
            if idx == -1:
                return -1
            after = text[idx + len(self.function_end_token) :]
            stripped = after.lstrip()
            if stripped == "" or stripped.startswith(self.tool_call_end_token):
                return idx
            search_pos = idx + len(self.function_end_token)

    def _scan_to_structural_function_end(
        self,
        after_func_open: str,
        valid_param_names: set[str] | None = None,
    ) -> int:
        """Scan a function body — text immediately following the closing
        ``>`` of ``<function=NAME>`` — by walking through structural
        ``<parameter=NAME>...</parameter>`` blocks and return the index of
        the structural ``</function>`` in ``after_func_open``.

        This is more robust than ``_find_true_function_end`` when the
        parameter value embeds a complete literal ``<tool_call>...
        </function>\\n</tool_call>`` block: that nested ``</function>``
        is followed by ``</tool_call>`` and would pass the lookahead
        heuristic, but it is INSIDE a parameter and must be skipped.

        Handles a "missing </parameter>" malformation by treating the
        next structural ``<parameter=NAME>`` (with NAME unseen so far)
        as an implicit end.

        Returns -1 if the body is incomplete or malformed.
        """
        pos = 0
        n = len(after_func_open)
        seen: set[str] = set()
        while pos < n:
            # Skip whitespace between params
            while pos < n and after_func_open[pos] in " \t\n\r":
                pos += 1
            if pos >= n:
                return -1
            if after_func_open[pos:].startswith(self.function_end_token):
                return pos
            if not after_func_open[pos:].startswith(self.parameter_prefix):
                # Unexpected token before </function>; fall back to the
                # legacy heuristic on the rest of the text.
                rest_offset = self._find_true_function_end(after_func_open[pos:])
                return pos + rest_offset if rest_offset != -1 else -1
            name_end = after_func_open.find(">", pos + len(self.parameter_prefix))
            if name_end == -1:
                return -1
            param_name = after_func_open[pos + len(self.parameter_prefix) : name_end]
            value_start = name_end + 1
            if value_start < n and after_func_open[value_start] == "\n":
                value_start += 1
            param_end = self._find_true_param_end(
                after_func_open[value_start:],
                valid_param_names,
                require_lookahead=True,
            )
            if param_end == -1:
                # Missing </parameter> malformation: try the next
                # structural <parameter=NAME> with NAME unseen so far
                # as the implicit end.
                unseen: set[str] | None = (
                    (valid_param_names - seen - {param_name})
                    if valid_param_names is not None
                    else None
                )
                implicit_end = self._next_structural_param_start(
                    after_func_open[value_start:], 0, unseen
                )
                if implicit_end == -1:
                    return -1
                pos = value_start + implicit_end
                seen.add(param_name)
                continue
            seen.add(param_name)
            pos = value_start + param_end + len(self.parameter_end_token)
        return -1

    def _advance_to_next_tool(self, current_text: str) -> None:
        """Advance streaming state to the next tool call.

        Updates _sent_content_idx to skip past the completed tool call's
        closing tag, then resets per-tool state for the next invocation.
        Called both on normal delta boundaries and during speculative-
        decoding recursion when multiple complete tool calls arrive in one
        delta.

        Uses STRUCTURAL ``</tool_call>`` positions so a literal
        ``</tool_call>`` embedded in a parameter value (e.g. a code
        snippet) does not move ``_sent_content_idx`` to the wrong place.
        """
        end_positions = self._structural_tool_call_end_positions(current_text)
        target = self.current_tool_index
        if target < len(end_positions):
            self._sent_content_idx = max(
                self._sent_content_idx,
                end_positions[target] + len(self.tool_call_end_token),
            )

        self.current_tool_index += 1
        self.header_sent = False
        self.param_count = 0
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self.is_tool_call_started = False

    def _find_true_tool_call_end(self, text: str) -> int:
        """Return the index of the real structural ``</tool_call>`` in
        text (followed with optional whitespace by another ``<tool_call>``
        or end of string), or -1 if none found.
        """
        search_pos = 0
        while True:
            idx = text.find(self.tool_call_end_token, search_pos)
            if idx == -1:
                return -1
            after = text[idx + len(self.tool_call_end_token) :]
            stripped = after.lstrip()
            if stripped == "" or stripped.startswith(self.tool_call_start_token):
                return idx
            search_pos = idx + len(self.tool_call_end_token)

    def _structural_tool_call_end_positions(self, text: str) -> list[int]:
        """Return positions of every STRUCTURAL ``</tool_call>`` in text.

        Walks each ``<tool_call>...</tool_call>`` top-level block by
        following ``<function=NAME>``, scanning the body via
        ``_scan_to_structural_function_end`` (which steps over parameter
        values that may contain literal ``<tool_call>``, ``<function=...>``,
        ``</function>`` or ``</tool_call>`` strings), then matching the
        trailing ``</tool_call>``.

        Falls back to a lookahead heuristic when the walker cannot
        determine a structural close (incomplete body, malformed XML).
        """
        positions: list[int] = []
        pos = 0
        n = len(text)
        while pos < n:
            tc_start = text.find(self.tool_call_start_token, pos)
            if tc_start == -1:
                break
            body_start = tc_start + len(self.tool_call_start_token)
            func_open = text.find(self.tool_call_prefix, body_start)
            if func_open == -1:
                break
            name_end = text.find(">", func_open + len(self.tool_call_prefix))
            if name_end == -1:
                break
            func_name = text[func_open + len(self.tool_call_prefix) : name_end]
            valid_params: set[str] | None = None
            if self.tools:
                cfg = find_tool_properties(self.tools, func_name)
                if cfg:
                    valid_params = set(cfg.keys())
            body_after_name = text[name_end + 1 :]
            func_end_rel = self._scan_to_structural_function_end(
                body_after_name, valid_params
            )
            if func_end_rel == -1:
                # Body incomplete; the structural </tool_call> is not
                # yet known.  Stop walking — DO NOT fall back to the
                # legacy heuristic for the rest of the text, because a
                # literal </tool_call> embedded in an unfinished
                # parameter would be erroneously treated as structural.
                break
            func_end_abs = (name_end + 1) + func_end_rel
            after = text[func_end_abs + len(self.function_end_token) :]
            i = 0
            while i < len(after) and after[i] in " \t\n\r":
                i += 1
            if not after[i:].startswith(self.tool_call_end_token):
                break
            tc_end_pos = func_end_abs + len(self.function_end_token) + i
            positions.append(tc_end_pos)
            pos = tc_end_pos + len(self.tool_call_end_token)
        return positions

    def _find_true_param_end(
        self,
        value_text: str,
        valid_param_names: set[str] | None = None,
        require_lookahead: bool = False,
    ) -> int:
        """Find the true end of a parameter value in value_text.

        A ``</parameter>`` is structural only when it is followed by
        another structural delimiter (schema-known ``<parameter=NAME>``,
        ``</function>``, ``</tool_call>``) or — in non-streaming mode —
        end-of-string.  Nested ``<parameter=NAME>`` opens are tracked
        for depth REGARDLESS of whether NAME is in the schema: a
        literal nested tool_call may use NAMEs that are not in the
        outer tool's schema, but its literal ``</parameter>`` still
        pairs with the literal open and must not be mistaken for a
        structural close.

        Returns the index of the true ``</parameter>`` in value_text, or
        -1 if incomplete.
        """
        depth = 0
        pos = 0
        param_prefix_len = len(self.parameter_prefix)
        param_end_len = len(self.parameter_end_token)

        while pos < len(value_text):
            # Use UNFILTERED structural opens for depth tracking so that
            # a literal ``<parameter=UNKNOWN>`` (NAME not in the outer
            # schema) still increments depth and its matching literal
            # ``</parameter>`` is balanced — otherwise that close would
            # appear unmatched and pass the structural lookahead.
            next_open = self._next_structural_param_start(value_text, pos, None)
            next_close = value_text.find(self.parameter_end_token, pos)
            if next_close == -1:
                return -1

            if next_open != -1 and next_open < next_close:
                depth += 1
                pos = next_open + param_prefix_len
            elif depth == 0:
                after = value_text[next_close + param_end_len :]
                stripped = after.lstrip()
                structural_next_param = False
                if stripped.startswith(self.parameter_prefix):
                    if valid_param_names is not None:
                        name_start = len(self.parameter_prefix)
                        name_end = stripped.find(">", name_start)
                        if name_end != -1:
                            structural_next_param = (
                                stripped[name_start:name_end] in valid_param_names
                            )
                    else:
                        structural_next_param = True
                if (
                    (stripped == "" and not require_lookahead)
                    or structural_next_param
                    or stripped.startswith(self.function_end_token)
                    or stripped.startswith(self.tool_call_end_token)
                ):
                    return next_close
                pos = next_close + param_end_len
            else:
                depth -= 1
                pos = next_close + param_end_len

        return -1

    @staticmethod
    def _is_valid_function_name(name: str) -> bool:
        """Return True when ``name`` looks like a real function identifier
        and not a stray template token, malformed tag, or freeform text.

        Rejects names that contain template-syntax characters (``{``,
        ``}``, ``<``, ``>``), whitespace, quotes, or are empty.  Permits
        identifiers, dashes (``max-retries``), dots (``user.name``),
        slashes (``namespace/tool``), and Unicode letters.
        """
        if not name:
            return False
        forbidden = set("{}<>\"' \t\n\r")
        return not any(c in forbidden for c in name)

    def _parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        # Reject phantom tool calls produced when the model writes an
        # unrendered Jinja template or pseudo-XML in its response (e.g.
        # ``<function={{ tc.name }}>``).  Surfacing such names as real
        # tool calls causes "tool not found" errors at the client and
        # makes agents loop.
        if not self._is_valid_function_name(function_name):
            return None
        param_config = find_tool_properties(self.tools, function_name)
        valid_param_names: set[str] | None = (
            set(param_config.keys()) if param_config else None
        )
        parameters = function_call_str[end_index + 1 :]
        param_dict: dict = {}
        pos = 0
        while True:
            # Find next structural <parameter=NAME> at the top level.  We
            # do NOT filter the outer search by schema: callers may
            # legitimately send a parameter whose name is not declared
            # in the schema (e.g. renamed fields).  Schema filtering is
            # applied only when scanning INSIDE a parameter value, to
            # disambiguate real nested delimiters from literal text.
            param_start = self._next_structural_param_start(parameters, pos, None)
            if param_start == -1:
                break
            name_start = param_start + len(self.parameter_prefix)
            name_end = parameters.find(">", name_start)
            if name_end == -1:
                break
            param_name = parameters[name_start:name_end]
            value_text = parameters[name_end + 1 :]

            param_end = self._find_true_param_end(value_text, valid_param_names)
            if param_end == -1:
                # No true </parameter> found (malformed XML or incomplete).
                # Fallback 1: next structural <parameter= acts as implicit end.
                next_struct_param = self._next_structural_param_start(
                    value_text, 0, valid_param_names
                )
                if next_struct_param != -1:
                    param_value = value_text[:next_struct_param]
                    pos = (name_end + 1) + next_struct_param
                else:
                    # Fallback 2: use structural </function> boundary or end
                    func_end = self._find_true_function_end(value_text)
                    if func_end != -1:
                        param_value = value_text[:func_end]
                    else:
                        param_value = value_text
                    pos = len(parameters)
            else:
                param_value = value_text[:param_end]
                pos = (name_end + 1) + param_end + len(self.parameter_end_token)

            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find tool_calls using a structural delimiter approach:
        # a real </tool_call> is followed by another <tool_call> or
        # end-of-text.  This skips </tool_call> that appears as literal
        # text inside a parameter value.
        raw_tool_calls: list[str] = []
        search_pos = 0
        while True:
            tc_start = model_output.find(self.tool_call_start_token, search_pos)
            if tc_start == -1:
                break
            after_open = model_output[tc_start + len(self.tool_call_start_token) :]
            tc_end = -1
            inner_search = 0
            while True:
                idx = after_open.find(self.tool_call_end_token, inner_search)
                if idx == -1:
                    tc_end = -1
                    break
                after_close = after_open[idx + len(self.tool_call_end_token) :]
                stripped = after_close.lstrip()
                if stripped == "" or stripped.startswith(self.tool_call_start_token):
                    tc_end = idx
                    break
                inner_search = idx + len(self.tool_call_end_token)
            if tc_end == -1:
                raw_tool_calls.append(after_open)
                break
            raw_tool_calls.append(after_open[:tc_end])
            search_pos = (
                tc_start
                + len(self.tool_call_start_token)
                + tc_end
                + len(self.tool_call_end_token)
            )

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        # Use a parameter-aware walk to find the structural </function>:
        # when the value of a parameter embeds a complete literal
        # ``<tool_call>...</function>\n</tool_call>`` block, the nested
        # ``</function>`` is followed by ``</tool_call>`` and would pass
        # the simple "followed by </tool_call>" lookahead.  Walking the
        # body parameter-by-parameter with ``_find_true_param_end``
        # correctly steps over the literal.
        function_calls: list[str] = []
        for tool_call in raw_tool_calls:
            func_start = tool_call.find(self.tool_call_prefix)
            if func_start == -1:
                continue
            after_func_open = tool_call[func_start + len(self.tool_call_prefix) :]
            name_end = after_func_open.find(">")
            valid_param_names: set[str] | None = None
            body_start = 0
            if name_end != -1:
                func_name = after_func_open[:name_end]
                cfg = find_tool_properties(self.tools, func_name)
                if cfg:
                    valid_param_names = set(cfg.keys())
                body_start = name_end + 1
            scan_end = self._scan_to_structural_function_end(
                after_func_open[body_start:], valid_param_names
            )
            if scan_end != -1:
                function_calls.append(after_func_open[: body_start + scan_end])
                continue
            # Fallback to legacy heuristic.
            func_end = self._find_true_function_end(after_func_open)
            if func_end == -1:
                function_calls.append(after_func_open)
            else:
                function_calls.append(after_func_open[:func_end])
        return function_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Quick check to avoid unnecessary processing
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls.  Anchor at the FIRST
            # ``<tool_call>`` that contains a real ``<function=NAME>``
            # opener — a bare ``<tool_call>...</tool_call>`` written by
            # the model in its narrative text (no function inside) is
            # NOT a real tool call and the surrounding text MUST stay
            # in ``content``.
            content_index = -1
            search_pos = 0
            tc_start_token = self.tool_call_start_token
            tc_end_token = self.tool_call_end_token
            while True:
                tc_pos = model_output.find(tc_start_token, search_pos)
                if tc_pos == -1:
                    break
                tc_close = model_output.find(tc_end_token, tc_pos + len(tc_start_token))
                # Look for a ``<function=`` inside this tool_call block
                # (or up to end-of-string if the block isn't closed).
                limit = tc_close if tc_close != -1 else len(model_output)
                func_pos = model_output.find(
                    self.tool_call_prefix, tc_pos + len(tc_start_token), limit
                )
                if func_pos != -1:
                    content_index = tc_pos
                    break
                search_pos = tc_close + len(tc_end_token) if tc_close != -1 else limit
            if content_index == -1:
                # No structural ``<tool_call>`` block contains a
                # ``<function=``: fall back to the standalone
                # ``<function=`` position (legacy behaviour).
                content_index = model_output.find(self.tool_call_prefix)
            content = (
                model_output[:content_index] if content_index >= 0 else model_output
            )
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
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
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # Use structural </tool_call> count: a literal </tool_call>
            # embedded in a parameter value must not trigger spurious
            # advance.
            tool_ends = len(self._structural_tool_call_end_positions(current_text))
            if tool_ends > self.current_tool_index:
                # Advance to next tool; is_tool_call_started is reset so
                # content between or after tool calls is emitted correctly.
                # We deliberately fall through (no early ``return None``):
                # the rest of this delta may carry trailing free text after
                # the closed </tool_call> or even an entire next tool call
                # (MTP / speculative decoding). The downstream code handles
                # both — emitting trailing content via the not-started
                # branch, or starting the next tool via tool_starts_count.
                self._advance_to_next_tool(current_text)

        content_message = None
        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            tool_starts_count = current_text.count(self.tool_call_start_token)
            start_signal = (
                self.tool_call_start_token_id in delta_token_ids
                or tool_starts_count > self.current_tool_index
            )
            # ``tool_starts_count`` is naive and over-counts when an
            # earlier tool's parameter value contains a literal
            # ``<tool_call>``.  Confirm a REAL next tool by locating an
            # opener past ``_sent_content_idx`` (which sits after the last
            # processed tool's structural ``</tool_call>``).
            last_start = -1
            if start_signal:
                last_start = current_text.find(
                    self.tool_call_start_token, self._sent_content_idx
                )
            if start_signal and last_start != -1:
                self.is_tool_call_started = True
                # Return any content before the tool call
                if last_start > self._sent_content_idx:
                    content_before = current_text[self._sent_content_idx : last_start]
                    self._sent_content_idx = last_start
                    if content_before:
                        content_message = DeltaMessage(content=content_before)
            else:
                # No real new tool starting in this delta — emit any
                # trailing/inter-call content.
                overlap = partial_tag_overlap(current_text, self.tool_call_start_token)
                sendable_idx = len(current_text) - overlap

                # Skip whitespace-only deltas right after a closed tool.
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    self._sent_content_idx = len(current_text)
                    return None

                if sendable_idx > self._sent_content_idx:
                    content = current_text[self._sent_content_idx : sendable_idx]
                    self._sent_content_idx = sendable_idx
                    if content:
                        return DeltaMessage(content=content)
                return None

        # Check if we're between tool calls (waiting for next one).
        # Only count structural <tool_call> starts (skip past each
        # </tool_call> of completed calls) so that <tool_call> tokens
        # embedded in a parameter value of a completed call are not
        # counted as spurious new tool calls.
        if self.tool_call_start_token not in current_text[self._sent_content_idx :]:
            return content_message

        # We're in a tool call, find the current tool call portion.
        # Build tool_start_positions by jumping OVER completed tool
        # calls (past each </tool_call>), so that <tool_call> tokens
        # embedded in parameter values of completed calls are never
        # included.
        # Use STRUCTURAL </tool_call> positions when jumping past
        # completed tool calls — naive ``current_text.find(</tool_call>)``
        # matches a literal ``</tool_call>`` embedded in a parameter
        # value and would land inside an earlier tool's content.
        structural_ends = self._structural_tool_call_end_positions(current_text)
        tool_start_positions: list[int] = []
        search_pos = 0
        for i in range(self.current_tool_index + 1):
            idx = current_text.find(self.tool_call_start_token, search_pos)
            if idx == -1:
                break
            tool_start_positions.append(idx)
            if i < self.current_tool_index:
                # Completed tool call: jump past its STRUCTURAL </tool_call>.
                end_idx = -1
                for end_pos in structural_ends:
                    if end_pos > idx:
                        end_idx = end_pos
                        break
                if end_idx == -1:
                    break
                search_pos = end_idx + len(self.tool_call_end_token)

        if self.current_tool_index >= len(tool_start_positions):
            return content_message

        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find this tool call's STRUCTURAL end (or use rest of text if
        # the tool isn't closed yet).  A naive find would truncate at a
        # literal </tool_call> inside a parameter value.
        tool_end_idx = -1
        for end_pos in structural_ends:
            if end_pos > tool_start_idx:
                tool_end_idx = end_pos
                break
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        tool_call_fragments = None
        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": "{}",
                        }
                    )

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool.append("")

                    tool_call_fragments = DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(
                            name=self.current_function_name, arguments=""
                        ),
                        type="function",
                    )
            if not self.header_sent:
                return content_message

        arguments_to_emit = ""
        # We've sent header, now handle function body
        if self.in_function:
            # Always send opening brace first, regardless of whether
            # parameter_prefix is in the current delta. With speculative
            # decoding, a single delta may contain both the opening brace
            # and parameter data; skipping "{" here would desync
            # json_started from what was actually streamed.
            if not self.json_started:
                self.json_started = True
                self.streamed_args_for_tool[self.current_tool_index] += "{"
                arguments_to_emit += "{"

            # Build param_starts using structural-aware lookup. Plain
            # tool_text.find(parameter_prefix) would return positions
            # inside parameter VALUES (e.g. Python code that embeds the
            # XML format), creating spurious extra params.  Use the
            # schema to filter nested <parameter=NAME> and advance
            # sequentially past each complete parameter's value.
            streaming_param_config = find_tool_properties(
                self.tools, self.current_function_name or ""
            )
            valid_param_names: set[str] | None = (
                set(streaming_param_config.keys()) if streaming_param_config else None
            )
            param_starts: list[int] = []
            search_idx = 0
            while True:
                # Don't filter top-level <parameter=NAME> by schema:
                # callers may send params whose names aren't declared
                # (e.g. renamed fields).  Schema filtering is applied
                # below when walking INSIDE a parameter value to
                # disambiguate nested literal XML.
                param_start_pos = self._next_structural_param_start(
                    tool_text, search_idx, None
                )
                if param_start_pos == -1:
                    break
                param_starts.append(param_start_pos)
                # Advance past this parameter's content.
                name_end_pos = tool_text.find(
                    ">", param_start_pos + len(self.parameter_prefix)
                )
                if name_end_pos == -1:
                    break
                after_name = tool_text[name_end_pos + 1 :]
                after_name_stripped = (
                    after_name[1:] if after_name.startswith("\n") else after_name
                )
                end_in_after = self._find_true_param_end(
                    after_name_stripped,
                    valid_param_names,
                    require_lookahead=True,
                )
                if end_in_after == -1:
                    # No structural ``</parameter>`` close yet.  A
                    # legitimate "missing </parameter>" malformation —
                    # the model jumps from ``<parameter=A>`` straight to
                    # ``<parameter=B>`` — is recoverable: treat the
                    # next structural ``<parameter=NAME>`` as implicit
                    # end of the current param.  But only if NAME has
                    # NOT already been parsed as a sibling param of this
                    # tool call (and is not the param currently being
                    # scanned).  A repeated NAME is almost always a
                    # literal embedded in the unfinished value, not a
                    # real next parameter.
                    cand_name = tool_text[
                        param_start_pos + len(self.parameter_prefix) : name_end_pos
                    ]
                    already_seen = set(self.accumulated_params.keys()) | (
                        {cand_name} if cand_name else set()
                    )
                    unseen_valid: set[str] | None = (
                        (valid_param_names - already_seen)
                        if valid_param_names is not None
                        else None
                    )
                    implicit_end = self._next_structural_param_start(
                        after_name_stripped, 0, unseen_valid
                    )
                    if implicit_end != -1:
                        search_idx = (
                            (name_end_pos + 1)
                            + (1 if after_name.startswith("\n") else 0)
                            + implicit_end
                        )
                    else:
                        # Wait for more data.
                        break
                else:
                    search_idx = (
                        (name_end_pos + 1)
                        + (1 if after_name.startswith("\n") else 0)
                        + end_in_after
                        + len(self.parameter_end_token)
                    )

            # Process ALL complete params in a loop (spec decode fix).
            # With speculative decoding a single delta can deliver
            # multiple complete parameters at once. The old single-pass
            # code would process one and ``return None`` if the next was
            # incomplete — skipping any already-complete params that
            # preceded it. Using a loop with ``break`` instead ensures
            # we emit every complete parameter before yielding control.
            json_fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                param_end_idx = self._find_true_param_end(
                    value_text, valid_param_names, require_lookahead=True
                )
                if param_end_idx == -1:
                    # Confirm via the parameter-aware walker that the
                    # function body is truly complete.  The legacy
                    # ``_find_true_function_end`` matches a ``</function>``
                    # at end-of-buffer (lstripped lookahead == ""), which
                    # is wrong in streaming when the literal close of a
                    # nested tool_call inside a parameter value sits at
                    # the buffer's end.  Walking the body via
                    # ``_scan_to_structural_function_end`` correctly
                    # steps over literal tags inside parameter values
                    # and returns -1 if any param is still open.
                    tc_open_in_tool = tool_text.find(self.tool_call_prefix)
                    body_func_end_in_value = -1
                    if tc_open_in_tool != -1:
                        name_end_in_tool = tool_text.find(
                            ">", tc_open_in_tool + len(self.tool_call_prefix)
                        )
                        if name_end_in_tool != -1:
                            body_after_name = tool_text[name_end_in_tool + 1 :]
                            body_func_end_rel = self._scan_to_structural_function_end(
                                body_after_name, valid_param_names
                            )
                            if body_func_end_rel != -1:
                                body_func_end_abs = (
                                    name_end_in_tool + 1 + body_func_end_rel
                                )
                                body_func_end_in_value = body_func_end_abs - value_start

                    if body_func_end_in_value > 0:
                        # Function body is structurally complete; the
                        # current param has missing </parameter>.  Use
                        # the next legitimate <parameter=NAME> (NAME
                        # unseen) before the structural </function> as
                        # the implicit end.
                        already_seen = set(self.accumulated_params.keys()) | (
                            {current_param_name} if current_param_name else set()
                        )
                        unseen_valid: set[str] | None = (
                            (valid_param_names - already_seen)
                            if valid_param_names is not None
                            else None
                        )
                        next_param_idx = self._next_structural_param_start(
                            value_text, 0, unseen_valid
                        )
                        if (
                            next_param_idx != -1
                            and next_param_idx < body_func_end_in_value
                        ):
                            param_end_idx = next_param_idx
                        else:
                            param_end_idx = body_func_end_in_value
                    else:
                        # Body not yet complete — wait for more data.
                        # Do NOT truncate at a literal </function> or
                        # </tool_call> that may sit inside a still-open
                        # parameter value.
                        break

                if param_end_idx == -1:
                    break

                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value

                param_config = find_tool_properties(
                    self.tools, self.current_function_name or ""
                )

                converted_value = self._convert_param_value(
                    param_value,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )

                serialized_value = json.dumps(converted_value, ensure_ascii=False)

                if self.param_count == 0:
                    json_fragment = f'"{current_param_name}": {serialized_value}'
                else:
                    json_fragment = f', "{current_param_name}": {serialized_value}'

                self.param_count += 1
                json_fragments.append(json_fragment)

            if json_fragments:
                combined = "".join(json_fragments)

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )
                arguments_to_emit += combined

            # Check for function end AFTER processing parameters.
            # This ordering is critical: with speculative decoding a
            # burst can deliver the final parameter value together with
            # </function>. If the close check ran first it would emit
            # "}" and set in_function=False before the parameter loop
            # ever ran, causing the parameter to be silently dropped.
            # Use the parameter-aware walker so a literal '</function>'
            # inside a parameter value (e.g. a content arg embedding a
            # complete nested tool_call) does not trigger a premature
            # close.
            true_func_end = -1
            tc_open_in_tool_for_close = tool_text.find(self.tool_call_prefix)
            if tc_open_in_tool_for_close != -1:
                name_end_in_tool = tool_text.find(
                    ">",
                    tc_open_in_tool_for_close + len(self.tool_call_prefix),
                )
                if name_end_in_tool != -1:
                    body_after_name = tool_text[name_end_in_tool + 1 :]
                    body_func_end_rel = self._scan_to_structural_function_end(
                        body_after_name, valid_param_names
                    )
                    if body_func_end_rel != -1:
                        true_func_end = name_end_in_tool + 1 + body_func_end_rel
            if not self.json_closed and true_func_end != -1:
                self.json_closed = True

                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = true_func_end
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                        )
                        if parsed_tool and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = parsed_tool.function.arguments
                    except Exception:
                        logger.debug(
                            "Failed to parse tool call during streaming: %s",
                            tool_text,
                            exc_info=True,
                        )

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += "}"
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )
                arguments_to_emit += "}"
                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

        if tool_call_fragments or arguments_to_emit:
            if not tool_call_fragments:
                tool_call_fragments = DeltaToolCall(
                    index=self.current_tool_index,
                    function=DeltaFunctionCall(arguments=arguments_to_emit),
                )
            else:
                tool_call_fragments.function.arguments += arguments_to_emit

            if content_message:
                content_message.tool_calls = [tool_call_fragments]
                result = content_message
            else:
                result = DeltaMessage(tool_calls=[tool_call_fragments])

            # Speculative decoding can deliver multiple complete tool
            # calls in a single delta.  If we just finished one and
            # another complete <tool_call>...</tool_call> remains in
            # current_text, advance and re-enter to emit it.  We pass a
            # non-empty `previous_text` sentinel so reset_streaming_state
            # is NOT triggered inside the recursion (which would clear
            # current_tool_index back to 0 and loop forever).
            if (
                self.json_closed
                and not self.in_function
                and len(self._structural_tool_call_end_positions(current_text))
                > self.current_tool_index + 1
            ):
                # Speculative decoding delivered multiple complete tool
                # calls in one delta; advance and recurse for the next.
                self._advance_to_next_tool(current_text)

                # Recurse with a sentinel previous_text so the entry
                # check `if not previous_text` does NOT reset the state.
                next_delta = self.extract_tool_calls_streaming(
                    previous_text or " ",
                    current_text,
                    delta_text,
                    previous_token_ids,
                    current_token_ids,
                    delta_token_ids,
                    request,
                )
                if next_delta is not None and next_delta.tool_calls:
                    if result.tool_calls is None:
                        result.tool_calls = []
                    result.tool_calls.extend(next_delta.tool_calls)
                    # Concatenate the recursion's content (e.g. text
                    # BETWEEN tool 1 and tool 2) with the outer's content
                    # (e.g. text BEFORE tool 1). Without this, the "between"
                    # fragment is silently dropped whenever the outer
                    # already produced its own content.
                    if next_delta.content:
                        result.content = (result.content or "") + next_delta.content

            # Emit trailing free text that follows the LAST structural
            # </tool_call> in this delta (MTP / spec-decoding bursts that
            # bundle N tool calls + trailing content into one chunk).
            # Without this the trailing text is buffered indefinitely:
            # the per-tool processing never advances ``_sent_content_idx``
            # past its tool's ``</tool_call>``, and an EOS-style empty
            # delta cannot recover content that was never emitted.
            if self.json_closed and not self.in_function:
                end_positions = self._structural_tool_call_end_positions(current_text)
                if end_positions:
                    last_end = end_positions[-1] + len(self.tool_call_end_token)
                    if (
                        last_end < len(current_text)
                        and last_end > self._sent_content_idx
                    ):
                        trailing = current_text[last_end:]
                        if trailing:
                            self._sent_content_idx = len(current_text)
                            result.content = (result.content or "") + trailing
            return result

        return content_message

    def get_structural_tag(self, request: ChatCompletionRequest):
        return get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
