# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.rust_tool_parser import RustToolParser

logger = init_logger(__name__)

# The MiniMax M3 namespace prefix that must precede every structural tag.
_NS = "]<]minimax[>["
_TOOL_CALL_OPEN = _NS + "<tool_call>"
_TOOL_CALL_CLOSE = _NS + "</tool_call>"
_NS_ESC = re.escape(_NS)

# Streaming: normalise any tag that is missing the NS prefix.
_MISSING_NS_RE = re.compile(
    r"(?<!" + _NS_ESC + r")(</?[A-Za-z_][A-Za-z0-9_.:-]*(?:\s[^>]*)?>)"
)

_NS_INVOKE_OPEN_RE = re.compile(_NS_ESC + r'<invoke name="([^"]+)">')
_NS_INVOKE_CLOSE = _NS + "</invoke>"
_NS_OPEN_TAG_RE = re.compile(_NS_ESC + r"<([A-Za-z_][A-Za-z0-9_-]*)>")
_NS_CLOSE_TAG_RE = re.compile(_NS_ESC + r"</([A-Za-z_][A-Za-z0-9_-]*)>")

_ITEM_OPEN_RE = re.compile(r"<item\b[^>]*>")
_ITEM_CLOSE = "</item>"


def _extract_top_level_items(s: str) -> list[str] | None:
    """Return the content of each top-level <item> element in *s*.

    Handles nesting: <item><item>100</item><item>150</item></item> correctly
    extracts the outer item's body (<item>100</item><item>150</item>).
    Returns None if no <item> tags are found.
    """
    results = []
    pos = 0
    while pos < len(s):
        m = _ITEM_OPEN_RE.search(s, pos)
        if m is None:
            break
        depth = 1
        inner_start = m.end()
        scan = inner_start
        while depth > 0 and scan < len(s):
            next_open = _ITEM_OPEN_RE.search(s, scan)
            next_close = s.find(_ITEM_CLOSE, scan)
            if next_close == -1:
                scan = len(s)
                break
            if next_open is not None and next_open.start() < next_close:
                depth += 1
                scan = next_open.end()
            else:
                depth -= 1
                if depth == 0:
                    results.append(s[inner_start:next_close])
                    pos = next_close + len(_ITEM_CLOSE)
                else:
                    scan = next_close + len(_ITEM_CLOSE)
        else:
            pos = scan
    return results if results else None


def _coerce_param(value: str, schema: dict | None) -> Any:
    """Type-coerce a raw string parameter value using its JSON Schema type."""
    if schema is None:
        return value
    t = schema.get("type")
    if t == "integer":
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if t == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if t == "boolean":
        return value.strip().lower() in ("true", "1", "yes")
    if t in ("array", "object"):
        v = value.strip()
        if not v:
            return [] if t == "array" else {}
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
        if t == "array":
            items = _extract_top_level_items(v)
            if items is not None:
                item_schema = schema.get("items")
                return [_coerce_param(item.strip(), item_schema) for item in items]
        return v
    return value


def _python_extract_tool_calls(
    model_output: str,
    tools_by_name: dict[str, dict],
) -> ExtractedToolCallInformation | None:
    """Parse MiniMax M3 tool-call XML entirely in Python.

    The MiniMax M3 model wraps every structural tag with the ``]<]minimax[>[``
    namespace prefix.  This pure-Python implementation handles all the
    malformed-output variants that defeat the Rust parser:

    * Unclosed parameter tags before ``</invoke>``
    * Missing namespace prefixes on inner tags (normalised before parsing)
    * Truncated outputs (partial tool calls are parsed as far as possible)
    * Mixed text + tool-call output

    Returns ``None`` if no tool-call start token is found.
    """
    tc_start = model_output.find(_TOOL_CALL_OPEN)
    if tc_start == -1:
        return None

    tc_close = model_output.find(_TOOL_CALL_CLOSE, tc_start + len(_TOOL_CALL_OPEN))
    tc_body = model_output[
        tc_start + len(_TOOL_CALL_OPEN) : tc_close if tc_close != -1 else len(model_output)
    ]

    tool_calls: list[ToolCall] = []
    pos = 0
    while pos < len(tc_body):
        m = _NS_INVOKE_OPEN_RE.search(tc_body, pos)
        if m is None:
            break
        tool_name = m.group(1)

        inv_close_pos = tc_body.find(_NS_INVOKE_CLOSE, m.end())
        inv_body = tc_body[m.end() : inv_close_pos if inv_close_pos != -1 else len(tc_body)]

        schema_props: dict = {}
        if tool_name in tools_by_name:
            schema_props = tools_by_name[tool_name].get("properties") or {}

        params: dict[str, Any] = {}
        ppos = 0
        while ppos < len(inv_body):
            om = _NS_OPEN_TAG_RE.search(inv_body, ppos)
            if om is None:
                break
            pname = om.group(1)
            # Find the matching close tag — next occurrence of </pname>
            close_pattern = re.compile(_NS_ESC + re.escape("</" + pname + ">"))
            cm = close_pattern.search(inv_body, om.end())
            if cm:
                raw_val = inv_body[om.end() : cm.start()]
                ppos = cm.end()
            else:
                # No closing tag: value runs to the next opening tag or end.
                next_open = _NS_OPEN_TAG_RE.search(inv_body, om.end())
                raw_val = inv_body[om.end() : next_open.start() if next_open else len(inv_body)]
                ppos = next_open.start() if next_open else len(inv_body)

            # Strip any residual NS prefixes embedded in the value (nested XML).
            raw_val = raw_val.replace(_NS, "").strip()
            params[pname] = _coerce_param(raw_val, schema_props.get(pname))

        tool_calls.append(
            ToolCall(
                id=make_tool_call_id(),
                type="function",
                function=FunctionCall(name=tool_name, arguments=json.dumps(params)),
            )
        )
        pos = inv_close_pos + len(_NS_INVOKE_CLOSE) if inv_close_pos != -1 else len(tc_body)

    if not tool_calls:
        return None

    normal_text = model_output[:tc_start].strip() or None
    return ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=tool_calls,
        content=normal_text,
    )


class MinimaxM3ToolParser(RustToolParser):
    """Adapter from the Rust MiniMax M3 parser to vLLM ToolParser.

    For complete (non-streaming) output: uses a pure-Python primary parser
    that tolerates all the malformed-output variants the model generates
    (missing NS prefix, unclosed param tags, truncated output).  Falls back
    to the Rust parser only if Python parsing yields nothing.

    For streaming output: delegates to the Rust incremental parser after
    normalising any missing NS prefixes on the incoming delta.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = _TOOL_CALL_OPEN

    def _tools_by_name(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if not self.tools:
            return result
        for tool in self.tools:
            try:
                name = tool.function.name
                params = tool.function.parameters or {}
            except AttributeError:
                try:
                    name = tool.name
                    params = getattr(tool, "parameters", {}) or {}
                except AttributeError:
                    continue
            result[name] = params
        return result

    # ------------------------------------------------------------------
    # Non-streaming: pure-Python primary path
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if _TOOL_CALL_OPEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Primary: pure-Python parser (handles all model output variants).
        py_result = _python_extract_tool_calls(model_output, self._tools_by_name())
        if py_result is not None:
            return py_result

        # Fallback: Rust parser (may handle edge cases the Python parser misses).
        parse_result = self._parse_complete(model_output)
        if parse_result is None:
            logger.warning(
                "MinimaxM3ToolParser: both Python and Rust parsers failed. "
                "Returning raw output as content."
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        parsed, tool_call_ids = parse_result
        tool_calls: list[ToolCall] = []
        self.prev_tool_call_arr.clear()
        for ptc in parsed.calls:
            name = ptc.name
            arguments = ptc.arguments or "{}"
            if name is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=tool_call_ids.get(ptc.tool_index) or make_tool_call_id(),
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=parsed.normal_text or None,
        )

    # ------------------------------------------------------------------
    # Streaming: Rust incremental parser with NS normalisation
    # ------------------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ):
        if _TOOL_CALL_OPEN in current_text:
            delta_text = _MISSING_NS_RE.sub(_NS + r"\1", delta_text)

        return super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
