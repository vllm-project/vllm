# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GranitePythonicToolParser
=========================
Handles the **Python-style** tool-call format produced by Granite 3.3 and
Granite 4.0 H-Small when served through vLLM's OpenAI-compatible endpoint.

Granite models in this family emit tool invocations as plain Python
function calls::

    get_weather(location="San Francisco", unit="celsius")
    search_web(query="vLLM release notes")

This parser detects those calls, converts the keyword arguments to a JSON
string, and returns an ``ExtractedToolCallInformation`` / ``DeltaMessage``
with a properly-formed ``tool_calls`` list so that any OpenAI-compatible
client can consume the response without modification.

Usage::

    vllm serve ibm-granite/granite-3.3-8b-instruct \\
        --tool-call-parser granite_pythonic \\
        --chat-template examples/tool_chat_template_granite.jinja
"""

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
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
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches a single complete Python-style function call at the start of a line.
# Group 1 -> function name  (e.g. "get_weather")
# Group 2 -> raw argument string inside the parentheses
_CALL_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$",
    re.DOTALL,
)

# Lookahead: a line *looks* like it might be a function call (partial match
# during streaming -- we see the opening paren but not the closing one yet).
_PARTIAL_CALL_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\(",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_kwargs(raw_args: str) -> dict[str, Any]:
    """Parse keyword arguments string into a dict.

    Granite 3.3 / 4.0 always emits *keyword* arguments, so we wrap the raw
    string in a synthetic function definition and evaluate it with
    ``ast.parse`` to avoid ``eval()``.

    Example::

        >>> _parse_kwargs('location="San Francisco", unit="celsius"')
        {'location': 'San Francisco', 'unit': 'celsius'}
    """
    if not raw_args.strip():
        return {}
    try:
        # Build a dummy call: f(<raw_args>) and extract the keyword nodes.
        tree = ast.parse(f"_f({raw_args})", mode="eval")
        call_node = tree.body  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for keyword in call_node.keywords:
            if keyword.arg is None:
                # **kwargs spread -- skip; not expected from Granite
                continue
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        return kwargs
    except Exception:
        logger.debug(
            "GranitePythonicToolParser: could not parse kwargs %r", raw_args
        )
        return {"_raw": raw_args}


def _try_parse_call(line: str) -> tuple[str, str] | None:
    """Return (function_name, json_arguments) if *line* is a complete call.

    Returns ``None`` if the line does not match.
    """
    m = _CALL_RE.match(line.strip())
    if m is None:
        return None
    name = m.group(1)
    raw_args = m.group(2)
    kwargs = _parse_kwargs(raw_args)
    return name, json.dumps(kwargs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------

class GranitePythonicToolParser(ToolParser):
    """Tool-call parser for Granite 3.3 and Granite 4.0 H-Small.

    These models produce Python-style function calls as plain text::

        get_weather(location="Paris", unit="celsius")

    The parser converts each such call into an OpenAI ``ToolCall`` object
    with ``type="function"`` and a JSON-serialised ``arguments`` string.

    Multiple consecutive tool calls (one per line) are supported.
    Lines that do not look like function calls are returned as plain
    ``content``.
    """

    def __init__(
        self,
        tokenizer,  # TokenizerLike -- kept untyped to avoid circular import
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        # _stream_buffer holds incomplete lines during streaming.
        # Base class provides: current_tool_id, streamed_args_for_tool,
        # prev_tool_call_arr -- we use those directly instead of private
        # duplicates so the rest of the vLLM serving layer stays in sync.
        self._stream_buffer: str = ""

    # ------------------------------------------------------------------
    # Batch (non-streaming) extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Parse a complete (non-streaming) model response.

        This method is intentionally **stateless**: it does not mutate any
        instance attributes so it is safe to call multiple times on the
        same parser instance without side-effects.
        """
        result = ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=model_output,
        )
        try:
            tool_calls: list[ToolCall] = []
            text_lines: list[str] = []

            for raw_line in model_output.splitlines():
                parsed = _try_parse_call(raw_line)
                if parsed is not None:
                    func_name, json_args = parsed
                    tc = ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json_args,
                        ),
                    )
                    tool_calls.append(tc)
                    # Do NOT mutate self.prev_tool_call_arr here -- batch
                    # extraction is stateless; streaming state is managed
                    # separately in _process_line.
                else:
                    text_lines.append(raw_line)

            remaining_text = "\n".join(text_lines).strip() or None

            result.tools_called = bool(tool_calls)
            result.tool_calls = tool_calls
            result.content = remaining_text
        except Exception:
            logger.exception(
                "GranitePythonicToolParser: error extracting tool calls."
            )
        return result

    # ------------------------------------------------------------------
    # Streaming extraction
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
    ) -> DeltaMessage | None:
        """Incrementally parse streaming model output.

        Strategy
        --------
        We accumulate text in ``_stream_buffer`` until we see a complete
        line (ending with ``\\n``) or the stream ends.  Each complete line
        is tested against ``_CALL_RE``:

        * **Match** -> emit a ``DeltaToolCall``.
        * **No match** -> emit the line as ``content``.

        A *partial* line that looks like the start of a function call
        (``name(``) is held in the buffer so we don't accidentally emit
        it as plain text before the closing ``)`` arrives.

        If the buffer holds a **complete** call without a trailing newline
        (i.e. the model finished generating without appending ``\\n``),
        it is flushed immediately rather than waiting forever.
        """
        try:
            self._stream_buffer += delta_text

            content_parts: list[str] = []
            delta_tool_calls: list[DeltaToolCall] = []

            # Process complete newline-terminated lines first.
            while "\n" in self._stream_buffer:
                line, self._stream_buffer = self._stream_buffer.split("\n", 1)
                self._process_line(line, content_parts, delta_tool_calls)

            # Flush the remaining buffer when:
            #   (a) it is already a complete, parseable tool call, OR
            #   (b) it cannot possibly be the start of a call.
            # This handles models that finish a tool call without a trailing
            # newline (e.g. get_weather(city="London") at EOS).
            remainder = self._stream_buffer
            if remainder:
                if (_try_parse_call(remainder) is not None
                        or not _PARTIAL_CALL_RE.match(remainder.strip())):
                    self._process_line(remainder, content_parts, delta_tool_calls)
                    self._stream_buffer = ""

            content = "".join(content_parts) or None
            msg = DeltaMessage(
                content=content,
                tool_calls=delta_tool_calls if delta_tool_calls else None,
            )
            if msg.content or msg.tool_calls:
                return msg
        except Exception:
            logger.exception(
                "GranitePythonicToolParser: error during streaming extraction."
            )
        return None

    def _process_line(
        self,
        line: str,
        content_parts: list[str],
        delta_tool_calls: list[DeltaToolCall],
    ) -> None:
        """Classify *line* and append to the appropriate output list.

        Uses base-class attributes (``current_tool_id``,
        ``streamed_args_for_tool``, ``prev_tool_call_arr``) so the vLLM
        serving layer remains consistent.
        """
        parsed = _try_parse_call(line)
        if parsed is not None:
            func_name, json_args = parsed
            self.current_tool_id += 1
            self.streamed_args_for_tool.append(json_args)
            self.prev_tool_call_arr.append(
                {"name": func_name, "arguments": json_args}
            )
            delta_tool_calls.append(
                DeltaToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    index=self.current_tool_id,
                    function=DeltaFunctionCall(
                        name=func_name,
                        arguments=json_args,
                    ).model_dump(exclude_none=True),
                )
            )
        else:
            if line:  # skip empty lines to avoid spurious newlines in content
                content_parts.append(line + "\n")
