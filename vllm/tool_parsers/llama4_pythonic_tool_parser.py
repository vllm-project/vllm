# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence

import regex as re
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    compute_tool_delta,
    handle_single_tool,
    make_valid_python,
)

logger = init_logger(__name__)


class Llama4PythonicToolParser(ToolParser):
    """
    Toolcall parser for Llama4 that produce tool calls in a pythonic style
    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    # TODO(mdepinet): Possible future improvements:
    #   1. Support text + tools separated by either <|python_tag|> or \n\n
    #   2. Support tools outside of a list (or separated by a semicolon).
    #      This depends on item 1 for consistent streaming.
    # Neither of these are necessary for e.g. ToolACE, but both would help make
    # Llama3.2 models more reliable.

    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        tools: list[Tool] | None = None,
    ):
        super().__init__(tokenizer, tools)

    # Rename for readability. This is NOT a tool id.
    @property
    def current_tool_index(self) -> int:
        return self.current_tool_id

    @current_tool_index.setter
    def current_tool_index(self, value: int) -> None:
        self.current_tool_id = value

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract the tool calls from a complete model response.
        """

        # remove <|python_start|> and <|python_end|>
        # as Llama 4 model sometime will output those tokens
        if model_output.startswith("<|python_start|>"):
            model_output = model_output[len("<|python_start|>") :]
            model_output = model_output.replace("<|python_end|>", "")

        is_tool_call_pattern = False
        try:
            is_tool_call_pattern = (
                self.TOOL_CALL_REGEX.match(
                    model_output, timeout=envs.VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS
                )
                is not None
            )
        except TimeoutError:
            logger.warning("Regex timeout occurred when matching tool call pattern.")
            logger.debug(
                "Regex timeout occurred when matching user input: %s", model_output
            )

        if not is_tool_call_pattern:
            # The model did not emit a pythonic-style tool call. Some Llama-4
            # checkpoints (e.g. served with the Llama-3.1 JSON chat template)
            # instead emit JSON tool calls, which previously fell through to
            # plain content. Try a conservative JSON fallback before giving up.
            json_tool_calls = self._extract_json_tool_calls(model_output)
            if json_tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=json_tool_calls,
                    content=None,
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            module = ast.parse(model_output)
            parsed = getattr(module.body[0], "value", None)
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=[
                        handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=None,
                )
            else:
                raise UnexpectedAstError("Tool output must be a list of function calls")
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    @staticmethod
    def _json_obj_to_tool_call(obj: object) -> ToolCall | None:
        """Convert a single JSON object into a ToolCall, or None if it does
        not look like a tool call. A tool call must be a dict with a non-empty
        string ``name`` and an explicit ``arguments`` or ``parameters`` dict.
        Requiring the args key keeps ordinary JSON content that merely has a
        ``name`` field (e.g. ``{"name": "Alice", "age": 30}``) as content."""
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None
        if "arguments" in obj:
            arguments = obj["arguments"]
        elif "parameters" in obj:
            arguments = obj["parameters"]
        else:
            return None
        if not isinstance(arguments, dict):
            return None
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _scan_json_tool_calls(
        self, model_output: str
    ) -> tuple[list[ToolCall], str]:
        """Scan ``model_output`` for JSON tool calls.

        Supports a single JSON object, several whitespace/comma separated
        objects, and a single JSON array of objects, e.g.::

            {"name": "Bash", "parameters": {"command": "echo hi"}}
            {"name": "a", ...}, {"name": "b", ...}
            [{"name": "a", ...}, {"name": "b", ...}]

        Returns ``(tool_calls, status)``. ``status`` is ``"ok"`` only when the
        entire output is cleanly consumed as one or more tool-call objects;
        anything else (leading/trailing non-JSON text, malformed JSON, or a
        well-formed object that is not a tool call) yields ``"not_tool"`` so the
        caller treats the output as plain content.
        """
        decoder = json.JSONDecoder()
        text = model_output.strip()
        in_array = text.startswith("[")
        if in_array:
            text = text[1:]

        tool_calls: list[ToolCall] = []
        array_closed = False
        # A comma is only a valid separator immediately after an object, and it
        # must be followed by another object: ``need_separator`` marks "just
        # parsed an object", ``pending_object`` marks "a comma still owes us an
        # object". This rejects leading, doubled and dangling/trailing commas.
        need_separator = False
        pending_object = False
        i = 0
        length = len(text)
        while i < length:
            while i < length and text[i] in " \t\r\n":
                i += 1
            if i >= length:
                break
            char = text[i]
            if in_array and char == "]":
                if pending_object:
                    return ([], "not_tool")
                i += 1
                array_closed = True
                break
            if char == ",":
                if not need_separator:
                    return ([], "not_tool")
                need_separator = False
                pending_object = True
                i += 1
                continue
            if char != "{":
                return ([], "not_tool")
            try:
                obj, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                # Malformed or truncated JSON: stay conservative and treat the
                # whole response as plain content rather than dropping the tail.
                return ([], "not_tool")
            i = end
            tool_call = self._json_obj_to_tool_call(obj)
            if tool_call is None:
                return ([], "not_tool")
            tool_calls.append(tool_call)
            need_separator = True
            pending_object = False

        # An array that never closed is truncated/malformed -> plain content.
        if in_array and not array_closed:
            return ([], "not_tool")
        # A dangling comma owed another object -> malformed -> plain content.
        if pending_object:
            return ([], "not_tool")
        # Trailing non-whitespace after the JSON tool calls means this is not a
        # clean tool-call payload; treat the whole output as content.
        if text[i:].strip():
            return ([], "not_tool")
        if not tool_calls:
            return ([], "not_tool")
        return (tool_calls, "ok")

    def _extract_json_tool_calls(self, model_output: str) -> list[ToolCall] | None:
        """Best-effort extraction of JSON tool calls from a complete response.

        Returns the parsed tool calls, or ``None`` when the output is not a
        clean JSON tool call (so the caller leaves it as plain content)."""
        if "{" not in model_output:
            return None
        try:
            tool_calls, status = self._scan_json_tool_calls(model_output)
        except Exception:
            logger.exception("Error in extracting JSON tool call from response.")
            return None
        if status != "ok":
            return None
        return tool_calls or None

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
        if not current_text.startswith("[") and not current_text.startswith(
            "<|python_start|>"
        ):
            return DeltaMessage(content=delta_text)

        try:
            # remove <|python_start|> and <|python_end|>
            if current_text.startswith("<|python_start|>"):
                current_text = current_text[len("<|python_start|>") :]
            if current_text.endswith("<|python_end|>"):
                current_text = current_text[: current_text.rfind("<|python_end|>")]
            valid_and_added_text = make_valid_python(current_text)
            if valid_and_added_text is None:
                return None
            valid_text, added_text = valid_and_added_text

            module = ast.parse(valid_text)
            parsed = getattr(module.body[0], "value", None)
            if not isinstance(parsed, ast.List) or not all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                raise UnexpectedAstError("Tool output must be a list of function calls")
            tool_calls = [
                handle_single_tool(e)  # type: ignore
                for e in parsed.elts
            ]

            tool_deltas = []
            for index, new_call in enumerate(tool_calls):
                if index < self.current_tool_index:
                    continue

                self.current_tool_index = index
                if len(self.streamed_args_for_tool) == index:
                    self.streamed_args_for_tool.append("")

                new_call_complete = (
                    index < len(tool_calls) - 1 or ")]" not in added_text
                )
                if new_call_complete:
                    self.current_tool_index += 1

                withheld_suffix = added_text[:-2] if not new_call_complete else ""
                if not new_call_complete and added_text[-2] == ")":
                    # Function call is incomplete. Withhold the closing bracket.
                    withheld_suffix = withheld_suffix + "}"
                # Strings get single quotes in the model-produced string.
                # JSON requires double quotes.
                withheld_suffix = withheld_suffix.replace("'", '"')
                delta = compute_tool_delta(
                    self.streamed_args_for_tool[index], new_call, index, withheld_suffix
                )

                if delta is not None:
                    tool_deltas.append(delta)
                    if (
                        delta.function is not None
                        and delta.function.arguments is not None
                    ):
                        self.streamed_args_for_tool[index] += delta.function.arguments

            # HACK: serving_chat.py inspects the internal state of tool parsers
            # when determining its final streaming delta, automatically
            # adding autocompleted JSON.
            # These two lines avoid that nonsense while ensuring finish_reason
            # is set to tool_calls when at least one tool is called.
            if tool_deltas and not self.prev_tool_call_arr:
                self.prev_tool_call_arr = [{"arguments": {}}]

            if tool_deltas:
                return DeltaMessage(tool_calls=tool_deltas)
            elif not added_text and self.current_tool_id > 0:
                # Return an empty DeltaMessage once the tool calls are all done
                # so that finish_reason gets set.
                return DeltaMessage(content="")
            else:
                return None
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
