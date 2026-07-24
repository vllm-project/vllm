# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow
from transformers import PreTrainedTokenizerBase

import vllm.envs as envs
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
from vllm.tool_parsers.utils import (
    UnexpectedAstError,
    compute_tool_delta,
    find_common_prefix,
    handle_single_tool,
    is_complete_json,
    make_valid_python,
    partial_json_loads,
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
            # Some Llama-4 checkpoints (e.g. served with the Llama-3.1 JSON
            # chat template rather than the pythonic one) emit tool calls as
            # JSON instead of the pythonic form this parser otherwise expects
            # (see issue #46863). Fall back to a conservative JSON parse
            # before giving up on the response.
            json_tool_calls = self._extract_json_tool_calls(model_output)
            if json_tool_calls is not None:
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=json_tool_calls, content=None
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
        """Build a ToolCall from a single JSON object, or None if it doesn't
        look like one. A tool call needs a non-empty string ``name`` plus a
        dict of ``arguments`` or ``parameters``; requiring the args key keeps
        ordinary JSON content that merely has a ``name`` field (e.g.
        ``{"name": "Alice", "age": 30}``) from being mistaken for a call."""
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
                name=name, arguments=json.dumps(arguments, ensure_ascii=False)
            ),
        )

    @classmethod
    def _extract_json_tool_calls(cls, model_output: str) -> list[ToolCall] | None:
        """Best-effort JSON fallback for a complete response: accepts a
        single JSON object, several comma-separated objects, or a JSON array
        of objects. The whole response must parse as one clean tool-call
        payload; anything else (malformed JSON, trailing prose, an ordinary
        JSON object with no ``name``/args) is left for the caller to treat as
        plain content, mirroring how the pythonic branch above requires
        ``ast.parse`` to consume the entire response."""
        if "{" not in model_output:
            return None
        stripped = model_output.strip()
        candidate = stripped if stripped.startswith("[") else f"[{stripped}]"
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list) or not parsed:
            return None
        tool_calls: list[ToolCall] = []
        for obj in parsed:
            tool_call = cls._json_obj_to_tool_call(obj)
            if tool_call is None:
                return None
            tool_calls.append(tool_call)
        return tool_calls

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
        has_python_tag = current_text.startswith("<|python_start|>")
        if (
            not has_python_tag
            and not current_text.startswith("[")
            and not current_text.startswith("{")
        ):
            return DeltaMessage(content=delta_text)

        # Some Llama-4 checkpoints emit JSON tool calls instead of the
        # pythonic form (see issue #46863). A bare JSON object is
        # unambiguous; a leading "[" is ambiguous between a pythonic call
        # list and a JSON array of tool calls, so peek at what follows the
        # bracket once enough text has arrived to tell the two apart.
        text_after_tag = (
            current_text[len("<|python_start|>") :] if has_python_tag else current_text
        )
        is_json = text_after_tag.startswith("{")
        if not is_json and text_after_tag.startswith("["):
            is_json = text_after_tag[1:].lstrip().startswith("{")
        if is_json:
            return self._extract_json_tool_calls_streaming(text_after_tag)

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

    def _extract_json_tool_calls_streaming(self, text: str) -> DeltaMessage | None:
        """Streaming counterpart to ``_extract_json_tool_calls``.

        Incrementally parses one or more JSON tool-call objects (bare,
        comma-separated, or wrapped in an array) out of the growing ``text``
        and emits deltas the same way Llama3JsonToolParser streams the JSON
        tool calls it recognizes: the function name first, then argument
        characters as they become available.
        """
        try:
            if text.endswith("<|python_end|>"):
                text = text[: text.rfind("<|python_end|>")]
            text = text.strip()
            if text.startswith("["):
                text = text[1:]
                if text.endswith("]"):
                    text = text[:-1]

            # Only hand out a partially streamed string once the function
            # name has already been sent; the API only ever streams a
            # complete name.
            flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

            tool_call_arr: list[dict] = []
            is_complete: list[bool] = []
            idx = 0
            try:
                while idx < len(text):
                    while idx < len(text) and text[idx] in " \t\r\n,":
                        idx += 1
                    if idx >= len(text):
                        break
                    obj, end = partial_json_loads(text[idx:], flags)
                    if not isinstance(obj, dict):
                        return None
                    is_complete.append(is_complete_json(text[idx : idx + end]))
                    idx += end
                    if "parameters" in obj and "arguments" not in obj:
                        obj["arguments"] = obj["parameters"]
                    tool_call_arr.append(obj)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse JSON tool call yet")
                return None

            if not tool_call_arr:
                return None

            current_tool_call = tool_call_arr[self.current_tool_id]

            # Starting a new tool in the array: flush whatever arguments were
            # auto-completed for the previous one but never streamed, then
            # move the cursor on.
            if len(tool_call_arr) > self.current_tool_id + 1:
                delta = None
                if self.current_tool_id >= 0:
                    cur_arguments = current_tool_call.get("arguments")
                    if cur_arguments:
                        cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                        sent = len(self.streamed_args_for_tool[self.current_tool_id])
                        flush_diff = cur_args_json[sent:]
                        delta = DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    function=DeltaFunctionCall(arguments=flush_diff),
                                )
                            ]
                        )
                        self.streamed_args_for_tool[self.current_tool_id] += flush_diff
                self.current_tool_id = len(tool_call_arr) - 1
                self.current_tool_name_sent = False
                # Usually a single new tool call surfaces per delta, but pad
                # up to the cursor in case a single large delta completed
                # more than one at once.
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                return delta

            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                has_args_key = (
                    "arguments" in current_tool_call
                    or "parameters" in current_tool_call
                )
                # Wait for both the name and an arguments/parameters key
                # before committing to a tool call, so an ordinary JSON
                # object that merely has a "name" field (e.g. {"name":
                # "Alice", "age": 30}) is never reported as one.
                if not function_name or not has_args_key:
                    return None
                self.current_tool_name_sent = True
                self.prev_tool_call_arr = tool_call_arr
                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            type="function",
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(name=function_name),
                        )
                    ]
                )

            delta = None
            cur_arguments = current_tool_call.get("arguments")
            if cur_arguments:
                sent = len(self.streamed_args_for_tool[self.current_tool_id])
                cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                    "arguments"
                )

                argument_diff: str | None = None
                if is_complete[self.current_tool_id]:
                    argument_diff = cur_args_json[sent:]
                elif prev_arguments:
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    if cur_args_json != prev_args_json:
                        argument_diff = find_common_prefix(
                            prev_args_json, cur_args_json
                        )[sent:]

                if argument_diff:
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=argument_diff),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            self.prev_tool_call_arr = tool_call_arr
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming JSON tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None
