# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
from collections.abc import Sequence
from typing import Any

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
    ToolParser,
)
from vllm.tool_parsers.utils import (
    _UnexpectedAstError,
    compute_tool_delta,
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

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

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
                        _handle_single_tool(e)  # type: ignore
                        for e in parsed.elts
                    ],
                    content=None,
                )
            else:
                raise _UnexpectedAstError(
                    "Tool output must be a list of function calls"
                )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            # Treat as regular text
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
                raise _UnexpectedAstError(
                    "Tool output must be a list of function calls"
                )
            tool_calls = [
                _handle_single_tool(e)  # type: ignore
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


def _get_parameter_value(val: ast.expr) -> Any:
    if isinstance(val, ast.Constant):
        return val.value
    elif isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            raise _UnexpectedAstError("Dict tool call arguments must have literal keys")
        return {
            k.value: _get_parameter_value(v)  # type: ignore
            for k, v in zip(val.keys, val.values)
        }
    elif isinstance(val, ast.List):
        return [_get_parameter_value(v) for v in val.elts]
    else:
        raise _UnexpectedAstError("Tool call arguments must be literals")


def _handle_single_tool(call: ast.Call) -> ToolCall:
    if not isinstance(call.func, ast.Name):
        raise _UnexpectedAstError("Invalid tool call name")
    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = _get_parameter_value(keyword.value)
    return ToolCall(
        type="function",
        function=FunctionCall(name=function_name, arguments=json.dumps(arguments)),
    )
