# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from openai.types.responses import FunctionTool
from partial_json_parser.core.options import Allow

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
from vllm.tool_parsers.internlm2_tool_parser import Internlm2ToolParser
from vllm.tool_parsers.utils import extract_intermediate_diff

logger = init_logger(__name__)

_ACTION_START = "<|action_start|>"
_PLUGIN = "<|plugin|>"
_ACTION_END = "<|action_end|>"

_ACTION_PREFIX_RE = re.compile(
    rf"{re.escape(_ACTION_START)}\s*{re.escape(_PLUGIN)}\s*",
    re.DOTALL,
)
_ACTION_BLOCK_RE = re.compile(
    rf"{re.escape(_ACTION_START)}\s*{re.escape(_PLUGIN)}\s*(.*?)\s*"
    rf"{re.escape(_ACTION_END)}",
    re.DOTALL,
)


def _find_action_prefix(text: str, start: int = 0) -> re.Match[str] | None:
    return _ACTION_PREFIX_RE.search(text, start)


def _extract_json_payload(text: str) -> str | None:
    json_start = text.find("{")
    if json_start == -1:
        return None
    return text[json_start:].strip()


def _build_initial_arguments_delta(
    current_arguments_json: str,
    delta_text: str,
) -> str:
    try:
        end_index = current_arguments_json.index(delta_text) + len(delta_text)
        return current_arguments_json[:end_index]
    except ValueError:
        return current_arguments_json


class InternS1ToolParser(Internlm2ToolParser):
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
        if _ACTION_START not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)

        if self.current_tool_id > 0:
            return DeltaMessage(content="")

        last_pos = self.position
        action_prefix = _find_action_prefix(current_text, last_pos)
        if action_prefix is None:
            return None

        text = current_text[last_pos : action_prefix.start()]
        if text:
            self.position = action_prefix.start()
            return DeltaMessage(content=text)

        action = current_text[action_prefix.end() :]
        action_end_idx = action.find(_ACTION_END)
        if action_end_idx != -1:
            action = action[:action_end_idx]
        payload = _extract_json_payload(action)
        if payload is None:
            return None

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_arr: dict = partial_json_parser.loads(payload, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            if not self.current_tool_name_sent:
                function_name = tool_call_arr.get("name")
                if function_name:
                    self.current_tool_id += 1
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append("")
                else:
                    delta = None
            else:
                prev_arguments = self.get_arguments(
                    self.prev_tool_call_arr[self.current_tool_id]
                )
                cur_arguments = self.get_arguments(tool_call_arr)

                if not cur_arguments and not prev_arguments:
                    delta = None
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset mid-arguments"
                    )
                    delta = None
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)
                    arguments_delta = _build_initial_arguments_delta(
                        cur_arguments_json,
                        delta_text,
                    )
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=arguments_delta
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                else:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
                    argument_diff = extract_intermediate_diff(
                        cur_args_json,
                        prev_args_json,
                    )
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=argument_diff
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            tool_call_arr["arguments"] = self.get_arguments(tool_call_arr)
            self.prev_tool_call_arr = [tool_call_arr]
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        action_blocks = list(_ACTION_BLOCK_RE.finditer(model_output))
        if not action_blocks:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        configured_tools = self.tools or list(request.tools or [])
        allowed_tools = {
            tool.name if isinstance(tool, FunctionTool) else tool.function.name
            for tool in configured_tools
        }
        if not allowed_tools:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        cursor = 0

        for match in action_blocks:
            content_parts.append(model_output[cursor : match.start()])
            cursor = match.end()

            payload = _extract_json_payload(match.group(1))
            if payload is None:
                logger.warning("Intern-S1 action block missing JSON payload.")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            try:
                action_dict = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Failed to parse Intern-S1 tool call JSON: %r", payload)
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            name = action_dict.get("name")
            if not isinstance(name, str) or not name:
                logger.warning(
                    "Intern-S1 action block missing a valid tool name: %r",
                    action_dict,
                )
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            if name not in allowed_tools:
                logger.warning(
                    "Model requested tool %r which is not in the provided tools %r.",
                    name,
                    sorted(allowed_tools),
                )
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            arguments_obj = self.get_arguments(action_dict)
            if arguments_obj is None:
                arguments_obj = {}
            elif not isinstance(arguments_obj, dict):
                logger.warning(
                    "Intern-S1 tool arguments must be an object, got %s for %r.",
                    type(arguments_obj).__name__,
                    name,
                )
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            tool_calls.append(
                ToolCall(
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments_obj, ensure_ascii=False),
                    )
                )
            )

        content_parts.append(model_output[cursor:])
        remaining_content: str | None = "".join(content_parts)
        if remaining_content is not None and remaining_content.strip() == "":
            remaining_content = None

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=remaining_content,
        )
