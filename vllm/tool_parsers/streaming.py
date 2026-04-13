# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import TYPE_CHECKING

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.tool_parsers.utils import partial_json_loads
from vllm.utils.mistral import (
    generate_mistral_tool_call_id,
    is_mistral_tokenizer,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object


def _bracket_level(s: str, opening: str = "{", closing: str = "}") -> int:
    """
    Calculate the current level of nested brackets in a given string.
    """
    level = 0
    for char in s:
        if char == opening:
            level += 1
        elif char == closing:
            level -= 1
    return level


def filter_delta_text(
    delta_text: str,
    previous_text: str,
) -> tuple[str, bool]:
    # remove last '},' of the tool definition stemming from the
    # "name"/"parameters" outer object or closing ']' of the tool list
    # count occurrences of opening and closing curly braces and
    # once level 0 is reached stop outputting text
    # if 0 is reached while parsing the delta_text we know the current
    # tool will finish in this current iteration
    bracket_level = _bracket_level(previous_text)
    updated_delta = ""
    passed_zero = False
    for char in delta_text:
        if char == "{":
            bracket_level += 1
            passed_zero = bracket_level == 0
        elif char == "}":
            bracket_level -= 1
            passed_zero = bracket_level == 0

        if bracket_level != 0:
            updated_delta += char
        else:
            # if a comma is reached at level 0 we can stop
            if char == ",":
                break
    return updated_delta, passed_zero


def extract_named_tool_call_streaming(
    *,
    delta_text: str,
    function_name: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
    tokenizer: "TokenizerLike",
    tool_call_array_index: int,
) -> tuple[DeltaMessage, bool, bool]:
    """Build a streaming tool-call delta for forced named tool choice."""
    created_new_tool_call = False
    if function_name_returned:
        delta_tool_call = DeltaToolCall(
            function=DeltaFunctionCall(arguments=delta_text),
            index=tool_call_array_index,
        )
    else:
        if is_mistral_tokenizer(tokenizer):
            tool_call_id = generate_mistral_tool_call_id()
        else:
            tool_call_id = make_tool_call_id(
                id_type=tool_call_id_type,
                func_name=function_name,
                idx=tool_call_idx,
            )
        delta_tool_call = DeltaToolCall(
            id=tool_call_id,
            type="function",
            function=DeltaFunctionCall(
                name=function_name,
                arguments=delta_text,
            ),
            index=tool_call_array_index,
        )
        function_name_returned = True
        created_new_tool_call = True

    return (
        DeltaMessage(tool_calls=[delta_tool_call]),
        function_name_returned,
        created_new_tool_call,
    )


def extract_required_tool_call_streaming(
    *,
    previous_text: str,
    current_text: str | None,
    delta_text: str,
    function_name_returned: bool,
    tool_call_idx: int | None,
    tool_call_id_type: str,
) -> tuple[DeltaMessage | None, bool, bool]:
    """Incrementally parse required-tool JSON output into DeltaMessage."""
    if current_text is None or current_text == "":
        # if the current text is empty, we cannot parse it
        return None, function_name_returned, False

    try:
        flags = Allow.ALL
        obj, _ = partial_json_loads(current_text, flags)
    except (
        partial_json_parser.core.exceptions.MalformedJSON,
        json.JSONDecodeError,
    ):
        # not enough tokens to parse into JSON yet
        obj = None

    # check if the current text is a valid array
    # containing a partial tool calling object
    # if not repeat
    if obj is None or not isinstance(obj, list) or not len(obj) > 0:
        return None, False, False

    _, finishes_previous_tool = filter_delta_text(delta_text, previous_text)
    # take the last tool call from the generated list
    current_tool_call = obj[-1]
    current_tool_call_index = len(obj) - 1

    if (
        function_name_returned
        and tool_call_idx is not None
        and current_tool_call_index >= tool_call_idx
    ):
        if finishes_previous_tool:
            # A new tool object may start in the same iteration that finishes
            # the previous one. In that case, keep streaming argument text for
            # the previous tool before resetting the function-name state.
            filtered_delta_text, _ = filter_delta_text(delta_text, previous_text)
            if filtered_delta_text == "":
                return None, False, False
            previous_tool_call_index = tool_call_idx - 1 if tool_call_idx > 0 else 0
            delta_message = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        function=DeltaFunctionCall(
                            name=None,
                            arguments=filtered_delta_text,
                        ),
                        index=previous_tool_call_index,
                    )
                ]
            )
            return delta_message, False, False
        function_name_returned = False

    if not finishes_previous_tool and (
        "name" not in current_tool_call or "parameters" not in current_tool_call
    ):
        return None, False, False

    if not function_name_returned:
        # get partly generated arguments from the latest tool call
        param_match = re.search(r'.*"parameters":\s*(.*)', current_text, re.DOTALL)
        arguments = param_match.group(1) if param_match else ""
        arguments, _ = filter_delta_text(arguments, previous_text)

        # if this iteration finishes a previous tool call but a
        # new incomplete tool is already generated, take the
        # previous from the list
        if finishes_previous_tool and "parameters" not in current_tool_call:
            current_tool_call = obj[-2]

        delta_message = DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    id=make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx,
                    ),
                    function=DeltaFunctionCall(
                        name=current_tool_call["name"],
                        arguments=arguments,
                    ),
                    index=len(obj) - 1,
                    type="function",
                )
            ]
        )
        return delta_message, True, True

    filtered_delta_text, _ = filter_delta_text(delta_text, previous_text)
    if filtered_delta_text == "":
        return None, function_name_returned, False

    delta_message = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                function=DeltaFunctionCall(
                    # OpenAI API returns None
                    # instead of name every time
                    name=None,
                    arguments=filtered_delta_text,
                ),
                index=len(obj) - 1,
            )
        ]
    )
    return delta_message, function_name_returned, False
