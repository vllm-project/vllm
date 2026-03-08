# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for pythonic-style tool call parsers.

Used by PythonicToolParser, Llama4PythonicToolParser, and
Olmo3PythonicToolParser to parse AST-based tool calls, compute
streaming deltas, and validate incomplete Python expressions.
"""

import ast
import json
from typing import Any

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class UnexpectedAstError(Exception):
    """Raised when the AST structure does not match the expected
    pythonic tool call format."""

    pass


_JSON_NAME_LITERALS = {
    "null": None,
    "true": True,
    "false": False,
}


def get_parameter_value(val: ast.expr) -> Any:
    """Extract a Python literal value from an AST expression node.

    Handles constants, dicts, lists, and JSON-style name literals
    (null, true, false) that some models produce instead of Python
    literals (None, True, False).

    Raises:
        UnexpectedAstError: If the AST node is not a supported literal type.
    """
    if isinstance(val, ast.Constant):
        return val.value
    elif isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            logger.warning(
                "Dict argument keys are not all literals: %s",
                ast.dump(val),
            )
            raise UnexpectedAstError(
                "Dict tool call arguments must have literal keys"
            )
        return {
            k.value: get_parameter_value(v)  # type: ignore
            for k, v in zip(val.keys, val.values)
        }
    elif isinstance(val, ast.List):
        return [get_parameter_value(v) for v in val.elts]
    elif isinstance(val, ast.Name) and val.id in _JSON_NAME_LITERALS:
        return _JSON_NAME_LITERALS[val.id]
    else:
        logger.warning(
            "Unsupported AST node type in tool call arguments: %s",
            ast.dump(val),
        )
        raise UnexpectedAstError("Tool call arguments must be literals")


def handle_single_tool(call: ast.Call) -> ToolCall:
    """Convert a single AST function call node into a ToolCall object.

    Raises:
        UnexpectedAstError: If the call node does not have a simple
            function name (e.g. it's an attribute access or subscript).
    """
    if not isinstance(call.func, ast.Name):
        logger.warning(
            "Tool call has non-simple function name: %s",
            ast.dump(call.func),
        )
        raise UnexpectedAstError("Invalid tool call name")
    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = get_parameter_value(keyword.value)
    return ToolCall(
        type="function",
        function=FunctionCall(
            name=function_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def make_valid_python(text: str) -> tuple[str, str] | None:
    """Attempt to close all open brackets/quotes to make partial Python valid.

    Used during streaming to parse incomplete tool call expressions by
    appending the necessary closing characters.

    Returns:
        A tuple of (completed_text, added_suffix) if the text can be
        made valid, or None if the text is too incomplete to complete
        meaningfully (e.g. mid-parameter-name or mid-dict-key).

    Raises:
        UnexpectedAstError: If mismatched brackets or parentheses
            are detected.
    """
    bracket_stack: list[str] = []
    for index, char in enumerate(text):
        if char in {"[", "(", "{"}:
            bracket_stack.append(char)
        elif char == "]":
            if not bracket_stack or bracket_stack.pop() != "[":
                raise UnexpectedAstError("Mismatched square brackets")
        elif char == ")":
            if not bracket_stack or bracket_stack.pop() != "(":
                raise UnexpectedAstError("Mismatched parentheses")
        elif char == "}":
            if not bracket_stack or bracket_stack.pop() != "{":
                raise UnexpectedAstError("Mismatched curly braces")
        elif char in {"'", '"'}:
            if bracket_stack and bracket_stack[-1] == char:
                if index > 0 and text[index - 1] == "\\":
                    pass
                else:
                    bracket_stack.pop()
            elif bracket_stack and bracket_stack[-1] in {"'", '"'}:
                pass
            else:
                bracket_stack.append(char)

    text = text.rstrip()
    if text.endswith("=") or text.endswith(":"):
        return None
    if bracket_stack and bracket_stack[-1] == "{":
        trailing_dict_text = text[: text.rfind("{")]
        num_keys = trailing_dict_text.count(":")
        num_values = trailing_dict_text.count(",")
        if num_keys <= num_values:
            return None
    if bracket_stack and bracket_stack[-1] == "(":
        trailing_params_text = text[: text.rfind("(")]
        num_full_param_names = trailing_params_text.count("=")
        num_full_param_values = trailing_params_text.count(",")
        if num_full_param_names <= num_full_param_values:
            return None
    if text.endswith(","):
        text = text[:-1]
    if (
        bracket_stack
        and bracket_stack[-1] == "["
        and not text.endswith("[")
        and not text.endswith(")")
    ):
        return None

    _CLOSING = {"[": "]", "(": ")", "{": "}", "'": "'", '"': '"'}
    added_text = ""
    for char in reversed(bracket_stack):
        added_text += _CLOSING[char]

    return text + added_text, added_text


def compute_tool_delta(
    previously_sent_args: str,
    new_call: ToolCall,
    index: int,
    withheld_suffix: str,
) -> DeltaToolCall | None:
    """Compute the incremental delta between previously streamed arguments
    and the current tool call state.

    Returns:
        A DeltaToolCall with only the new argument characters, or None
        if there is no difference from what was previously sent.
    """
    new_call_args = new_call.function.arguments
    if withheld_suffix:
        if not new_call_args.endswith(withheld_suffix):
            logger.error(
                "Tool call arguments '%s' do not end with expected "
                "withheld suffix '%s'",
                new_call_args,
                withheld_suffix,
            )
        assert new_call_args.endswith(withheld_suffix)
        new_call_args = new_call_args[: -len(withheld_suffix)]
    if not previously_sent_args:
        return DeltaToolCall(
            id=new_call.id,
            type="function",
            index=index,
            function=DeltaFunctionCall(
                name=new_call.function.name,
                arguments=new_call_args,
            ),
        )

    arg_diff = new_call_args[len(previously_sent_args) :]
    return (
        DeltaToolCall(
            id=None,
            index=index,
            function=DeltaFunctionCall(arguments=arg_diff),
        )
        if arg_diff
        else None
    )
