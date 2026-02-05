# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Structural Tag Utilities for Tool Calling

This module provides utilities for generating structural tags that enable
constrained generation of tool calls. Structural tags allow the model to:

1. Generate native tool call tokens (e.g., <tool_call>) freely
2. Apply JSON schema constraints ONLY within the arguments region
3. Continue generating freely after the constrained region

This is in contrast to full JSON schema constraints which force the entire
output to follow a schema, preventing the model from using its native format.

Example (tool_choice="required"):
    Without structural tags:
        [{"name": "get_weather", "parameters": {"city": "NYC"}}]
        Issue: First token forced to '[', bypasses native format

    With structural tags:
        <tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.openai.protocol import ChatCompletionToolsParam


@dataclass(frozen=True)
class StructureInfo:
    """
    Defines the structural boundaries for constrained tool call generation.

    The structure works as follows:
        [free text] [trigger] [begin] [CONSTRAINED JSON] [end] [free text]

    Attributes:
        trigger: Token sequence that activates constraint checking (e.g., "<tool_call>")
        begin: Fixed prefix that includes everything up to the constrained region.
               This can include special tokens AND the function name.
        end: Fixed suffix after the constrained region.
        schema_is_args_only: If True, the schema constrains only the arguments JSON.
                             If False, the schema constrains the full tool object.

    Example formats:

        Name INSIDE JSON (Hermes, Qwen2.5):
            begin='<tool_call>{"name": "get_weather", "arguments": '
            end='}</tool_call>'
            schema_is_args_only=True  # Schema is for {"city": "..."}, not full object

        Name OUTSIDE JSON (DeepSeek, Kimi K2):
            begin='<|tool▁call▁begin|>get_weather<|tool▁sep|>'
            end='<|tool▁call▁end|>'
            schema_is_args_only=True  # Arguments JSON directly follows begin

    TODO(mgoin): Add support for Pythonic/XML-ish formats
    (e.g., get_weather(city="NYC")) because they cannot use structural tags since
    the content between begin/end is not JSON. In the future, we should add support
    for arbitrary EBNF grammars.
    """

    trigger: str
    begin: str
    end: str
    schema_is_args_only: bool = True  # Most formats only constrain the arguments


def build_structural_tag_config(
    tools: list[ChatCompletionToolsParam],
    get_structure_info: Callable[[str], StructureInfo],
) -> dict[str, Any]:
    """
    Build a structural tag configuration for constrained tool call generation.

    Args:
        tools: List of available tools with their schemas
        get_structure_info: Function (tool_name) -> StructureInfo for each tool

    Returns:
        A structural tag configuration dict that can be serialized to JSON
        and passed to the structured outputs backend.

    Example output:
        {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": '<tool_call>{"name": "get_weather", "arguments": ',
                    "schema": {"type": "object", "properties": {"city": {...}}},
                    "end": "}</tool_call>"
                }
            ],
            "triggers": ["<tool_call>"]
        }
    """
    structures: list[dict[str, Any]] = []
    triggers: set[str] = set()

    for tool in tools:
        name = tool.function.name
        info = get_structure_info(name)

        # Determine schema based on format
        if info.schema_is_args_only:
            # Most formats: function name is in begin pattern, only constrain arguments
            schema = tool.function.parameters or {
                "type": "object",
                "properties": {},
            }
        else:
            # Rare case: constrain the full tool object including name selection
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "arguments": tool.function.parameters
                    or {
                        "type": "object",
                        "properties": {},
                    },
                },
                "required": ["name", "arguments"],
            }

        structures.append(
            {
                "begin": info.begin,
                "schema": schema,
                "end": info.end,
            }
        )
        triggers.add(info.trigger)

    return {
        "type": "structural_tag",
        "structures": structures,
        "triggers": list(triggers),
    }
