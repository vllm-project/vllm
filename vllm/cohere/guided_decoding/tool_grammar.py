# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import regex as re
import xgrammar as xgr

from vllm.cohere.guided_decoding.cohere_constants import COMMAND_R_TOOLS_TAG
from vllm.reasoning.cohere_command_reasoning_parser import (
    COMMAND_A_TOOLS_TAG,
    MODEL_TO_TAG_STYLE,
    CohereNormalizedTool,
    CohereTagRegistry,
    CohereTagStyle,
)

# Single registration for guided decoding + structural tags (Command R / C2).
COMMAND_R_TOOLS_TAG = CohereTagRegistry(
    trigger=COMMAND_R_TOOLS_TAG.trigger,
    end=COMMAND_R_TOOLS_TAG.end,
)
MODEL_TO_TAG_STYLE["CohereForCausalLM"] = CohereTagStyle(
    json_tags=(),
    tools=COMMAND_R_TOOLS_TAG,
)


def collect_tool_schema_v2(
    text_model_arch: str, tool_schema: list[CohereNormalizedTool]
) -> str:
    tool_dictionary = {}
    grammar = None
    styles = MODEL_TO_TAG_STYLE[text_model_arch]
    tool_style = styles.tools
    for tool in tool_schema:
        tool_name = tool["name"]
        tool_parameters = json.dumps(tool["parameters"])
        if tool_style == COMMAND_A_TOOLS_TAG:
            json_schema = f"""{{
                            "type": "object",
                            "properties": {{
                                "tool_call_id": {{
                                    "type": "string",
                                    "pattern": "^[0-9]+$"
                                }},
                                "tool_name": {{
                                    "type": "string",
                                    "const": "{tool_name}"
                                }},
                                "parameters": {tool_parameters}
                                }}
                                }}"""
        elif tool_style == COMMAND_R_TOOLS_TAG:
            json_schema = f"""{{
                            "type": "object",
                            "properties": {{
                                "tool_name": {{
                                    "type": "string",
                                    "const": "{tool_name}"
                                }},
                                "parameters": {tool_parameters}
                                }}
                                }}"""
        tool_grammar = xgr.Grammar.from_json_schema(json_schema)
        tool_grammar = str(tool_grammar)
        matches = re.findall(r"\b(\w+)\s*::=", tool_grammar)
        for match in matches:
            tool_grammar = tool_grammar.replace(match, tool_name + match)
        tool_dictionary[tool_name] = (
            tool_name + " ::= " + tool_name + "root" + "\n" + tool_grammar
        )
    tool_alternatives = "tool ::= " + " | ".join(tool_dictionary.keys())
    tool_rules = "\n    ".join(tool_dictionary.values())
    grammar = f"""root ::= tools
    tools ::= ws "[" ws tool ws ("," ws tool)*  ws "]" ws
    ws    ::= (" " | "\\t" | "\\n")*
    {tool_alternatives}
    {tool_rules}
    """
    return grammar
