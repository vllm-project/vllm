# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import regex as re
import xgrammar as xgr

from vllm.cohere.guided_decoding.cohere_constants import MODEL_TO_TOOL_SCHEME
from vllm.cohere.utils import get_text_model_name
from vllm.v1.engine.async_llm import AsyncLLM


def collect_tool_schema(engine: AsyncLLM, tool_schema: list[dict]) -> str:
    model_architecture = get_text_model_name(engine.model_config)
    grammar = collect_tool_schema_v2(model_architecture, tool_schema)
    return grammar


def collect_tool_schema_v2(text_model_arch: str, tool_schema: list[dict]) -> str:
    tool_dictionary = {}
    grammar = None
    for tool in tool_schema:
        tool_name = tool["name"]
        tool_parameters = json.dumps(tool["parameters"])
        if MODEL_TO_TOOL_SCHEME[text_model_arch] == "command-a-tools":
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
        elif MODEL_TO_TOOL_SCHEME[text_model_arch] == "command-r-tools":
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
