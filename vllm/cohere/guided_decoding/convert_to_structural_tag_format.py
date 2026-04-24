# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json

from vllm.cohere.guided_decoding.cohere_constants import MODEL_TO_TAG_STYLE
from vllm.cohere.guided_decoding.tool_grammar import collect_tool_schema  # type: ignore


# Builder: produces vLLM response_format in xgrammar's canonical format.
# See xgrammar docs: type "structural_tag" with "format" = triggered_tags
# and tag content type = json_schema | grammar.
def convert_schema_to_structural_tags(
    schema: dict | None = None, tools: str | None = None, engine=None
) -> str | None:
    """
    Returns a response_format string accepted by xgrammar's structural tag format.
    Uses the canonical shape: {"type": "structural_tag", "format": {...}} with
    format.type "triggered_tags" and tag content type "json_schema" or "grammar".
    """
    tags = []
    triggers = []
    model_architecture = engine.model_config.architectures[0]  # add better naming
    json_tags, tool_tags = MODEL_TO_TAG_STYLE[model_architecture]
    if schema and tools is None:
        if json_tags is None:
            return None

        style = json_tags
        tags.append(
            {
                "begin": style.begin_template,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": style.end,
            }
        )
        triggers.append(style.trigger)

    if tools:
        # Tools work for Command 3 and Command 2
        # The tool grammar takes in the tool schemas
        # provided by the user and converts them to EBNF format.
        # The reason of using this form is to align with the tool
        # output format in Command 3.

        tool_grammar = collect_tool_schema(engine, json.loads(tools))

        style = tool_tags
        if schema is not None and json_tags is not None:
            # This is a support requested by endpoints for a North
            # usecase where the model should follow a JSON mode
            # for a GG request if the model decides it does not
            # want to use any tools. for which we pass two tags,
            # one for tools and one for jsonschema
            style_json = json_tags
            tags.append(
                {
                    "begin": style_json.begin_template,
                    "content": {"type": "json_schema", "json_schema": schema},
                    "end": style_json.end,
                }
            )
            triggers.append(style_json.trigger)

        tags.append(
            {
                "begin": style.begin_template,
                "content": {"type": "grammar", "grammar": tool_grammar},
                "end": style.end,
            }
        )
        triggers.append(style.trigger)

    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": triggers,
                "tags": tags,
            },
        }
    )
