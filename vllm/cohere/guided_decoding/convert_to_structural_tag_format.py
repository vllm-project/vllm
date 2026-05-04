# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
from typing import Any

# Import tool_grammar before MODEL_TO_TAG_STYLE so module-level registration runs.
from vllm.cohere.guided_decoding.tool_grammar import collect_tool_schema_v2
from vllm.cohere.utils import get_text_model_name
from vllm.reasoning.cohere_command_reasoning_parser import (
    MODEL_TO_TAG_STYLE,
    CohereTagRegistry,
    _has_effective_tools,
    _tool_definitions_to_schema_list,
)


# Builder: produces vLLM response_format in xgrammar's canonical format.
# See xgrammar docs: type "structural_tag" with "format" = triggered_tags
# and tag content type = json_schema | grammar.
def convert_schema_to_structural_tags(
    schema: dict | None = None,
    tools: str | list[Any] | None = None,
    model_architecture: str | None = None,
    engine=None,
) -> str | None:
    """
    Returns a response_format string accepted by xgrammar's structural tag format.
    Uses the canonical shape: {"type": "structural_tag", "format": {...}} with
    format.type "triggered_tags" and tag content type "json_schema" or "grammar".

    Callers that are not on an engine path (e.g. the reasoning parser) must pass
    ``model_architecture`` explicitly.

    When ``engine`` is provided and ``model_architecture`` is omitted, the
    architecture is resolved from ``engine.model_config`` (including vision
    subconfigs).

    ``MODEL_TO_TAG_STYLE`` lists are ordered as zero or more JSON structural-tag
    styles followed by exactly one tool tag; see ``cohere_constants``.

    ``_has_effective_tools`` and ``_tool_definitions_to_schema_list`` are shared
    with ``vllm.reasoning.cohere_command_reasoning_parser``.
    """
    arch = model_architecture
    if arch is None and engine is not None:
        arch = get_text_model_name(engine.model_config)

    if arch is None or arch not in MODEL_TO_TAG_STYLE:
        return None

    styles = MODEL_TO_TAG_STYLE[arch]
    json_tags = styles.json_tags
    tool_style = styles.tools

    tags: list[dict] = []
    triggers: list[str] = []

    def _add_tag(tag: CohereTagRegistry, content: dict) -> None:
        tags.append({"begin": tag.trigger, "content": content, "end": tag.end})
        triggers.append(tag.trigger)

    if schema is not None:
        if not json_tags:
            return None
        for jt in json_tags:
            _add_tag(jt, {"type": "json_schema", "json_schema": schema})

    if _has_effective_tools(tools):
        # ``tools`` may be a JSON string (poseidon / RESPONSE_FORMAT_TOOL_DEFINITIONS)
        # or a list (Chat Completions ``request.tools`` as Pydantic models or dicts).
        tool_schema_list = _tool_definitions_to_schema_list(tools)
        if not tool_schema_list:
            raise ValueError(
                "No valid tool definitions could be parsed from the request for "
                "structural tag conversion."
            )
        tool_grammar = collect_tool_schema_v2(arch, tool_schema_list)
        _add_tag(tool_style, {"type": "grammar", "grammar": tool_grammar})

    if not tags:
        return None
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
