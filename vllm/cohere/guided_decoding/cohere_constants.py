# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import cast

# C3
C3_TOOL_CALL_PREFIX = "<|START_ACTION|>"
C3_TOOL_CALL_POSTFIX = "<|END_ACTION|>"
C3_RESPONSE_PREFIX = "<|START_RESPONSE|>"
C3_RESPONSE_POSTFIX = "<|END_RESPONSE|>"
C2_RESPONSE_PREFIX = "<|START_OF_TURN_TOKEN|>"
C2_RESPONSE_POSTFIX = "<|END_OF_TURN_TOKEN|>"

# C2
C2_TOOL_CALL_PREFIX = "Action: ```json"
C2_TOOL_CALL_POSTFIX = "```"

# start thinking token
START_THINKING_TOKEN = "<|START_THINKING|>"
# end thinking token
END_THINKING_TOKEN = "<|END_THINKING|>"


@dataclass
class TagRegistry:
    name: str
    trigger: str
    begin_template: str
    end: str


TagStyle = TagRegistry | None
ModelTagStyles = tuple[TagRegistry | None, TagRegistry]


"""
The structural tag formatter is built of two main components:
1) A registry of tag styles (TagRegistry, TAG_REGISTRY) - 
These are the different tag styles we support for different 
model architectures and usecases.
2) A builder function that produces a response_format dict 
accepted by Xgrammar's structural tags format
   format (convert_schema_to_structural_tags)
"""

TAG_REGISTRY: dict[str, TagStyle] = {
    "command-a-json": TagRegistry(
        name="command-a-json",
        trigger="<|START_RESPONSE|>",
        begin_template="<|START_RESPONSE|>",
        end="<|END_RESPONSE|>",
    ),
    "command-a-tools": TagRegistry(
        name="command-a-tools",
        trigger="<|START_ACTION|>",
        begin_template="<|START_ACTION|>",
        end="<|END_ACTION|>",
    ),
    "command-r-tools": TagRegistry(
        name="command-r-tools",
        trigger="Action: ```json",
        begin_template="Action: ```json",
        end="```",
    ),
    "command-r-json": None,
}

MODEL_TO_TAG_STYLE: dict[str, ModelTagStyles] = {
    "Cohere2ForCausalLM": (
        cast(TagRegistry, TAG_REGISTRY["command-a-json"]),
        cast(TagRegistry, TAG_REGISTRY["command-a-tools"]),
    ),
    "Cohere2MoeForCausalLM": (
        cast(TagRegistry, TAG_REGISTRY["command-a-json"]),
        cast(TagRegistry, TAG_REGISTRY["command-a-tools"]),
    ),
    "Cohere2VisionForConditionalGeneration": (
        cast(TagRegistry, TAG_REGISTRY["command-a-json"]),
        cast(TagRegistry, TAG_REGISTRY["command-a-tools"]),
    ),
    "CohereForCausalLM": (
        TAG_REGISTRY["command-r-json"],
        cast(TagRegistry, TAG_REGISTRY["command-r-tools"]),
    ),
}

MODEL_TO_TOOL_SCHEME: dict[str, str] = {
    "Cohere2ForCausalLM": "command-a-tools",
    "Cohere2MoeForCausalLM": "command-a-tools",
    "Cohere2VisionForConditionalGeneration": "command-a-tools",
    "CohereForCausalLM": "command-r-tools",
}

MODEL_TO_PREFIX_POSTFIX: dict[str, tuple[str, str]] = {
    "Cohere2ForCausalLM": (C3_RESPONSE_PREFIX, C3_RESPONSE_POSTFIX),
    "Cohere2MoeForCausalLM": (C3_RESPONSE_PREFIX, C3_RESPONSE_POSTFIX),
    "Cohere2VisionForConditionalGeneration": (C3_RESPONSE_PREFIX, C3_RESPONSE_POSTFIX),
    "CohereForCausalLM": (C2_RESPONSE_PREFIX, C2_RESPONSE_POSTFIX),
}
