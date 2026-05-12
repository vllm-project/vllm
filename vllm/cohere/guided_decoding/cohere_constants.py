# ruff: noqa: E501
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.cohere_command_reasoning_parser import CohereTagRegistry

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

# BLS tags
BLS_RESPONSE_PREFIX = "<|START_TEXT|>"
BLS_RESPONSE_POSTFIX = "<|END_TEXT|>"

# start thinking token
START_THINKING_TOKEN = "<|START_THINKING|>"
# end thinking token
END_THINKING_TOKEN = "<|END_THINKING|>"


COMMAND_R_TOOLS_TAG = CohereTagRegistry(
    trigger="Action: ```json",
    end="```",
)


MODEL_TO_PREFIX_POSTFIX: dict[str, tuple[str, str]] = {
    "Cohere2ForCausalLM": (C3_RESPONSE_PREFIX, C3_RESPONSE_POSTFIX),
    "Cohere2MoeForCausalLM": (C3_RESPONSE_PREFIX, C3_RESPONSE_POSTFIX),
    "Cohere2VisionForConditionalGeneration": (
        BLS_RESPONSE_PREFIX,
        BLS_RESPONSE_POSTFIX,
    ),
    "CohereForCausalLM": (C2_RESPONSE_PREFIX, C2_RESPONSE_POSTFIX),
}
