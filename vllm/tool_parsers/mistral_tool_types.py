# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import ToolCall
from vllm.utils.mistral import (
    generate_mistral_tool_call_id,
    is_valid_mistral_tool_call_id,
)


class MistralToolCall(ToolCall):
    """ToolCall variant with Mistral-compatible tool call IDs."""

    id: str = Field(default_factory=generate_mistral_tool_call_id)

    @staticmethod
    def generate_random_id() -> str:
        return generate_mistral_tool_call_id()

    @staticmethod
    def is_valid_id(value: str) -> bool:
        return is_valid_mistral_tool_call_id(value)
