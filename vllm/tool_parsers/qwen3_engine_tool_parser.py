# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.registered_adapters import Qwen3ParserToolAdapter
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from xgrammar import StructuralTag


class Qwen3EngineToolParser(Qwen3ParserToolAdapter):  # type: ignore[valid-type, misc]
    structural_tag_model = "qwen_3_coder"

    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(
            request, reasoning=reasoning, strict_level=strict_level
        )
        if tag is None:
            return None
        from vllm.tool_parsers.structural_tag_registry import (
            allow_unwrapped_qwen_xml_tool_calls,
        )

        # The parser also extracts calls written without <tool_call>.
        return allow_unwrapped_qwen_xml_tool_calls(tag)
