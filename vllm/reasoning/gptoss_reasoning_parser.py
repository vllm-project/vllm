# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

no_func_reasoning_tag = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            {
                "begin": "<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            }
        ],
        "triggers": ["<|channel|>analysis"],
        "stop_after_first": False,
    },
}


def from_builtin_tool_to_tag(tool: str) -> list[dict]:
    tag = [
        {
            "begin": f"<|channel|>commentary to={tool}",
            "content": {"type": "any_text"},
            "end": "<|end|>",
        },
        {
            "begin": f"<|channel|>analysis to={tool}",
            "content": {"type": "any_text"},
            "end": "<|end|>",
        },
    ]
    return tag


def tag_with_builtin_funcs(no_func_reasoning_tag, builtin_tool_list: list[str]) -> dict:
    import copy

    new_tag = copy.deepcopy(no_func_reasoning_tag)
    new_tag["format"]["triggers"].append("<|channel|>commentary to=")

    for tool in builtin_tool_list:
        new_tag["format"]["tags"].extend(from_builtin_tool_to_tag(tool))
    return new_tag


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return True

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return True

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "GptOssReasoningParser only provides boundary detection. "
            "Use HarmonyParser for output parsing."
        )

    # This function prepares the structural tag to format reasoning output
    def prepare_structured_tag(
        self, original_tag: str | None, tool_server: ToolServer | None
    ) -> str | None:
        if original_tag is None:
            if tool_server is None:
                return json.dumps(no_func_reasoning_tag)
            else:
                builtin_tool_list: list[str] = []
                if tool_server.has_tool("browser"):
                    builtin_tool_list.append("browser")
                if tool_server.has_tool("python"):
                    builtin_tool_list.append("python")
                if tool_server.has_tool("container"):
                    builtin_tool_list.append("container")

                if len(builtin_tool_list) > 0:
                    logger.info("Builtin_tool_list: %s", builtin_tool_list)
                    func_tag = json.dumps(
                        tag_with_builtin_funcs(no_func_reasoning_tag, builtin_tool_list)
                    )
                else:
                    logger.info("Builtin_tool_list is empty")
                    func_tag = json.dumps(no_func_reasoning_tag)

                return func_tag
        else:
            # There is potential risk for appending the tag to the original tag
            return original_tag
