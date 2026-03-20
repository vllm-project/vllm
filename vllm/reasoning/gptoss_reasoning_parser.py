# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.mcp.tool_server import ToolServer
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output
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
    new_tag = copy.deepcopy(no_func_reasoning_tag)
    new_tag["format"]["triggers"].append("<|channel|>commentary to=")

    for tool in builtin_tool_list:
        new_tag["format"]["tags"].extend(from_builtin_tool_to_tag(tool))
    return new_tag


def from_function_tool_to_tag(name: str, parameters: dict | None) -> list[dict]:
    content = (
        {"type": "json_schema", "json_schema": parameters}
        if parameters
        else {"type": "any_text"}
    )
    return [
        {
            "begin": f"<|channel|>commentary to=functions.{name}<|message|>",
            "content": content,
            "end": "<|end|>",
        },
        {
            "begin": f"<|channel|>analysis to=functions.{name}<|message|>",
            "content": content,
            "end": "<|end|>",
        },
    ]


def tag_with_function_tools(base_tag: dict, function_tools: list[dict]) -> dict:
    new_tag = copy.deepcopy(base_tag)

    # Add commentary trigger for function tools if not already covered
    # by the general commentary trigger (added by builtin tools).
    if "<|channel|>commentary to=" not in new_tag["format"]["triggers"]:
        new_tag["format"]["triggers"].append("<|channel|>commentary to=functions.")

    for tool in function_tools:
        new_tag["format"]["tags"].extend(
            from_function_tool_to_tag(tool["name"], tool.get("parameters"))
        )
    return new_tag


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # The model can output some special tokens between "final" and "<|message|>"
        # So we need to look for both sequences to determine the end of reasoning.
        self.reasoning_end_token_ids_prefix = self.model_tokenizer.encode(
            "<|channel|>final"
        )
        self.reasoning_end_token_ids_suffix = self.model_tokenizer.encode("<|message|>")
        # We also need to check for the <|end|> token to avoid false positives from
        # previous messages in multi-turn conversations.
        self.eom_token_id = self.vocab["<|end|>"]
        self.reasoning_max_num_between_tokens = 20

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        end_token_ids_prefix = self.reasoning_end_token_ids_prefix
        end_token_ids_suffix = self.reasoning_end_token_ids_suffix
        assert len(end_token_ids_prefix) > 0, "reasoning_end_token_ids_prefix is empty"
        assert len(end_token_ids_suffix) > 0, "reasoning_end_token_ids_suffix is empty"
        # Check if the end sequence is present in the input_ids.
        # We search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - len(end_token_ids_prefix), -1, -1):
            if input_ids[i] == self.eom_token_id:
                # We looped backwards far enough to find the end of a previous message,
                # which means we have searched the entirety of the current message
                # and can exit early without searching further back into prior
                # messages of the conversation.
                return False
            if input_ids[i : i + len(end_token_ids_prefix)] == end_token_ids_prefix:
                # We have found the prefix, now we look for the suffix after the prefix.
                suffix_start = i + len(end_token_ids_prefix)
                for j in range(
                    suffix_start, len(input_ids) - len(end_token_ids_suffix) + 1
                ):
                    if j - suffix_start >= self.reasoning_max_num_between_tokens:
                        break
                    if (
                        input_ids[j : j + len(end_token_ids_suffix)]
                        == end_token_ids_suffix
                    ):
                        return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content, _ = parse_chat_output(input_ids)
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content, _ = parse_chat_output(list(previous_token_ids))
        cur_reasoning, cur_content, _ = parse_chat_output(list(current_token_ids))
        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            if cur_reasoning.startswith(prev_r):
                reasoning_delta = cur_reasoning[len(prev_r) :] or None
            else:
                reasoning_delta = cur_reasoning
        if cur_content is not None:
            prev_c = prev_content or ""
            if cur_content.startswith(prev_c):
                content_delta = cur_content[len(prev_c) :] or None
            else:
                content_delta = cur_content
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning=reasoning_delta, content=content_delta)

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "gpt-oss has a special branch for parsing reasoning in non-streaming mode. This method shouldn't be used."  # noqa: E501
        )

    # This function prepares the structural tag to format reasoning output
    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: ToolServer | None,
        final_content_format: dict | None = None,
        tool_choice: str | dict | None = None,
        function_tools: list[dict] | None = None,
    ) -> str | None:
        if original_tag is not None:
            # There is potential risk for appending the tag to the original tag
            return original_tag

        base_tag: dict[str, Any] = copy.deepcopy(no_func_reasoning_tag)

        # Add builtin tool tags unless tool_choice is "none" or a named
        # function dict — named forcing should only allow the specific
        # function, not builtin channels that could satisfy at_least_one.
        is_named_function_choice = isinstance(tool_choice, dict)
        if (
            tool_choice != "none"
            and not is_named_function_choice
            and tool_server is not None
        ):
            builtin_tool_list: list[str] = []
            if tool_server.has_tool("browser"):
                builtin_tool_list.append("browser")
            if tool_server.has_tool("python"):
                builtin_tool_list.append("python")
            if tool_server.has_tool("container"):
                builtin_tool_list.append("container")

            if builtin_tool_list:
                logger.info("Builtin_tool_list: %s", builtin_tool_list)
                base_tag = tag_with_builtin_funcs(base_tag, builtin_tool_list)
            else:
                logger.info("Builtin_tool_list is empty")

        # Add function tool tags (unless tool_choice is "none")
        effective_function_tools = None
        if tool_choice != "none" and function_tools:
            effective_function_tools = function_tools
            # If named tool choice, filter to only the named tool
            if isinstance(tool_choice, dict):
                named = tool_choice.get("name")
                effective_function_tools = [
                    t for t in function_tools if t["name"] == named
                ]
            if effective_function_tools:
                base_tag = tag_with_function_tools(base_tag, effective_function_tools)

        # Add final channel tag unless tool_choice blocks it
        if tool_choice != "required" and not isinstance(tool_choice, dict):
            has_function_tools = bool(effective_function_tools)
            if has_function_tools or final_content_format:
                final_content = (
                    final_content_format
                    if final_content_format
                    else {"type": "any_text"}
                )
                base_tag["format"]["tags"].append(
                    {
                        "begin": "<|channel|>final<|message|>",
                        "content": final_content,
                        "end": "<|end|>",
                    }
                )
                base_tag["format"]["triggers"].append("<|channel|>final")

        # For tool_choice=required or named tool, force at least one triggered
        # tag. This blocks <|channel|>final and EOS at the grammar level until
        # the model has emitted at least one tool-call channel.
        if tool_choice == "required" or isinstance(tool_choice, dict):
            # Remove the pure analysis tag (no recipient) from the tag list so
            # that triggered_tags_first only contains function-call tags.  The
            # analysis trigger is kept so analysis-to-functions tags remain
            # reachable in triggered_tags_sub.  This prevents the model from
            # satisfying at_least_one with a pure reasoning channel instead of
            # an actual tool call.
            base_tag["format"]["tags"] = [
                t
                for t in base_tag["format"]["tags"]
                if t.get("begin") != "<|channel|>analysis<|message|>"
            ]
            base_tag["format"]["at_least_one"] = True

        return json.dumps(base_tag)
