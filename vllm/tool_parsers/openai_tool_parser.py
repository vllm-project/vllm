# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import parse_output_into_messages
from vllm.logger import init_logger
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

logger = init_logger(__name__)


class OpenAIToolParser(ToolParser):
    """
    Tool parser for GPT-OSS Harmony models.

    Supports tool_choice="required" via EBNF grammar that constrains
    generation to analysis/commentary channels, blocking the final channel.
    """

    def __init__(self, tokenizer: "TokenizerLike"):
        super().__init__(tokenizer)

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if not request.tools or request.tool_choice != "required":
            return super().adjust_request(request)

        tool_names = [t.function.name for t in request.tools]
        grammar = self._build_tool_required_grammar(tool_names)
        request.structured_outputs = StructuredOutputsParams(grammar=grammar)  # type: ignore[call-arg]
        request.response_format = None
        return request

    @staticmethod
    def _build_tool_required_grammar(tool_names: list[str]) -> str:
        """Build EBNF grammar that enforces tool calls for Harmony format.

        The grammar:
        - Allows analysis blocks (multi-round reasoning)
        - Allows commentary preambles
        - Requires at least one tool call (commentary to=functions.X)
        - Blocks the final channel entirely (not defined in grammar)

        Content rule uses ([^<] | "<" [^|])* to allow '<' in text
        while blocking Harmony special tokens (<|...|>).
        """
        for n in tool_names:
            if '"' in n or "\n" in n:
                raise ValueError(
                    f"Tool name {n!r} contains characters invalid for EBNF grammar"
                )
        func_alts = " | ".join(f'"functions.{n}"' for n in tool_names)
        return (
            "root ::= non_tool_block* tool_block more_tool*\n"
            'non_tool_block ::= ("analysis" | "commentary")'
            ' "<|message|>" content "<|end|>"'
            ' "<|start|>" "assistant" "<|channel|>"\n'
            'tool_block ::= "commentary to=" func_name'
            ' "<|message|>" content "<|end|>" "<|call|>"\n'
            'more_tool ::= "<|start|>" "assistant" "<|channel|>"'
            " non_tool_block* tool_block\n"
            f"func_name ::= {func_alts}\n"
            'content ::= ([^<] | "<" [^|])*'
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support text-based extraction."  # noqa: E501
            )

        parser = parse_output_into_messages(token_ids)
        tool_calls = []
        final_content = None
        commentary_content = None

        if len(parser.messages) > 0:
            for msg in parser.messages:
                if len(msg.content) < 1:
                    continue
                msg_text = msg.content[0].text
                if msg.recipient and msg.recipient.startswith("functions."):
                    # If no content-type is given assume JSON, as that's the
                    # most common case with gpt-oss models.
                    if not msg.content_type or "json" in msg.content_type:
                        # load and dump the JSON text to check validity and
                        # remove any extra newlines or other odd formatting.
                        # Use raw_decode to handle trailing garbage from
                        # partial Harmony parsing (e.g. structural tokens).
                        try:
                            obj, _ = json.JSONDecoder().raw_decode(msg_text)
                            tool_args = json.dumps(obj)
                        except (json.JSONDecodeError, ValueError):
                            logger.exception(
                                "Error decoding JSON tool call from response."
                            )
                            tool_args = msg_text
                    else:
                        tool_args = msg_text
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=msg.recipient.split("functions.")[1],
                                arguments=tool_args,
                            ),
                        )
                    )
                elif msg.channel == "final":
                    final_content = msg_text
                elif msg.channel == "commentary" and not msg.recipient:
                    commentary_content = msg_text

        # Extract partial content from the parser state if the generation was truncated
        if parser.current_content:
            if parser.current_channel == "final":
                final_content = parser.current_content
            elif (
                parser.current_channel == "commentary" and not parser.current_recipient
            ):
                commentary_content = parser.current_content

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            # prefer final content over commentary content if both are present
            # commentary content is tool call preambles meant to be shown to the user
            content=final_content or commentary_content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        raise NotImplementedError(
            "Not being used, manual parsing in serving_chat.py"  # noqa: E501
        )
