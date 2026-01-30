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
from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_output_into_messages,
)
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
else:
    TokenizerLike = object

logger = init_logger(__name__)

# Harmony special token names
HARMONY_END_TOKEN = "<|end|>"
HARMONY_START_TOKEN = "<|start|>"
HARMONY_CHANNEL_TOKEN = "<|channel|>"


class OpenAIToolParser(ToolParser):
    """Tool parser for GPT-OSS Harmony models."""

    def __init__(self, tokenizer: "TokenizerLike"):
        super().__init__(tokenizer)
        # Cache token IDs for tool_choice=required
        self._trigger_pattern: list[int] | None = None
        self._forced_sequence: list[int] | None = None
        self._init_harmony_tokens()

    def _init_harmony_tokens(self) -> None:
        """Initialize Harmony token sequences from vocabulary."""
        try:
            vocab = self.vocab

            # Trigger pattern: <|end|><|start|>assistant<|channel|>
            end_id = vocab.get(HARMONY_END_TOKEN)
            start_id = vocab.get(HARMONY_START_TOKEN)
            channel_id = vocab.get(HARMONY_CHANNEL_TOKEN)
            assistant_ids = self.model_tokenizer.encode(
                "assistant", add_special_tokens=False
            )

            if (
                end_id is not None
                and start_id is not None
                and channel_id is not None
                and len(assistant_ids) == 1
            ):
                self._trigger_pattern = [
                    end_id,
                    start_id,
                    assistant_ids[0],
                    channel_id,
                ]

            # Forced sequence: "commentary to="
            forced_ids = self.model_tokenizer.encode(
                "commentary to=", add_special_tokens=False
            )
            if forced_ids:
                self._forced_sequence = forced_ids

        except Exception:
            logger.warning(
                "Failed to initialize Harmony tokens. "
                "tool_choice='required' may not work correctly."
            )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        token_ids: Sequence[int] | None = None,
    ) -> ExtractedToolCallInformation:
        if token_ids is None:
            raise NotImplementedError(
                "OpenAIToolParser requires token IDs and does not support "
                "text-based extraction."
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
                        # remove any extra newlines or other odd formatting
                        try:
                            tool_args = json.dumps(json.loads(msg_text))
                        except json.JSONDecodeError:
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

        # Extract partial content from the parser state if truncated
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
            content=final_content or commentary_content,
        )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Adjust request for Harmony format tool_choice='required'."""
        request = super().adjust_request(request)

        if request.tool_choice == "required" and request.tools:
            if self._trigger_pattern and self._forced_sequence:
                if request.vllm_xargs is None:
                    request.vllm_xargs = {}
                request.vllm_xargs["harmony_tool_required"] = {
                    "trigger_pattern": self._trigger_pattern,
                    "forced_sequence": self._forced_sequence,
                }
                request.structured_outputs = None
            else:
                logger.warning(
                    "Harmony tokens not initialized. "
                    "tool_choice='required' will not be enforced."
                )

        return request

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
        raise NotImplementedError("Not being used, manual parsing in serving_chat.py")
