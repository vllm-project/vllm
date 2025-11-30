# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from vllm.entrypoints.chat_utils import CustomChatCompletionMessageParam
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

logger = logging.getLogger(__name__)


class StreamableParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(self, *, tokenizer, reasoning_parser: ReasoningParser):
        self.chat_completion_messages: list[CustomChatCompletionMessageParam] = []
        self.tokens: list[int] = []
        self.tokenizer = tokenizer

        # Initialize reasoning parser instance if provided
        self.reasoning_parser_instance = reasoning_parser(tokenizer)

        # start like this
        self.current_role = "assistant"
        # self.current_sentence = Sentence(
        #     author=Author(role=self.current_role), content=[]
        # )
        self.current_chat_completion_message = CustomChatCompletionMessageParam(
            role=self.current_role, content=[]
        )
        self.current_channel = "think"
        self.current_text = ""

    def render_for_completion(self):
        """TODO: Maybe this can be the chat template
        to help generate the initial prompt?"""
        pass

    def process(self, token: int) -> "StreamableParser":
        # see process_next()
        # https://github.com/openai/harmony/blob/main/src/encoding.rs#L1114
        self.tokens.append(token)
        decoded = self.tokenizer.decode(token)
        if self.reasoning_parser_instance.is_reasoning_end([token]):
            # TODO: how to capture reasoning?
            # new_content = {
            #     "role": "assistant",
            #     "reasoning_content": self.current_text
            # }

            new_content = ResponseReasoningTextContent(
                text=self.current_text, type="reasoning_text"
            )

            self.current_chat_completion_message["content"].append(new_content)

            self.current_text = ""
            self.current_channel = "final"
        elif token == self.tokenizer.eos_token_id:
            # end of sentence
            new_content = ChatCompletionContentPartTextParam(
                text=self.current_text, type="text"
            )
            self.current_chat_completion_message["content"].append(new_content)
            self.chat_completion_messages.append(self.current_chat_completion_message)

            self.current_text = ""
            self.current_channel = None
        else:
            self.current_text += decoded

        # TODO: current state of sentences, etc
        return self


def get_streamable_parser_for_simple_context(
    *, tokenizer, reasoning_parser: ReasoningParser, sentences
) -> StreamableParser:
    """Factory function to create a StreamableParser with optional reasoning parser.

    Args:
        tokenizer: The tokenizer to use for decoding tokens
        reasoning_parser: Optional reasoning parser class (e.g., MiniMaxM2ReasoningParser)

    Returns:
        StreamableParser instance configured with the provided parser
    """
    return StreamableParser(tokenizer=tokenizer, reasoning_parser=reasoning_parser)


"""
TODO:
how to figure out which tokens are special tokens

system
tool
ai
"""
