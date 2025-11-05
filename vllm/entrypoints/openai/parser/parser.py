# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

from vllm.entrypoints.openai.parser.sentence import Author, Role, Sentence, TextContent
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

logger = logging.getLogger(__name__)


class StreamableParser:
    """Incremental parser over completion tokens with reasoning support."""

    def __init__(self, *, tokenizer, reasoning_parser: ReasoningParser | None = None):
        self.sentences: list[Sentence] = []
        self.tokens: list[int] = []
        self.tokenizer = tokenizer

        # Initialize reasoning parser instance if provided
        self.reasoning_parser_instance = None
        if reasoning_parser is not None:
            try:
                self.reasoning_parser_instance = reasoning_parser(tokenizer)
            except Exception as e:
                # If instantiation fails, we'll skip reasoning parsing
                logger.warning(f"Failed to instantiate reasoning parser: {e}")

        # start like this
        self.current_role = Role.ASSISTANT
        self.current_sentence = Sentence(
            author=Author(role=self.current_role), content=[]
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
            new_content = TextContent(
                text=self.current_text, channel=self.current_channel
            )
            self.current_sentence.content.append(new_content)

            self.current_text = ""
            self.current_channel = "final"
        elif token == self.tokenizer.eos_token_id:
            # end of sentence
            new_content = TextContent(
                text=self.current_text, channel=self.current_channel
            )
            self.current_sentence.content.append(new_content)
            self.sentences.append(self.current_sentence)

            self.current_text = ""
            self.current_channel = None
        else:
            self.current_text += decoded

        # TODO: current state of sentences, etc
        return self


def get_streamable_parser_for_simple_context(
    *, tokenizer, reasoning_parser=None
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
