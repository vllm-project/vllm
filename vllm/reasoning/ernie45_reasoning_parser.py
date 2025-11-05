# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

logger = init_logger(__name__)


class Ernie45ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Ernie45 thinking model.
    The Ernie45 thinking model ouput format is
        abc\n</think>\n\n<response>\ndef\n</response>\n
    or  abc\n</think>\ndef
    """

    response_start_token: str = "<response>"
    response_end_token: str = "</response>"
    newline_token: str = "<0x0A>"

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        self.response_start_token_id = self.vocab.get(self.response_start_token)
        self.response_end_token_id = self.vocab.get(self.response_end_token)
        self.newline_token_id = self.vocab.get(self.newline_token)

        self.parser_token_ids = [self.end_token_id, self.response_end_token_id]

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                "Ernie45 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        The Ernie45 thinking model ouput format is
            abc\n</think>\n\n<response>\ndef\n</response>\n
        or  abc\n</think>\ndef
        - 'abc' goes to reasoning_content
        - 'def' goes to content
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0]
            in [
                self.start_token_id,
                self.end_token_id,
                self.response_start_token_id,
                self.response_end_token_id,
            ]
        ):
            return None

        # No <think> in previous or delta, also need to check for </think>.
        # Because the model may have generated </think> without <think>
        if self.end_token_id in delta_token_ids:
            # </think> in delta with more tokens,
            # extract reasoning content and content
            think_end_index = delta_text.find(self.end_token)
            reasoning_content = delta_text[:think_end_index]
            content = delta_text[think_end_index + len(self.end_token) :]
            content = content.lstrip("\n")
            response_start_idx = content.find(self.response_start_token)
            response_end_idx = content.rfind(self.response_end_token)
            if response_start_idx != -1:
                content = content[response_start_idx + len(self.response_start_token) :]
            if response_end_idx != -1:
                content = content[:response_end_idx]
            return DeltaMessage(
                reasoning_content=reasoning_content,
                content=content if content else None,
            )
        elif self.end_token_id in previous_token_ids:
            # </think> in previous, thinking content ends
            content = delta_text
            if self.response_start_token_id in delta_token_ids:
                content = content.lstrip("\n")
                response_start_idx = content.find(self.response_start_token)
                content = content[response_start_idx + len(self.response_start_token) :]
                # if have </response>, remove it
                response_end_idx = content.rfind(self.response_end_token)
                if response_end_idx != -1:
                    content = content[:response_end_idx]
            elif self.response_end_token_id in delta_token_ids:
                response_end_idx = content.rfind(self.response_end_token)
                content = content[:response_end_idx]
            # remove \n after </think>  or </response>
            if previous_token_ids[-1] in self.parser_token_ids and (
                len(delta_token_ids) > 0 and delta_token_ids[0] == self.newline_token_id
            ):
                content = content.lstrip("\n")
            # remove \n after </think>\n
            if (
                len(previous_token_ids) > 1
                and previous_token_ids[-2] == self.end_token_id
            ) and (
                len(delta_token_ids) > 0 and delta_token_ids[0] == self.newline_token_id
            ):
                content = content.lstrip("\n")

            return DeltaMessage(content=content if content else None)
        else:
            # no </think> in previous or delta, reasoning content continues
            return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.
        The Ernie45 thinking model ouput format is
            abc\n</think>\n\n\n<response>\ndef\n</response>\n
        or  abc\n</think>\ndef
        - 'abc' goes to reasoning_content
        - 'def' goes to content
        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """
        reasoning_content, content = super().extract_reasoning_content(
            model_output, request
        )
        if content:
            start_idx = content.find(self.response_start_token)
            end_idx = content.rfind(self.response_end_token)
            # Simultaneously existing and in the correct order
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                content = content[start_idx + len(self.response_start_token) : end_idx]
        final_content = content or None

        return reasoning_content, final_content
