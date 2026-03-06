# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import (
        ResponsesRequest,
    )
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base class for reasoning parsers that use thinking tokens.

    This class provides common functionality for parsers that use start and end
    tokens to delimit reasoning content (
        e.g., <think>...</think>, <seed:think>...</seed:think>).

    Subclasses must implement the start and end tokens via abstract
    properties.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        raise NotImplementedError

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        raise NotImplementedError

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        if not self.start_token or not self.end_token:
            raise ValueError("start_token and end_token must be defined in subclasses")

        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)
        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} reasoning parser could not locate "
                "think start/end tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        end_token_id = self.end_token_id
        return end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract the content after the end tokens
        """
        if self.end_token_id not in islice(input_ids, 0, max(0, len(input_ids) - 1)):
            return []
        else:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning_streaming(
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
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        # Check if start token is present in previous or delta.
        # Keep compatibility with models that don't generate start tokens.
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                # start token in previous, end token in delta,
                # extract reasoning content
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            elif self.end_token_id in previous_token_ids:
                # start token in previous, end token in previous,
                # reasoning content continues
                return DeltaMessage(content=delta_text)
            else:
                # start token in previous, no end token in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                # start token in delta, end token in delta,
                # extract reasoning content
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning, content=content if content else None
                )
            else:
                # start token in delta, no end token in delta,
                # reasoning content continues
                return DeltaMessage(reasoning=delta_text)
        else:
            # not find thinking start token
            return DeltaMessage(content=delta_text)

import re  

class BaseThinkingReasoningParser(BaseReasoningParser):
    """Base parser for models that use  tags for reasoning."""

    start_token: str = ""
    end_token: str = ""

    def extract_reasoning_streaming(
        self, delta_text: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str]:
       
        ...

         def extract_reasoning(self, model_output: str, request: ChatCompletionRequest | ResponsesRequest) -> tuple[str | None, str]:
        """
        Extract reasoning and content from model output, supporting both standard  tags and escaped <\think> tags.
        
        Args:
            model_output: Raw output string from the model
            request: Original chat/completion request object
            
        Returns:
            Tuple of (reasoning_content, main_content):
            - reasoning_content: Extracted reasoning text (None if no tags found)
            - main_content: Remaining content without reasoning tags
        """
        # Guard clause to avoid errors from empty input/tokens
        if not model_output or not self.start_token or not self.end_token:
            return None, model_output.strip()

        # Regex pattern to match both standard and escaped tags (core fix)
        # - Matches  or <\think> for start token
        # - Matches  or <\/think> for end token
        start_pattern = re.escape(self.start_token).replace('<', r'<\\?')
        end_pattern = re.escape(self.end_token).replace('</', r'<\\?/')
        full_pattern = fr"({start_pattern})(.*?)({end_pattern})"

        # Extract all tag matches using non-greedy matching
        matches = re.findall(full_pattern, model_output, re.DOTALL)
        
        # Return full output if no tags are found
        if not matches:
            return None, model_output.strip()

        # Split content into reasoning (inside tags) and main content (outside tags)
        reasoning_parts = []
        content_parts = []
        last_end = 0

        for start_tag, reasoning_text, end_tag in matches:
            # Add content before current tag to main content
            content_parts.append(model_output[last_end:model_output.index(start_tag, last_end)])
            # Add reasoning text (inside tags) to reasoning parts
            reasoning_parts.append(reasoning_text.strip())
            # Update cursor to end of current end tag
            last_end = model_output.index(end_tag, last_end) + len(end_tag)

        # Add remaining content after last tag to main content
        content_parts.append(model_output[last_end:].strip())

        # Combine parts and strip whitespace for clean output
        final_reasoning = "\n".join(reasoning_parts).strip()
        final_content = "\n".join(content_parts).strip()

        # Return None if reasoning is empty, otherwise return the reasoning text
        return final_reasoning if final_reasoning else None, final_content
   
    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count tokens that fall within start/end thinking markers.

        Uses a depth counter so nested spans are handled safely and stray end
        tokens do not drive the counter negative.
        """
        count = 0
        depth = 0
        for token_id in token_ids:
            if token_id == self.start_token_id:
                depth += 1
                continue
            if token_id == self.end_token_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count
