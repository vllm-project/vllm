# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from itertools import islice
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
else:
    ChatCompletionRequest = Any
    ResponsesRequest = Any


class BaseThinkingReasoningParser(ReasoningParser):
    """
    Base class for reasoning parsers that use thinking tokens.

    This class provides common functionality for parsers that use start and end
    tokens to delimit reasoning content (e.g., <think>...</think>,
    <seed:think>...</seed:think>).
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

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        return self.end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract the content after the end tokens."""
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
        """
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [self.start_token_id, self.end_token_id]):
            return None

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=reasoning, content=content if content else None)
            elif self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            else:
                return DeltaMessage(reasoning=delta_text)
                
        elif self.start_token_id in delta_token_ids:
            if self.end_token_id in delta_token_ids:
                start_index = delta_text.find(self.start_token)
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(reasoning=reasoning, content=content if content else None)
            else:
                return DeltaMessage(reasoning=delta_text)
        else:
            return DeltaMessage(content=delta_text)

    def extract_reasoning(self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
                         ) -> tuple[str | None, str]:
        """
        Extract reasoning and content from model output, supporting both standard tags and escaped <\think> tags.
        """
        if not model_output or not self.start_token or not self.end_token:
            return None, model_output.strip()

        # Constructing robust regex pattern to handle escaped variations (e.g., <\think>)
        start_pattern = re.escape(self.start_token).replace(r'\<', r'\<\\?')
        end_pattern = re.escape(self.end_token).replace(r'\<', r'\<\\?').replace(r'\/', r'\\?\/')
        full_pattern = fr"({start_pattern})(.*?)({end_pattern})"

        # Using finditer for safer substring extraction without relying on str.index()
        matches = list(re.finditer(full_pattern, model_output, re.DOTALL))
        
        if not matches:
            return None, model_output.strip()

        reasoning_parts = []
        content_parts = []
        last_end = 0

        for match in matches:
            _ = match.group(1)
            reasoning_text = match.group(2)
            
            # Text before the start tag belongs to main content
            content_parts.append(model_output[last_end : match.start()])
            # Text inside tags belongs to reasoning
            reasoning_parts.append(reasoning_text.strip())
            # Update cursor
            last_end = match.end()

        # Add remaining trailing text to main content
        content_parts.append(model_output[last_end:].strip())

        final_reasoning = "\n".join(reasoning_parts).strip()
        final_content = "".join(content_parts).strip()

        return final_reasoning if final_reasoning else None, final_content

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count tokens that fall within start/end thinking markers."""
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
        
