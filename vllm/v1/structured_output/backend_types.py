# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.transformers_utils.tokenizer import AnyTokenizer


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests."""

    @abstractmethod
    def jump_forward_string(self) -> str | None:
        """
        Get jump forward string and returns its tokens and string

        Returns:
            str: Optional jump forward string
        """

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """
        Determines whether the provided tokens are accepted for the
        given request.

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        """
        Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill
        """

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.
        """

    @abstractmethod
    def find_token_divergence(
        self,
        request_id: str,
        prev_tokens: list[int],
        combined_tokens: list[int],
    ) -> int:
        """
        Finds the index where two token sequences diverge.
        Note that each grammar should handle its FSM rollback accordingly.

        Args:
            request_id: The unique identifier for the request.
            prev_tokens: Original token sequence
            combined_tokens: New token sequence that
              should start with prev_tokens

        Returns:
            int: Index where the sequences diverge
        """


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests.

    Make sure that all subclasses are also dataclasses."""
    vllm_config: VllmConfig
    tokenizer: AnyTokenizer
    vocab_size: int

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        """
        Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
              output request.
            grammar_spec (str): The grammar specification to compile.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.
        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> None:
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
              to allocate the bitmask.
        """

    @abstractmethod
    def encode_with_jump(
        self,
        output_token_ids: list[int],
        jump_forward_string: str,
    ) -> list[int]:
        """
        Handle retokenization with the jump forward string and
        returns the new tokens and the number of previous tokens to replace.

        Args:
            request_id: The unique identifier for the request.
            jump_forward_string: The string to jump forward with

        Returns:
            list[int]: Returns list of new tokens
                including the jump forward string.
        """
