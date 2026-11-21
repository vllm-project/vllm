# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.tokenizers import TokenizerLike
else:
    VllmConfig = object
    TokenizerLike = object


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


@dataclass(frozen=True)
class GrammarCompileOptions:
    """Request-level options that change how a grammar is compiled.

    Values are passed through unresolved: None means the backend uses
    its server-level setting.
    """

    disable_any_whitespace: bool | None = None

    @classmethod
    def from_params(cls, params: "StructuredOutputsParams") -> "GrammarCompileOptions":
        return cls(disable_any_whitespace=params.disable_any_whitespace)


def resolve_disable_any_whitespace(
    options: GrammarCompileOptions | None, server_value: bool
) -> bool:
    """The request-level value when set, otherwise the server-level one."""
    if options is None or options.disable_any_whitespace is None:
        return server_value
    return options.disable_any_whitespace


class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests."""

    @abstractmethod
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Determines whether the provided tokens are accepted for the
        given request.

        A false result must leave the grammar state unchanged.

        Args:
            request_id (str): The unique identifier for the request.
            tokens (list[int]): A list of token IDs to evaluate.

        Returns:
            bool: True if the tokens are accepted, False otherwise.

        """

    @abstractmethod
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Validates the provided tokens against the grammar.
        Will not advance the FSM.

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.

        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.

        """

    @abstractmethod
    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
        """Fills the bitmask for a specific batch index.

        Args:
            bitmask (torch.Tensor): The bitmask to fill
            batch_index (int): The index in the bitmask to fill

        """

    @abstractmethod
    def is_terminated(self) -> bool:
        """Checks whether the structured output process has terminated.

        Returns:
            bool: True if the process is terminated, False otherwise.

        """

    @abstractmethod
    def reset(self):
        """Resets the state of the structured output grammar."""


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests."""

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    @abstractmethod
    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        stop_token_ids: set[int] | None = None,
        options: GrammarCompileOptions | None = None,
    ) -> StructuredOutputGrammar:
        """Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
                output request.
            grammar_spec (str): The grammar specification to compile.
            stop_token_ids (set[int] | None): The request's EOS and user
                stop-token ids (`SamplingParams.all_stop_token_ids`), masked
                until the grammar terminates.
            options (GrammarCompileOptions | None): Request-level compile
                options. None, or a None field, uses the server-level setting.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.

        """

    @abstractmethod
    def destroy(self):
        """Backend-specific cleanup."""
