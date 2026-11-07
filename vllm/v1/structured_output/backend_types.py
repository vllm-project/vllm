# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformers import PythonBackend, TokenizersBackend

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
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


def decode_prefill_tokens(tokenizer: "TokenizerLike", tokens: list[int]) -> str:
    if isinstance(tokenizer, (PythonBackend, TokenizersBackend)):
        return tokenizer.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    return tokenizer.decode(tokens, skip_special_tokens=False)


class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests."""

    def accept_prefix(self, prefix: str, tokenizer: "TokenizerLike") -> bool:
        """消费可以无损表示为 token 的已有文本。"""
        tokens = tokenizer.encode(prefix, add_special_tokens=False)
        if decode_prefill_tokens(tokenizer, tokens) != prefix or any(
            t in tokenizer.all_special_ids for t in tokens
        ):
            raise VLLMValidationError(
                "The structured output prefill cannot be represented exactly "
                "by this backend's tokenizer."
            )
        if len(self.validate_tokens(tokens)) != len(tokens):
            return False
        return self.accept_tokens("prefill", tokens)

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

    def compile_grammar_with_prefix(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        stop_token_ids: set[int] | None = None,
        prefix: str | None = None,
    ) -> StructuredOutputGrammar:
        grammar = self.compile_grammar(request_type, grammar_spec, stop_token_ids)
        if prefix and not grammar.accept_prefix(prefix, self.tokenizer):
            raise VLLMValidationError(
                "The assistant prefill is not a valid prefix of the structured "
                "output constraint.",
                parameter="structured_output_prefill",
            )
        return grammar

    @abstractmethod
    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        stop_token_ids: set[int] | None = None,
    ) -> StructuredOutputGrammar:
        """Compiles a grammar specification into a structured output grammar.

        Args:
            request_type (StructuredOutputOptions): The type of structured
                output request.
            grammar_spec (str): The grammar specification to compile.
            stop_token_ids (set[int] | None): The request's EOS and user
                stop-token ids (``SamplingParams.all_stop_token_ids``), masked
                until the grammar terminates.

        Returns:
            StructuredOutputGrammar: The compiled structured output grammar.

        """

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int) -> "torch.Tensor":
        """Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.

        """

    @abstractmethod
    def destroy(self):
        """Backend-specific cleanup."""


def create_structured_output_backend(
    config: "VllmConfig", tokenizer: "TokenizerLike", name: str
) -> StructuredOutputBackend:
    from vllm.v1.structured_output.backend_guidance import GuidanceBackend
    from vllm.v1.structured_output.backend_lm_format_enforcer import (
        LMFormatEnforcerBackend,
    )
    from vllm.v1.structured_output.backend_outlines import OutlinesBackend
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

    backends: dict[
        str, Callable[[VllmConfig, TokenizerLike, int], StructuredOutputBackend]
    ] = {
        "xgrammar": XgrammarBackend,
        "guidance": GuidanceBackend,
        "outlines": OutlinesBackend,
        "lm-format-enforcer": LMFormatEnforcerBackend,
    }
    if name not in backends:
        raise ValueError(f"Unsupported structured output backend: {name}")
    return backends[name](config, tokenizer, config.model_config.get_vocab_size())
