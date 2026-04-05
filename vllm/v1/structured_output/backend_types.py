# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import functools
import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.tokenizers import TokenizerLike
else:
    VllmConfig = object
    TokenizerLike = object


@functools.lru_cache(maxsize=1)
def _get_fill_bitmask_executor(max_workers: int) -> ThreadPoolExecutor:
    """Get or create a ThreadPoolExecutor with the specified max_workers."""
    return ThreadPoolExecutor(max_workers=max_workers)


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


class StructuredOutputGrammar(ABC):
    """Request-level backend for structured output requests."""

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
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """
        Validates the provided tokens against the grammar.
        Will not advance the FSM.

        Args:
            tokens (list[int]): A list of token IDs to validate.

        Returns:
            list[int]: A list of accepted token IDs. Will be a prefix
                of the input tokens, and empty if none are accepted.
        """

    @abstractmethod
    def rollback(self, num_tokens: int) -> None:
        """
        Rolls back the state of the grammar by a specified number of tokens.
        Will also revert counters for the number of processed tokens.

        Args:
            num_tokens (int): The number of tokens to roll back.
        """

    @abstractmethod
    def fill_bitmask(self, bitmask: "torch.Tensor", batch_index: int) -> None:
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
    def reset(self):
        """
        Resets the state of the structured output grammar.
        """


@dataclass
class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests."""

    vllm_config: VllmConfig
    tokenizer: TokenizerLike
    vocab_size: int

    # Parallel bitmask filling configuration - override in subclasses if needed
    fill_bitmask_parallel_threshold: int = 128
    fill_bitmask_batch_size: int = 16
    fill_bitmask_max_workers: int | None = None  # None = auto-detect

    @property
    def max_num_spec_tokens(self) -> int:
        """Get max speculative tokens from config."""
        if self.vllm_config.speculative_config is not None:
            return self.vllm_config.speculative_config.num_speculative_tokens
        return 0

    @abstractmethod
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
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
    def allocate_token_bitmask(self, max_num_seqs: int) -> "torch.Tensor":
        """
        Allocates a token bitmask for the specified maximum number of sequences.

        Args:
            max_num_seqs (int): The maximum number of sequences for which
                to allocate the bitmask.
        """

    @abstractmethod
    def destroy(self):
        """
        Backend-specific cleanup.
        """

    def fill_bitmasks_batch(
        self,
        requests: list[tuple[StructuredOutputGrammar, int, bool, list[int]]],
        bitmask: "torch.Tensor",
        full_mask: "torch.Tensor",
    ) -> None:
        """Fill bitmasks for a batch of requests.

        Default implementation uses ThreadPoolExecutor for large non-speculative
        batches (matching previous behavior). Guidance-based backends override
        with native Rust-level parallelism.

        Args:
            requests: List of (grammar, start_index, apply_bitmask, spec_tokens)
                      tuples. For non-speculative requests, spec_tokens will be
                      empty.
            bitmask: The bitmask tensor to fill
            full_mask: Scalar tensor with value -1 for non-constrained positions
        """
        # Check if any request has speculative tokens
        has_spec_tokens = any(spec_tokens for _, _, _, spec_tokens in requests)

        # Use parallel processing for large non-speculative batches
        if len(requests) > self.fill_bitmask_parallel_threshold and not has_spec_tokens:
            self._fill_bitmasks_batch_parallel(requests, bitmask, full_mask)
        else:
            self._fill_bitmasks_batch_serial(requests, bitmask, full_mask)

    def _fill_bitmasks_batch_serial(
        self,
        requests: list[tuple[StructuredOutputGrammar, int, bool, list[int]]],
        bitmask: "torch.Tensor",
        full_mask: "torch.Tensor",
    ) -> None:
        """Serial implementation of fill_bitmasks_batch."""
        for grammar, start_index, apply_bitmask, spec_tokens in requests:
            num_positions = len(spec_tokens) + 1

            if not apply_bitmask or grammar.is_terminated():
                for i in range(num_positions):
                    bitmask[start_index + i].copy_(full_mask)
                continue

            # Rollback pattern: accept then undo
            state_advancements = 0
            for i, token in enumerate(spec_tokens + [None]):
                if grammar.is_terminated():
                    bitmask[start_index + i].copy_(full_mask)
                else:
                    grammar.fill_bitmask(bitmask, start_index + i)
                    if token is not None:
                        grammar.accept_tokens("", [token])
                        state_advancements += 1
            if state_advancements > 0:
                grammar.rollback(state_advancements)

    def _fill_bitmasks_batch_parallel(
        self,
        requests: list[tuple[StructuredOutputGrammar, int, bool, list[int]]],
        bitmask: "torch.Tensor",
        full_mask: "torch.Tensor",
    ) -> None:
        """Parallel implementation using ThreadPoolExecutor for large batches."""

        def fill_chunk(
            chunk: list[tuple[StructuredOutputGrammar, int, bool, list[int]]],
        ) -> None:
            for grammar, start_index, apply_bitmask, _ in chunk:
                if apply_bitmask and not grammar.is_terminated():
                    grammar.fill_bitmask(bitmask, start_index)
                else:
                    bitmask[start_index].copy_(full_mask)

        if self.fill_bitmask_max_workers is None:
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
        else:
            max_workers = self.fill_bitmask_max_workers

        executor = _get_fill_bitmask_executor(max_workers)
        futures = []

        # Split into chunks and submit to executor
        batch_size = self.fill_bitmask_batch_size
        for i in range(0, len(requests), batch_size):
            chunk = requests[i : i + batch_size]
            futures.append(executor.submit(fill_chunk, chunk))

        # Wait for all to complete
        for future in futures:
            future.result()

    def accept_tokens_batch(
        self,
        requests: list[tuple[StructuredOutputGrammar, list[int]]],
    ) -> list[bool]:
        """Accept tokens for a batch of requests.

        Default implementation processes serially. Guidance-based backends
        override with native Rust-level parallelism.

        Args:
            requests: List of (grammar, token_ids) tuples. token_ids can be
                      a single token or multiple tokens (for speculative decoding).

        Returns:
            List of success/failure booleans for each request.
        """
        results = []
        for grammar, token_ids in requests:
            result = grammar.accept_tokens("", token_ids)
            results.append(result)
        return results
