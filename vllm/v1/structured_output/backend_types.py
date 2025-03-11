# SPDX-License-Identifier: Apache-2.0

import enum
from abc import ABC, abstractmethod


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
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        pass

    @abstractmethod
    def fill_bitmask(self, bitmask: list[int], batch_index: int) -> None:
        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass


class StructuredOutputBackend(ABC):
    """Engine-level backend for structured output requests."""

    @abstractmethod
    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        pass

    @abstractmethod
    def allocate_token_bitmask(self, max_num_seqs: int):
        pass
