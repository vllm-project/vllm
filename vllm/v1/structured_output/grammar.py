# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils import LazyLoader

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class StructuredOutputOptions(enum.Enum):
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()


StructuredOutputKey = tuple[StructuredOutputOptions, str]


@dataclass
class Grammar:
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # TODO: support max_rollback_tokens
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.", request_id, token)
                return False
            self.num_processed_tokens += 1
        return True

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> bool:
        return self.matcher.fill_next_token_bitmask(bitmask, idx)

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()

    def __copy__(self):
        return Grammar(
            matcher=xgr.GrammarMatcher(self.ctx),
            vocab_size=self.vocab_size,
            ctx=self.ctx,
        )
