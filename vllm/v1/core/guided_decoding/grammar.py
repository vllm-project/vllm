from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Generic, Literal, TypeVar, get_args,
                    overload)

from typing_extensions import Annotated, LiteralString
import torch
import xgrammar as xgr

if TYPE_CHECKING:
    from typing_extensions import Self

T = TypeVar("T", bound=Annotated[LiteralString, str])


class Grammar:
    finished: bool = False
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string for jump-forward decoding

    def __init__(self, matcher: xgr.GrammarMatcher, vocab_size: int,
                 ctx: xgr.CompiledGrammar) -> None:
        # TODO: support max_rollback_tokens
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx

    def accept_token(self, token: int) -> bool:
        # NOTE: accept_token will determines whether we accept this token
        # and will also update the machine state
        return self.matcher.accept_token(token)

    def allocate_bitmask(self, batch_size: int,
                         vocab_size: int) -> torch.Tensor:
        return xgr.allocate_token_bitmask(batch_size, vocab_size)

    # this should be ran in parallel with model decoding
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    @staticmethod
    def apply_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        xgr.apply_token_bitmask_inplace(logits, vocab_mask)

    def reset(self):
        self.matcher.reset()

    def copy(self):
        return Grammar(matcher=xgr.GrammarMatcher(self.ctx),
                        vocab_size=self.vocab_size,
                        ctx=self.ctx)

    def __copy__(self):
        return self.copy()
