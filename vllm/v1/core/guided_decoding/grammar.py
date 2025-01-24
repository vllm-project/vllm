from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Generic, Literal, TypeVar, get_args,
                    overload)

from vllm.utils import LazyLoader

if TYPE_CHECKING:
    import torch
    import xgrammar as xgr
    from typing_extensions import LiteralString, Self
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    torch = LazyLoader("torch", globals(), "torch")

T = TypeVar("T", bound=str)


class Grammar(ABC, Generic[T]):
    finished: bool = False

    @abstractmethod
    def accept_token(self, token: int) -> bool:
        """Whether to accept the token and advance the machine state."""

    @abstractmethod
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        """Fill the bitmask for the token at the given index."""

    @abstractmethod
    def allocate_bitmask(self, batch_size: int,
                         vocab_size: int) -> torch.Tensor:
        """Allocate a bitmask for the given batch size and vocabulary size."""

    @staticmethod
    @abstractmethod
    def apply_bitmask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        """Apply the bitmask to the logits."""

    @abstractmethod
    def reset(self):
        """Reset the machine state."""

    @abstractmethod
    def copy(self) -> Self:
        """Copy the grammar object."""

    def __copy__(self):
        return self.copy()

    _registry: dict[str, type[Grammar[T]]] = {}
    _backend: T

    def __init_subclass__(cls):
        if not hasattr(cls, '__orig_bases__'):
            raise TypeError(
                f"Class {cls.__qualname__} must be a subclass of GrammarObject"
            )

        backend = None
        for base in cls.__orig_bases__:
            if (origin := get_args(base)) and issubclass(
                    base.__origin__, Grammar):
                backend = get_args(origin[0])[0]
                break

        if backend is None:
            raise TypeError(
                f"Class {cls.__qualname__} must specify backend as Literal type"
            )

        if backend in cls._registry:
            name = cls._registry[backend].__qualname__
            raise ValueError(
                f"Backend '{backend}' is already registered to {name}")

        # Set the backend value from the Literal type
        cls._backend = backend
        cls._registry[backend] = cls

    @overload
    @classmethod
    def from_backend(
        cls,
        backend: Literal["xgrammar"] = ...,
        *,
        matcher: xgr.GrammarMatcher = ...,
        vocab_size: int = ...,
        ctx: xgr.CompiledGrammar = ...,
    ) -> XGrammar:
        ...

    @overload
    @classmethod
    def from_backend(cls,
                     backend: LiteralString = ...,
                     **kwargs: Any) -> Grammar:
        ...

    @classmethod
    def from_backend(cls, backend: LiteralString = "xgrammar", **kwargs: Any) -> Grammar[T]:
        grammar_cls = cls._registry.get(backend)
        if grammar_cls is None:
            raise ValueError(
                f"No grammar implementation registered for '{backend}'")
        return grammar_cls(**kwargs)


class XGrammar(Grammar[Literal["xgrammar"]]):
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

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
        # Note: In this method, if the tensors have different dimensions
        # on CPU device fails, but on GPU it runs without error. Hence the
        # unsqueeze above for scores, to match the token bitmask shape
        xgr.apply_token_bitmask_inplace(logits, vocab_mask)

    def reset(self):
        self.matcher.reset()

    def copy(self):
        return XGrammar(matcher=xgr.GrammarMatcher(self.ctx),
                        vocab_size=self.vocab_size,
                        ctx=self.ctx)
