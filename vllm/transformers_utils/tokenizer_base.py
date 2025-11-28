# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import warnings
from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import Self, runtime_checkable

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


def __getattr__(name: str):
    # TODO: Move TokenizerLike into `tokenizer.py`
    # and move TokenizerRegistry into `registry.py` with a deprecation
    if name == "TokenizerBase":
        warnings.warn(
            "`vllm.transformers_utils.tokenizer_base.TokenizerBase` has been moved to "
            "`vllm.transformers_utils.tokenizer.TokenizerLike`. "
            "The old name will be removed in v0.13.",
            DeprecationWarning,
            stacklevel=2,
        )

        return TokenizerLike

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@runtime_checkable
class TokenizerLike(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        /,
        *,
        revision: str | None = None,
    ) -> "Self":
        raise NotImplementedError

    @property
    def all_special_tokens(self) -> list[str]:
        raise NotImplementedError

    @property
    def all_special_ids(self) -> list[int]:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def is_fast(self) -> bool:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def max_token_id(self) -> int:
        raise NotImplementedError

    @property
    def truncation_side(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str] | list[int],
        text_pair: str | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[int]:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        raise NotImplementedError


class TokenizerRegistry:
    # Tokenizer name -> (tokenizer module, tokenizer class)
    REGISTRY: dict[str, tuple[str, str]] = {}

    @staticmethod
    def register(name: str, module: str, class_name: str) -> None:
        TokenizerRegistry.REGISTRY[name] = (module, class_name)

    @staticmethod
    def get_tokenizer(
        tokenizer_name: str,
        *args,
        **kwargs,
    ) -> TokenizerLike:
        tokenizer_cls = TokenizerRegistry.REGISTRY.get(tokenizer_name)
        if tokenizer_cls is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found.")

        tokenizer_module = importlib.import_module(tokenizer_cls[0])
        class_ = getattr(tokenizer_module, tokenizer_cls[1])
        return class_.from_pretrained(*args, **kwargs)
