# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TokenizerLike(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        /,
        *,
        revision: str | None = None,
    ) -> Self:
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
