# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    from transformers import BatchEncoding

    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TokenizerLike(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerLike":
        raise NotImplementedError

    def num_special_tokens_to_add(self) -> int:
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
    def pad_token_id(self) -> int:
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
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> "BatchEncoding":
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError

    @property
    def added_tokens_decoder(self) -> dict[int, Any]:
        raise NotImplementedError

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        raise NotImplementedError

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        raise NotImplementedError

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        raise NotImplementedError

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        raise NotImplementedError
