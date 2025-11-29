# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.tokenizer_base import TokenizerBase, TokenizerRegistry

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TestTokenizer(TokenizerBase):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "TestTokenizer":
        return TestTokenizer()

    @property
    def all_special_tokens(self) -> list[str]:
        raise NotImplementedError()

    @property
    def all_special_ids(self) -> list[int]:
        raise NotImplementedError()

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    def max_token_id(self) -> int:
        raise NotImplementedError()

    @property
    def truncation_side(self) -> str:
        raise NotImplementedError()

    def __call__(
        self,
        text: str | list[str] | list[int],
        text_pair: str | None = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ):
        raise NotImplementedError()

    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def get_added_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        raise NotImplementedError()

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        raise NotImplementedError()

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[int]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError()

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = True) -> str:
        raise NotImplementedError()

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        raise NotImplementedError()


def test_customized_tokenizer():
    TokenizerRegistry.register(
        "test_tokenizer", "tests.tokenization.test_tokenizer_registry", "TestTokenizer"
    )

    tokenizer = TokenizerRegistry.get_tokenizer("test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1

    tokenizer = get_tokenizer("test_tokenizer", tokenizer_mode="custom")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
