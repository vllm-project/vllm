# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.tokenizers import TokenizerLike, TokenizerRegistry


class TestTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "TestTokenizer":
        return TestTokenizer()  # type: ignore

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1


def test_customized_tokenizer():
    TokenizerRegistry.register(
        "test_tokenizer",
        "tests.transformers_utils.test_tokenizer_registry",
        "TestTokenizer",
    )

    tokenizer = TokenizerRegistry.get_tokenizer("test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1

    tokenizer = get_tokenizer("test_tokenizer", tokenizer_mode="custom")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
