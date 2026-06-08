# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path

import pytest

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.registry import (
    TokenizerRegistry,
    get_tokenizer,
    resolve_tokenizer_args,
)


class TestTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TestTokenizer":
        return TestTokenizer(path_or_repo_id)  # type: ignore

    def __init__(self, path_or_repo_id: str | Path) -> None:
        super().__init__()

        self.path_or_repo_id = path_or_repo_id

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int:
        return 2

    @property
    def is_fast(self) -> bool:
        return True


@pytest.mark.parametrize("runner_type", ["generate", "pooling"])
def test_resolve_tokenizer_args_idempotent(runner_type):
    tokenizer_mode, tokenizer_name, args, kwargs = resolve_tokenizer_args(
        "facebook/opt-125m",
        runner_type=runner_type,
    )

    assert (tokenizer_mode, tokenizer_name, args, kwargs) == resolve_tokenizer_args(
        tokenizer_name, *args, **kwargs
    )


def test_resolve_tokenizer_args_remote_gguf_base_tokenizer(monkeypatch):
    from vllm.tokenizers import registry

    def fake_resolve_gguf_tokenizer_source(tokenizer_name, revision=None):
        assert tokenizer_name == "org/model-GGUF:UD-IQ4_NL"
        assert revision == "gguf-rev"
        return "org/base"

    monkeypatch.setattr(
        registry,
        "resolve_gguf_tokenizer_source",
        fake_resolve_gguf_tokenizer_source,
    )

    tokenizer_mode, tokenizer_name, args, kwargs = resolve_tokenizer_args(
        "org/model-GGUF:UD-IQ4_NL",
        revision="gguf-rev",
    )

    assert tokenizer_mode == "hf"
    assert tokenizer_name == "org/base"
    assert args == ()
    assert "gguf_file" not in kwargs
    assert kwargs["revision"] is None


def test_resolve_tokenizer_args_local_gguf_base_tokenizer(monkeypatch):
    from vllm.tokenizers import registry

    def fake_resolve_gguf_tokenizer_source(tokenizer_name, revision=None):
        assert tokenizer_name == "/models/qwen.gguf"
        assert revision == "local-rev"
        return "org/base"

    monkeypatch.setattr(
        registry,
        "is_gguf",
        lambda tokenizer_name: tokenizer_name == "/models/qwen.gguf",
    )
    monkeypatch.setattr(
        registry,
        "check_gguf_file",
        lambda tokenizer_name: tokenizer_name == "/models/qwen.gguf",
    )
    monkeypatch.setattr(
        registry,
        "resolve_gguf_tokenizer_source",
        fake_resolve_gguf_tokenizer_source,
    )

    tokenizer_mode, tokenizer_name, args, kwargs = resolve_tokenizer_args(
        "/models/qwen.gguf",
        revision="local-rev",
    )

    assert tokenizer_mode == "hf"
    assert tokenizer_name == "org/base"
    assert args == ()
    assert "gguf_file" not in kwargs
    assert kwargs["revision"] is None


def test_customized_tokenizer():
    TokenizerRegistry.register("test_tokenizer", __name__, TestTokenizer.__name__)

    tokenizer = TokenizerRegistry.load_tokenizer("test_tokenizer", "abc")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2

    tokenizer = get_tokenizer("abc", tokenizer_mode="test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.path_or_repo_id == "abc"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.pad_token_id == 2
