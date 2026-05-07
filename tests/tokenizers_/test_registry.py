# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers import registry
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


def test_deepseek_v3_overrides_incorrect_hub_tokenizer_class(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    vocab = {
        "MLA": 0,
        "Ġattention": 1,
        ",": 2,
        "Ġor": 3,
        "Ġmore": 4,
        "Ġspecifically": 5,
        ".ĊĊ": 6,
        "<｜begin▁of▁sentence｜>": 7,
        "<｜end▁of▁sentence｜>": 8,
    }
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizerFast",
                "legacy": True,
                "clean_up_tokenization_spaces": False,
                "bos_token": "<｜begin▁of▁sentence｜>",
                "eos_token": "<｜end▁of▁sentence｜>",
                "pad_token": "<｜end▁of▁sentence｜>",
                "unk_token": None,
            }
        )
    )

    monkeypatch.setattr(
        registry,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(model_type="deepseek_v3"),
    )
    registry.cached_get_tokenizer.cache_clear()
    registry.cached_resolve_tokenizer_args.cache_clear()

    tokenizer = get_tokenizer(str(tmp_path), trust_remote_code=True)

    assert tokenizer.decode([0, 1, 2, 3, 4, 5, 6]) == (
        "MLA attention, or more specifically.\n\n"
    )
