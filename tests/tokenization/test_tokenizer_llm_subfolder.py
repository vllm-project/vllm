# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.transformers_utils import tokenizer as tokenizer_module
from vllm.transformers_utils.tokenizer import get_tokenizer


class _DummyTokenizer:
    def __init__(self):
        self.all_special_ids: list[int] = []
        self.all_special_tokens: list[str] = []
        self.all_special_tokens_extended: list[str] = []
        self.special_tokens_map: dict[str, str] = {}
        self.vocab_size = 1

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 1

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def encode(self, *args: Any, **kwargs: Any) -> list[int]:
        return []


def test_tokenizer_prefers_llm_subfolder(monkeypatch):
    captured = {}

    def fake_file_exists(repo_id: str, file_name: str, **kwargs: Any) -> bool:
        return file_name == "llm/tokenizer.json"

    def fake_auto_from_pretrained(*args: Any, **kwargs: Any):
        captured["subfolder"] = kwargs.get("subfolder")
        return _DummyTokenizer()

    monkeypatch.setattr(tokenizer_module, "file_exists", fake_file_exists)
    monkeypatch.setattr(
        tokenizer_module.AutoTokenizer,
        "from_pretrained",
        classmethod(
            lambda cls, *args, **kwargs: fake_auto_from_pretrained(*args, **kwargs)
        ),
    )

    tokenizer = get_tokenizer("fake/model")

    assert tokenizer is not None
    assert captured["subfolder"] == "llm"
