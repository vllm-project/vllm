# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import types
from pathlib import Path
from unittest.mock import patch

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


def test_modelscope_remote_gguf_tokenizer_preserves_quant_selector(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/modelscope/repo"

    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")
    snapshot_module.snapshot_download = fake_snapshot_download
    hub_module = types.ModuleType("modelscope.hub")
    hub_module.snapshot_download = snapshot_module
    modelscope_module = types.ModuleType("modelscope")
    modelscope_module.hub = hub_module

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_MODELSCOPE", "True")
        m.setitem(sys.modules, "modelscope", modelscope_module)
        m.setitem(sys.modules, "modelscope.hub", hub_module)
        m.setitem(sys.modules, "modelscope.hub.snapshot_download", snapshot_module)

        with patch(
            "vllm.tokenizers.registry.get_gguf_file_path_from_hf",
            return_value="model.Q4_K_M.gguf",
        ):
            _, tokenizer_name, _, kwargs = resolve_tokenizer_args(
                "owner/repo-GGUF:Q4_K_M",
                revision="main",
                download_dir="/tmp/cache",
            )

    assert tokenizer_name == "/tmp/modelscope/repo"
    assert kwargs["gguf_file"] == "model.Q4_K_M.gguf"
    assert calls == [
        {
            "model_id": "owner/repo-GGUF",
            "cache_dir": "/tmp/cache",
            "revision": "main",
            "local_files_only": False,
            "ignore_file_pattern": None,
            "allow_patterns": "model.Q4_K_M.gguf",
        }
    ]
