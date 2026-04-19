# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from vllm.tokenizers import TokenizerLike
from vllm.tokenizers.registry import (
    TokenizerRegistry,
    cached_get_tokenizer,
    cached_resolve_tokenizer_args,
    cached_tokenizer_from_config,
    get_tokenizer,
    resolve_tokenizer_args,
)
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig


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


def test_cached_tokenizer_from_config_registers_local_config(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe"}),
        encoding="utf-8",
    )

    model_config = SimpleNamespace(
        skip_tokenizer_init=False,
        tokenizer=str(tmp_path),
        runner_type="generate",
        tokenizer_mode="hf",
        tokenizer_revision=None,
        trust_remote_code=True,
        hf_config=Qwen3_5MoeConfig(),
    )

    registered_config = CONFIG_MAPPING._extra_content.pop("qwen3_5_moe", None)
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()

    try:
        def fake_from_pretrained(path_or_repo_id: str, *args, **kwargs):
            loaded_config = AutoConfig.from_pretrained(
                path_or_repo_id,
                trust_remote_code=False,
            )
            assert isinstance(loaded_config, Qwen3_5MoeConfig)
            return SimpleNamespace(is_fast=True)

        with (
            patch(
                "vllm.tokenizers.registry.logger.debug_once",
                lambda *args, **kwargs: None,
            ),
            patch(
                "vllm.tokenizers.hf.AutoTokenizer.from_pretrained",
                side_effect=fake_from_pretrained,
            ),
            patch(
                "vllm.tokenizers.hf.get_cached_tokenizer",
                side_effect=lambda tokenizer: tokenizer,
            ),
        ):
            tokenizer = cached_tokenizer_from_config(model_config)

        assert tokenizer.is_fast is True
    finally:
        cached_get_tokenizer.cache_clear()
        cached_resolve_tokenizer_args.cache_clear()
        CONFIG_MAPPING._extra_content.pop("qwen3_5_moe", None)
        if registered_config is not None:
            CONFIG_MAPPING._extra_content["qwen3_5_moe"] = registered_config
