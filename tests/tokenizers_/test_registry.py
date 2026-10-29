# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import AutoConfig, Qwen3_5MoeConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from vllm.tokenizers import TokenizerLike, registry
from vllm.tokenizers.registry import (
    TokenizerRegistry,
    cached_get_tokenizer,
    cached_resolve_tokenizer_args,
    cached_tokenizer_from_config,
    get_tokenizer,
    resolve_tokenizer_args,
    tokenizer_args_from_config,
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


@pytest.mark.parametrize("input_kwargs", [{}, {"mistral_format": False}])
def test_resolve_tokenizer_args_forces_hf_mistral_format_false(input_kwargs):
    resolved_mode, _, _, kwargs = resolve_tokenizer_args(
        "mistralai/Mistral-Nemo-Instruct-2407",
        tokenizer_mode="hf",
        **input_kwargs,
    )

    assert resolved_mode == "hf"
    assert kwargs["mistral_format"] is False


def test_resolve_tokenizer_args_rejects_hf_mistral_format_true():
    with pytest.raises(
        ValueError,
        match="mistral_format=True is not supported with tokenizer_mode='hf'",
    ):
        resolve_tokenizer_args(
            "mistralai/Mistral-Nemo-Instruct-2407",
            tokenizer_mode="hf",
            mistral_format=True,
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
        model=str(tmp_path),
        skip_tokenizer_init=False,
        tokenizer=str(tmp_path),
        runner_type="generate",
        tokenizer_mode="hf",
        tokenizer_revision=None,
        code_revision=None,
        trust_remote_code=True,
        hf_config=Qwen3_5MoeConfig(),
    )

    registered_config = CONFIG_MAPPING._extra_content.pop("qwen3_5_moe", None)
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()

    try:

        def fake_from_pretrained(path_or_repo_id: str, *args, **kwargs):
            passed_config = kwargs.pop("config")
            assert isinstance(passed_config, Qwen3_5MoeConfig)
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


@pytest.fixture
def config_tokenizer_backend(monkeypatch):
    """Keep config helpers and dispatch real; replace Hub/backend loading."""
    calls = []

    class Backend:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(is_fast=True)

    monkeypatch.setattr(registry, "get_config", lambda *a, **kw: None)
    monkeypatch.setattr(registry, "_maybe_register_hf_config", lambda config: None)
    monkeypatch.setattr(registry, "is_mistral_model_repo", lambda **kw: True)
    monkeypatch.setattr(registry, "any_pattern_in_repo_files", lambda **kw: True)
    monkeypatch.setattr(
        registry, "get_non_weight_snapshot_path", lambda name, **kw: name
    )
    monkeypatch.setattr(TokenizerRegistry, "load_tokenizer_cls", lambda mode: Backend)
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()
    yield calls
    cached_get_tokenizer.cache_clear()
    cached_resolve_tokenizer_args.cache_clear()


def _load_config_tokenizer(loader, config, **kwargs):
    result = loader(config, **kwargs)
    if loader is tokenizer_args_from_config:
        mode, name, args, resolved_kwargs = result
        return get_tokenizer(name, *args, tokenizer_mode=mode, **resolved_kwargs)
    return result


@pytest.mark.parametrize(
    "loader", (tokenizer_args_from_config, cached_tokenizer_from_config)
)
@pytest.mark.parametrize("mode", ["mistral", "auto"])
@pytest.mark.parametrize("code_revision", [None, "model-code"])
@pytest.mark.parametrize("overrides", [{}, {"code_revision": None}])
def test_config_mistral_does_not_forward_code_revision(
    loader, mode, code_revision, overrides, config_tokenizer_backend
):
    """HF code revisions must not reach MistralCommonBackend's strict kwargs."""
    config = SimpleNamespace(
        model="test-model",
        tokenizer="test-model",
        runner_type="generate",
        tokenizer_mode=mode,
        tokenizer_revision=None,
        code_revision=code_revision,
        trust_remote_code=False,
        skip_tokenizer_init=False,
    )
    _load_config_tokenizer(loader, config, **overrides)
    assert len(config_tokenizer_backend) == 1
    assert "code_revision" not in config_tokenizer_backend[0]


@pytest.mark.parametrize(
    "loader", (tokenizer_args_from_config, cached_tokenizer_from_config)
)
@pytest.mark.parametrize("missing", [False, True])
def test_config_tokenizer_without_code_revision(
    loader, missing, config_tokenizer_backend
):
    """Default or minimal configs must not introduce an HF-only plugin kwarg."""
    config = SimpleNamespace(
        model="test-model",
        tokenizer="test-model",
        runner_type="generate",
        tokenizer_mode="test_tokenizer",
        tokenizer_revision=None,
        trust_remote_code=False,
        skip_tokenizer_init=False,
    )
    if not missing:
        config.code_revision = None
    _load_config_tokenizer(loader, config)
    assert len(config_tokenizer_backend) == 1
    assert "code_revision" not in config_tokenizer_backend[0]


@pytest.mark.parametrize(
    "loader", (tokenizer_args_from_config, cached_tokenizer_from_config)
)
@pytest.mark.parametrize("mode", ["hf", "cohere", "kimi_k3", "inkling", "deepseek_v4"])
@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, "model-code"),
        ({"code_revision": None}, None),
        ({"code_revision": "override"}, "override"),
    ],
)
def test_config_hf_backed_tokenizer_preserves_code_revision(
    loader, mode, overrides, expected, config_tokenizer_backend
):
    """Filtering Mistral kwargs must preserve HF wrappers and caller overrides."""
    config = SimpleNamespace(
        model="test-model",
        tokenizer="test-model",
        runner_type="generate",
        tokenizer_mode=mode,
        tokenizer_revision=None,
        code_revision="model-code",
        trust_remote_code=True,
        skip_tokenizer_init=False,
    )
    _load_config_tokenizer(loader, config, **overrides)
    assert len(config_tokenizer_backend) == 1
    assert config_tokenizer_backend[0]["code_revision"] == expected


def test_hub_id_tokenizer_class_survives_local_snapshot():
    # transformers picks TokenizersBackend for this checkpoint by its Hub ID;
    # the class declared on the Hub (LlamaTokenizerFast) drops every space
    tokenizer = get_tokenizer("deepseek-ai/deepseek-coder-1.3b-base")
    text = "key 96727 value 4922"
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
