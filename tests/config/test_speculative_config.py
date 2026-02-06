# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM


class _DummyDraftModelConfig:
    def __init__(self, model: str, model_type: str, architectures: list[str]):
        self.model = model
        self.hf_config = SimpleNamespace(
            model_type=model_type,
            architectures=architectures,
        )
        self.hf_text_config = self.hf_config
        self.architectures = architectures
        self.max_model_len = 4096

    def verify_with_parallel_config(self, parallel_config: ParallelConfig) -> None:
        del parallel_config


def _make_target_model_config(model_type: str = "qwen"):
    return SimpleNamespace(
        model="dummy-target",
        tokenizer="dummy-target",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path=None,
        allowed_media_domains=None,
        dtype="float16",
        seed=0,
        revision=None,
        tokenizer_revision=None,
        max_model_len=4096,
        quantization=None,
        enforce_eager=False,
        max_logprobs=20,
        config_format="hf",
        hf_text_config=SimpleNamespace(model_type=model_type),
        get_vocab_size=lambda: 1000,
    )


def _model_config_factory(
    model_type: str = "dummy",
    architectures: list[str] | None = None,
):
    def _factory(*args, **kwargs):
        del args
        return _DummyDraftModelConfig(
            model=kwargs["model"],
            model_type=model_type,
            architectures=architectures or ["DummyForCausalLM"],
        )

    return _factory


def test_dflash_method_is_supported_in_speculative_config():
    with patch(
        "vllm.config.speculative.ModelConfig",
        new=_model_config_factory(),
    ):
        config = SpeculativeConfig(
            method="dflash",
            model="z-lab/Qwen3-8B-DFlash-b16",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(),
            target_parallel_config=ParallelConfig(),
        )
    assert config.method == "dflash"


def test_dflash_method_auto_detects_from_model_name():
    with patch(
        "vllm.config.speculative.ModelConfig",
        new=_model_config_factory(),
    ):
        config = SpeculativeConfig(
            model="z-lab/Qwen3-8B-DFlash-b16",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(),
            target_parallel_config=ParallelConfig(),
        )
    assert config.method == "dflash"


def test_dflash_method_auto_detects_from_draft_architecture():
    with patch(
        "vllm.config.speculative.ModelConfig",
        new=_model_config_factory(architectures=["DFlashDraftModel"]),
    ):
        config = SpeculativeConfig(
            model="org/draft-model-without-name-hint",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(),
            target_parallel_config=ParallelConfig(),
        )
    assert config.method == "dflash"


def test_dflash_target_model_validation():
    with (
        patch(
            "vllm.config.speculative.ModelConfig",
            new=_model_config_factory(),
        ),
        pytest.raises(ValueError, match="dflash is only supported"),
    ):
        SpeculativeConfig(
            method="dflash",
            model="z-lab/Qwen3-8B-DFlash-b16",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(model_type="opt"),
            target_parallel_config=ParallelConfig(),
        )


def test_dflash_hash_differs_from_non_eagle3_method():
    ngram_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    with patch(
        "vllm.config.speculative.ModelConfig",
        new=_model_config_factory(),
    ):
        dflash_config = SpeculativeConfig(
            method="dflash",
            model="org/dflash-draft",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(),
            target_parallel_config=ParallelConfig(),
        )
    assert dflash_config.compute_hash() != ngram_config.compute_hash()


def test_dflash_architecture_is_registered():
    assert "DFlashDraftModel" in ModelRegistry.get_supported_archs()
    model_cls = ModelRegistry._try_load_model_cls("DFlashDraftModel")
    assert model_cls is DFlashQwen3ForCausalLM


def test_use_eagle_returns_true_for_dflash():
    with patch(
        "vllm.config.speculative.ModelConfig",
        new=_model_config_factory(),
    ):
        config = SpeculativeConfig(
            method="dflash",
            model="org/dflash-draft",
            num_speculative_tokens=1,
            target_model_config=_make_target_model_config(),
            target_parallel_config=ParallelConfig(),
        )
    assert config.use_eagle()


def test_dflash_aux_layers_from_dflash_config():
    model = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    model.config = SimpleNamespace(
        dflash_config={"layer_ids": [31, 33, 35]},
        target_layer_count=36,
    )
    assert model.get_eagle3_aux_hidden_state_layers() == (31, 33, 35)


def test_dflash_aux_layers_fallback_to_eagle_aux_layer_ids():
    model = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    model.config = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=[7, 11, 15],
        target_layer_count=36,
    )
    assert model.get_eagle3_aux_hidden_state_layers() == (7, 11, 15)


def test_dflash_aux_layers_fallback_to_target_layer_count_default():
    model = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    model.config = SimpleNamespace(target_layer_count=8)
    assert model.get_eagle3_aux_hidden_state_layers() == (2, 4, 5)


def test_set_dflash_aux_layers_updates_dflash_config():
    model = DFlashQwen3ForCausalLM.__new__(DFlashQwen3ForCausalLM)
    model.config = SimpleNamespace(dflash_config={})
    model.set_aux_hidden_state_layers((1, 3, 5))
    assert model.config.dflash_config["layer_ids"] == [1, 3, 5]
