# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for draft config overrides used by SpeculativeConfig.

Callable ``hf_overrides`` on the target model config (e.g. the
``dummy_hf_overrides`` shrink used by ``tests/models/test_initialization.py``)
must also be applied when building the draft ``ModelConfig``. Otherwise a
draft belonging to a large target model is instantiated at full size even
when the target itself is shrunk — which is what kept spec-decode archs like
``EagleMistralLarge3ForCausalLM`` stuck at ``is_available_online=False``
("TODO: revert once figuring out OOM in CI").
"""

import functools
import json
from unittest.mock import MagicMock, patch

import pytest
from transformers import Glm5NextConfig, Lfm2Config, PreTrainedConfig, Qwen3Config

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _make_hf_config(**kwargs) -> PreTrainedConfig:
    defaults = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return PreTrainedConfig(**defaults)


@pytest.mark.cpu_test
def test_dict_overrides_are_not_forwarded_to_draft():
    """Dict overrides are target-specific key patches; the draft must get
    only the architecture-mapping override."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"max_position_embeddings": 1234}
    )
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
def test_none_overrides_fall_back_to_arch_mapping():
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    assert composed is SpeculativeConfig.hf_config_override


def _make_speculative_config(
    hf_kwargs: dict,
    architecture: str,
    **speculative_kwargs,
) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(_make_hf_config(**hf_kwargs))
    draft = MagicMock(
        hf_config=hf_config,
        architectures=hf_config.architectures,
        max_model_len=128,
        model="draft",
    )
    draft.registry.inspect_model_cls.return_value = (None, architecture)
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = hf_kwargs["model_type"]

    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
            model="draft",
            **speculative_kwargs,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("hf_kwargs", "n_predict", "architecture"),
    [
        (
            dict(
                model_type="deepseek_v4",
                num_nextn_predict_layers=3,
                dspark_block_size=5,
            ),
            3,
            "DSparkDraftModel",
        ),
        (
            dict(
                model_type="deepseek_v41",
                num_nextn_predict_layers=3,
                dspark_block_size=5,
            ),
            3,
            "DSparkV41DraftModel",
        ),
        (
            dict(
                model_type="gemma4_text",
                architectures=["Gemma4DSparkModel"],
                block_size=5,
            ),
            None,
            "Gemma4DSparkModel",
        ),
    ],
)
@pytest.mark.parametrize("speculative_kwargs", [{"num_speculative_tokens": 5}, {}])
def test_dspark_width_is_independent_of_mtp_stages(
    hf_kwargs, n_predict, architecture, speculative_kwargs
):
    config = _make_speculative_config(
        hf_kwargs, architecture, method="dspark", **speculative_kwargs
    )

    assert config.num_speculative_tokens == 5
    assert getattr(config.draft_model_config.hf_config, "n_predict", None) == n_predict
    assert config.draft_model_config.hf_config.architectures == [architecture]


def _lfm2_dspark_hf_kwargs(**overrides) -> dict:
    config = dict(
        model_type="qwen3",
        architectures=["Lfm2DSparkDraftModel"],
        hidden_size=2048,
        num_hidden_layers=4,
        vocab_size=128000,
        block_size=9,
        markov_rank=256,
        rope_is_neox_style=False,
        enable_confidence_head=True,
        dflash_config={
            "mask_token_id": 125017,
            "target_layer_ids": [2, 9, 17, 21, 27],
            "num_target_layers": 30,
        },
    )
    config.update(overrides)
    return config


@pytest.fixture
def liquid_model_dir(tmp_path_factory):
    # Keep "dspark" out of the path to exercise architecture-based detection.
    return tmp_path_factory.mktemp("liquid")


def _make_liquid_speculative_config(model_dir, hf_kwargs: dict, **speculative_kwargs):
    Lfm2Config(
        architectures=["Lfm2ForCausalLM"],
        hidden_size=2048,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=128000,
        max_position_embeddings=512,
    ).save_pretrained(model_dir / "target")
    Qwen3Config(**hf_kwargs).save_pretrained(model_dir / "liquid-draft")
    target = ModelConfig(model=str(model_dir / "target"), skip_tokenizer_init=True)
    return SpeculativeConfig(
        model=str(model_dir / "liquid-draft"),
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
        **speculative_kwargs,
    )


@pytest.mark.cpu_test
def test_lfm2_dspark_override_maps_architecture_before_registry():
    """The registry does not know ``Lfm2DSparkDraftModel``.

    ``ModelConfig`` inspects the registry right after applying
    ``hf_config_override``, so the mapping has to happen inside the override.
    """
    hf_config = SpeculativeConfig.hf_config_override(
        _make_hf_config(**_lfm2_dspark_hf_kwargs())
    )

    assert hf_config.architectures == ["Qwen3DSparkModel"]
    assert hf_config.model_type == "qwen3"
    assert hf_config.is_neox_style is False
    assert hf_config.confidence_head_with_markov is True


@pytest.mark.cpu_test
@pytest.mark.parametrize("with_markov", [True, False])
def test_lfm2_dspark_preserves_explicit_confidence_layout(with_markov):
    hf_config = SpeculativeConfig.hf_config_override(
        _make_hf_config(
            **_lfm2_dspark_hf_kwargs(confidence_head_with_markov=with_markov)
        )
    )

    assert hf_config.confidence_head_with_markov is with_markov


@pytest.mark.cpu_test
def test_lfm2_dspark_draft_model_config_loads_with_override(tmp_path, monkeypatch):
    """Build a real draft ModelConfig from a local Liquid-shaped config."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    config = _lfm2_dspark_hf_kwargs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        markov_rank=8,
        dflash_config={
            "mask_token_id": 3,
            "target_layer_ids": [0, 1],
            "num_target_layers": 2,
        },
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Lfm2DSparkDraftModel"):
        ModelConfig(model=str(tmp_path), runner="draft", skip_tokenizer_init=True)

    draft = ModelConfig(
        model=str(tmp_path),
        runner="draft",
        skip_tokenizer_init=True,
        hf_overrides=SpeculativeConfig.hf_config_override,
    )
    assert draft.architectures == ["Qwen3DSparkModel"]
    assert draft.architecture == "Qwen3DSparkModel"
    assert draft.hf_config.is_neox_style is False


@pytest.mark.cpu_test
def test_lfm2_dspark_resolves_to_qwen3_dense_drafter(liquid_model_dir):
    """Liquid's architecture must become the dense Qwen3 drafter.

    The model name here does not contain "dspark", so resolution has to come
    from the architecture. The unknown-architecture path would otherwise
    rewrite the draft to DeepSeek-V4.
    """
    config = _make_liquid_speculative_config(
        liquid_model_dir,
        _lfm2_dspark_hf_kwargs(),
    )
    hf_config = config.draft_model_config.hf_config

    assert config.method == "dspark"
    assert config.num_speculative_tokens == 9
    assert hf_config.model_type == "qwen3"
    assert hf_config.architectures == ["Qwen3DSparkModel"]
    assert hf_config.is_neox_style is False
    assert hf_config.hidden_size == 2048
    assert hf_config.num_hidden_layers == 4
    assert hf_config.vocab_size == 128000
    assert hf_config.block_size == 9
    assert hf_config.markov_rank == 256
    assert hf_config.enable_confidence_head is True
    assert hf_config.dflash_config == {
        "mask_token_id": 125017,
        "target_layer_ids": [2, 9, 17, 21, 27],
        "num_target_layers": 30,
    }


@pytest.mark.cpu_test
def test_lfm2_dspark_keeps_explicit_is_neox_style(liquid_model_dir):
    config = _make_liquid_speculative_config(
        liquid_model_dir,
        _lfm2_dspark_hf_kwargs(is_neox_style=True),
        method="dspark",
        num_speculative_tokens=9,
    )

    assert config.draft_model_config.hf_config.is_neox_style is True
    assert config.draft_model_config.hf_config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.cpu_test
def test_qwen3_dspark_does_not_copy_liquid_rope_field(liquid_model_dir):
    config = _make_liquid_speculative_config(
        liquid_model_dir,
        dict(
            model_type="qwen3",
            architectures=["DSparkDraftModel"],
            block_size=7,
            rope_is_neox_style=False,
        ),
        method="dspark",
    )

    hf_config = config.draft_model_config.hf_config
    assert config.num_speculative_tokens == 7
    assert hf_config.model_type == "qwen3"
    assert hf_config.architectures == ["Qwen3DSparkModel"]
    assert getattr(hf_config, "is_neox_style", None) is None
    assert getattr(hf_config, "confidence_head_with_markov", None) is None


@pytest.mark.cpu_test
def test_mtp_stages_are_independent_of_dspark_width():
    hf_kwargs = dict(
        model_type="deepseek_v4", num_nextn_predict_layers=3, dspark_block_size=5
    )
    config = _make_speculative_config(
        hf_kwargs, "DeepSeekV4MTPModel", method="mtp", num_speculative_tokens=6
    )

    assert config.num_speculative_tokens == 6
    assert config.draft_model_config.hf_config.n_predict == 3
    assert config.draft_model_config.hf_config.architectures == ["DeepSeekV4MTPModel"]

    with pytest.raises(ValueError, match="must be divisible by n_predict=3"):
        _make_speculative_config(
            hf_kwargs, "DeepSeekV4MTPModel", method="mtp", num_speculative_tokens=5
        )


@pytest.mark.cpu_test
def test_callable_overrides_reach_the_draft_config():
    """A callable override (config-to-config transform) composes with the
    architecture-mapping override and is applied to the draft config."""

    def shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        hf_config.num_hidden_layers = 1
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(shrink)
    assert composed is not SpeculativeConfig.hf_config_override

    out = composed(_make_hf_config())
    # The shrink transform must have been applied to the draft config.
    assert out.num_hidden_layers == 1


@pytest.mark.cpu_test
def test_arch_mapping_applies_before_callable_override():
    """The static arch-mapping override runs first, so the user callable
    observes (and may adjust) the post-mapping config."""
    seen_architectures: list[str] = []

    def record(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        seen_architectures.append(hf_config.architectures[0])
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(record)

    # MiMo is one of the arch-mapped model types: hf_config_override
    # rewrites architectures to ["MiMoMTPModel"].
    mimo = _make_hf_config(
        architectures=["MiMoForCausalLM"],
        model_type="mimo",
        num_nextn_predict_layers=1,
    )
    composed(mimo)
    assert seen_architectures == ["MiMoMTPModel"]


@pytest.mark.cpu_test
def test_inkling_override_exposes_all_mtp_depths():
    text_config = _make_hf_config(
        architectures=["InklingForCausalLM"],
        model_type="inkling_model",
        local_layer_ids=[1, 3],
    )
    config = _make_hf_config(
        architectures=["InklingForConditionalGeneration"],
        model_type="inkling_mm_model",
        text_config=text_config,
        mtp_config={
            "num_nextn_predict_layers": 8,
            "local_layer_ids": [0, 2, 4],
        },
    )

    out = SpeculativeConfig.hf_config_override(config)

    assert out is text_config
    assert out.model_type == "inkling_mtp"
    assert out.architectures == ["InklingMTPModel"]
    # Multi-module MTP: every checkpoint depth is exposed (module i drafts
    # speculative token i), no longer clamped to the first depth.
    assert out.n_predict == 8
    assert out.num_nextn_predict_layers == 8
    assert out.chain_hidden_post_norm is False
    assert out.local_layer_ids == [0, 2, 4]


def _module_level_shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
    hf_config.num_hidden_layers = 1
    return hf_config


@pytest.mark.cpu_test
def test_composed_override_is_picklable():
    """The draft ``ModelConfig`` is sent to spawned engine-core processes, so
    the composed override must be picklable. A nested local closure is not
    (it raised ``Can't get local object`` on DFlashDraftModel); a
    ``functools.partial`` over a module-referenceable static method is.
    Guard against regressing to a closure."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(_module_level_shrink)

    assert isinstance(composed, functools.partial)
    assert composed.func is SpeculativeConfig._apply_composed_hf_override

    out = composed(_make_hf_config())
    assert out.num_hidden_layers == 1


def _make_mtp_speculative_config(
    override: bool | None,
    checkpoint_value: bool,
) -> SpeculativeConfig:
    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=1,
        index_share_for_mtp_iteration=checkpoint_value,
    )
    draft_model_config = MagicMock(
        model="draft",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        return SpeculativeConfig(
            model="draft",
            method="mtp",
            num_speculative_tokens=1,
            index_share_for_mtp_iteration=override,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("override", "checkpoint_value", "expected"),
    [(None, True, True), (False, True, False), (True, False, True)],
)
def test_mtp_index_share_override(
    override: bool | None, checkpoint_value: bool, expected: bool
):
    speculative_config = _make_mtp_speculative_config(override, checkpoint_value)
    assert (
        speculative_config.draft_model_config.hf_config.index_share_for_mtp_iteration
        is expected
    )


@pytest.mark.cpu_test
def test_glm5_next_mtp_override_promotes_index_share():
    """The flag lives on text_config, but the MTP proposers read the draft config."""
    config = Glm5NextConfig(
        architectures=["Glm5NextForConditionalGeneration"],
        text_config={
            "num_nextn_predict_layers": 1,
            "index_share_for_mtp_iteration": True,
        },
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.architectures == ["Glm5NextMTPModel"]
    assert draft_config.n_predict == 1
    assert draft_config.index_share_for_mtp_iteration is True
