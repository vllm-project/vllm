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
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig, Glm5NextConfig, PreTrainedConfig

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig


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
    hf_config = composed(_make_hf_config(model_type="deepseek_v3"))
    assert hf_config.architectures == ["DeepSeekMTPModel"]
    assert getattr(hf_config, "max_position_embeddings", None) != 1234


@pytest.mark.cpu_test
def test_none_overrides_fall_back_to_arch_mapping():
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    hf_config = composed(_make_hf_config(model_type="deepseek_v3"))
    assert hf_config.architectures == ["DeepSeekMTPModel"]


def _make_speculative_config(
    hf_kwargs: dict, architecture: str, **speculative_kwargs
) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(_make_hf_config(**hf_kwargs))
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, architecture)
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = hf_kwargs["model_type"]
    target.using_transformers_backend.return_value = False

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
def test_speculators_proposal_count_defaults_draft_tokens():
    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        {
            "speculators_model_type": "eagle3",
            "transformer_layer_config": {"model_type": "llama", "vocab_size": 32},
            "speculators_config": {"proposal_methods": [{"speculative_tokens": 3}]},
        }
    )
    config = _make_speculative_config(
        hf_config, "Eagle3LlamaForCausalLM", method="eagle3"
    )
    assert config.num_speculative_tokens == 3


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
    target_model_config.using_transformers_backend.return_value = False

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


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("model_type", "model_impl", "architecture"),
    [
        ("glm4_moe", "transformers", "TransformersMoEForMTP"),
        ("glm4_moe", "auto", "Glm4MoeMTPModel"),
        # Does not follow the Transformers MTP standard (no `num_mtp_layers`)
        ("qwen3_next", "transformers", "Qwen3NextMTP"),
    ],
)
def test_mtp_draft_architecture_follows_target_model_impl(
    tmp_path, model_type: str, model_impl: str, architecture: str
):
    """The MTP draft of a Transformers backend target uses the Transformers backend
    too if the model follows the Transformers MTP standard, while other targets keep
    their native MTP draft."""
    config = AutoConfig.for_model(model_type, num_nextn_predict_layers=1)
    config.architectures = [config.__class__.__name__.replace("Config", "ForCausalLM")]
    config.save_pretrained(tmp_path)
    target = ModelConfig(str(tmp_path), model_impl=model_impl, skip_tokenizer_init=True)

    speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=1,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )

    assert speculative_config.method == "mtp"
    draft = speculative_config.draft_model_config
    assert draft.architecture == architecture
    assert draft.hf_config.n_predict == 1
