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
import logging
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from transformers import (
    AutoConfig,
    Glm5NextConfig,
    LlamaConfig,
    PreTrainedConfig,
    Qwen3Config,
)

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.config import patch_rope_parameters
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig


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
    hf_kwargs: dict,
    architecture: str,
    target_model: str = "target",
    **speculative_kwargs,
) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(_make_hf_config(**hf_kwargs))
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, architecture)
    target = MagicMock(
        model=target_model, max_model_len=128, quantization=None, hf_overrides={}
    )
    target.hf_config.model_type = hf_kwargs["model_type"]
    target.using_transformers_backend.return_value = False
    speculative_kwargs.setdefault("model", "draft")

    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
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


def _make_minimax_dspark_hf_kwargs() -> dict:
    return dict(
        model_type="minimax_m3_vl",
        architectures=["DSparkMiniMaxDraftModel"],
        quantization_config={
            "quant_method": "mxfp8",
            "ignored_layers": ["language_model.model.dspark.fc"],
        },
        text_config=_make_hf_config(
            architectures=["DSparkMiniMaxDraftModel"],
            model_type="",
            num_hidden_layers=5,
            dspark_block_size=7,
            dspark_markov_rank=256,
            dflash_config={"block_size": 7},
        ),
    )


@pytest.mark.cpu_test
def test_minimax_dspark_draft_promotes_text_config():
    """The MiniMax DSpark draft ships a VL wrapper config. It must keep its own
    draft class (not MiniMax MTP or DeepSeek DSpark), with the DSpark keys and
    the wrapper's quantization config on the promoted text config."""
    hf_kwargs = _make_minimax_dspark_hf_kwargs()
    config = _make_speculative_config(
        hf_kwargs, "DSparkMiniMaxDraftModel", method="dspark"
    )

    hf_config = config.draft_model_config.hf_config
    assert hf_config is hf_kwargs["text_config"]
    assert hf_config.architectures == ["DSparkMiniMaxDraftModel"]
    assert config.num_speculative_tokens == 7
    assert hf_config.quantization_config == hf_kwargs["quantization_config"]
    assert hf_config.markov_rank == 256
    assert hf_config.dflash_config["causal"] is False


@pytest.mark.cpu_test
def test_minimax_dspark_defaults_to_dspark_subfolder(tmp_path):
    """With only method="dspark", MiniMax-M3 loads the draft checkpoint shipped
    in the target's dspark/ subfolder, with its own quantization config."""
    (tmp_path / "dspark").mkdir()
    (tmp_path / "dspark" / "config.json").write_text("{}")
    config = _make_speculative_config(
        _make_minimax_dspark_hf_kwargs(),
        "DSparkMiniMaxDraftModel",
        target_model=str(tmp_path),
        model=None,
        method="dspark",
    )

    assert config.model == str(tmp_path / "dspark")
    assert config.quantization is None
    assert config.num_speculative_tokens == 7


@pytest.mark.cpu_test
def test_minimax_dspark_without_subfolder_raises(tmp_path):
    with pytest.raises(ValueError, match="no draft checkpoint in the 'dspark/'"):
        _make_speculative_config(
            _make_minimax_dspark_hf_kwargs(),
            "DSparkMiniMaxDraftModel",
            target_model=str(tmp_path),
            model=None,
            method="dspark",
        )


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


def _make_rope_speculative_config(
    tmp_path, draft_hf, target_hf, method, *, parallel_drafting=False
):
    draft = MagicMock(
        model=str(tmp_path),
        hf_config=draft_hf,
        architectures=draft_hf.architectures,
        max_model_len=128,
    )
    draft.registry.inspect_model_cls.return_value = (None, "Eagle3LlamaForCausalLM")
    draft.get_vocab_size.return_value = draft_hf.vocab_size
    target = MagicMock(
        hf_config=target_hf,
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )
    target.get_vocab_size.return_value = target_hf.vocab_size
    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
            model=str(tmp_path),
            method=method,
            num_speculative_tokens=3,
            parallel_drafting=parallel_drafting,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["eagle", "eagle3", "draft_model"])
@pytest.mark.parametrize(
    ("draft_rope", "target_theta", "expect_warning"),
    [
        ({}, 500000.0, True),
        ({}, 10000.0, False),
        ({"rope_theta": 10000.0}, 500000.0, False),
        ({"rope_theta": 500000.0}, 500000.0, False),
        (
            {"rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}},
            500000.0,
            False,
        ),
        (
            {"rope_scaling": {"rope_theta": 10000.0, "rope_type": "default"}},
            500000.0,
            False,
        ),
        (
            {},
            {
                "rope_theta": 1000000.0,
                "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
            False,
        ),
    ],
)
def test_eagle_draft_rope_diagnostics(
    tmp_path, caplog, monkeypatch, method, draft_rope, target_theta, expect_warning
):
    """Report the effective RoPE without changing the draft's training settings."""
    raw_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        **draft_rope,
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    draft_hf = LlamaConfig(**raw_config)
    original_rope = deepcopy(draft_hf.rope_parameters)
    target_hf = (
        PreTrainedConfig(rope_parameters=target_theta, vocab_size=draft_hf.vocab_size)
        if isinstance(target_theta, dict)
        else LlamaConfig(rope_theta=target_theta)
    )
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="vllm.config.speculative"):
        config = _make_rope_speculative_config(tmp_path, draft_hf, target_hf, method)

    assert config.draft_model_config.hf_config.rope_parameters == original_rope
    messages = [record.getMessage() for record in caplog.records]
    is_eagle = method in ("eagle", "eagle3")
    assert any("draft model RoPE" in message for message in messages) is is_eagle
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "RoPE" in record.getMessage()
    ]
    assert bool(warnings) is (expect_warning and is_eagle)
    if warnings:
        assert "10000" in warnings[0]
        assert "500000" in warnings[0]


@pytest.mark.cpu_test
@pytest.mark.parametrize("cached_config", [None, "invalid JSON", "[]"])
def test_eagle_rope_diagnostics_tolerate_unavailable_checkpoint(
    tmp_path, caplog, monkeypatch, cached_config
):
    """Diagnostics must work even if the source config is no longer readable."""
    if cached_config is not None:
        (tmp_path / "config.json").write_text(cached_config)
    draft_hf = LlamaConfig(architectures=["LlamaForCausalLM"])
    target_hf = LlamaConfig(rope_theta=500000.0)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="vllm.config.speculative"):
        config = _make_rope_speculative_config(tmp_path, draft_hf, target_hf, "eagle3")

    assert config.draft_model_config.hf_config.rope_parameters["rope_theta"] == 10000.0
    assert any("draft model RoPE" in record.getMessage() for record in caplog.records)


@pytest.mark.cpu_test
def test_eagle_rope_diagnostics_respect_runtime_override(tmp_path, caplog, monkeypatch):
    """A nondefault resolved base is not a missing-checkpoint fallback."""
    (tmp_path / "config.json").write_text('{"model_type": "llama"}')
    draft_hf = LlamaConfig(architectures=["LlamaForCausalLM"], rope_theta=750000.0)
    target_hf = LlamaConfig(rope_theta=500000.0)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="vllm.config.speculative"):
        config = _make_rope_speculative_config(tmp_path, draft_hf, target_hf, "eagle3")

    assert config.draft_model_config.hf_config.rope_parameters["rope_theta"] == 750000.0
    assert any("750000" in record.getMessage() for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)


@pytest.mark.cpu_test
@pytest.mark.parametrize("speculators_model_type", ["eagle3", "peagle"])
@pytest.mark.parametrize(
    ("rope_parameters", "expected_theta"),
    [(None, 1000000.0), ({"rope_type": "default"}, 10000.0)],
)
def test_eagle_rope_diagnostics_use_qwen3_defaults(
    tmp_path,
    caplog,
    monkeypatch,
    rope_parameters,
    expected_theta,
    speculators_model_type,
):
    """Report Qwen3's effective base without rewriting normalized parameters."""
    layer_config = {
        "model_type": "qwen3",
        "vocab_size": 32000,
        "max_position_embeddings": 128,
    }
    if rope_parameters is not None:
        layer_config["rope_parameters"] = rope_parameters
    raw_config = {
        "speculators_model_type": speculators_model_type,
        "transformer_layer_config": layer_config,
    }
    if speculators_model_type == "peagle":
        raw_config["mask_token_id"] = 0
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    draft_hf = SpeculatorsConfig.from_pretrained(tmp_path)
    target_hf = Qwen3Config(rope_theta=1000000.0, vocab_size=32000)
    original_draft = deepcopy(draft_hf.to_dict())
    original_target = deepcopy(target_hf.to_dict())
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="vllm.config.speculative"):
        config = _make_rope_speculative_config(
            tmp_path,
            draft_hf,
            target_hf,
            "eagle3",
            parallel_drafting=speculators_model_type == "peagle",
        )

    diagnostic = next(
        record for record in caplog.records if "draft model RoPE" in record.getMessage()
    )
    assert diagnostic.args[0]["rope_theta"] == expected_theta
    assert diagnostic.args[1]["rope_theta"] == 1000000.0
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert config.draft_model_config.hf_config.to_dict() == original_draft
    assert target_hf.to_dict() == original_target


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "explicit_rope",
    [
        {"rope_theta": 10000.0},
        {"rope_scaling": {"rope_theta": 10000.0, "rope_type": "default"}},
        {"rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}},
    ],
)
def test_eagle_rope_diagnostics_recognize_speculators_rope(
    tmp_path, caplog, monkeypatch, explicit_rope
):
    """Explicit Speculators training settings must not trigger a missing-key warning."""
    raw_config = {
        "speculators_model_type": "eagle3",
        "transformer_layer_config": {
            "model_type": "llama",
            "vocab_size": 32000,
            "max_position_embeddings": 128,
            **explicit_rope,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    draft_hf = SpeculatorsConfig.from_pretrained(tmp_path)
    patch_rope_parameters(draft_hf)
    original_draft = deepcopy(draft_hf.to_dict())
    target_hf = LlamaConfig(rope_theta=500000.0)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="vllm.config.speculative"):
        config = _make_rope_speculative_config(tmp_path, draft_hf, target_hf, "eagle3")

    assert any("draft model RoPE" in record.getMessage() for record in caplog.records)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    assert config.draft_model_config.hf_config.to_dict() == original_draft


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
