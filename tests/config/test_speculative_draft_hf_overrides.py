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
from transformers import PretrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig


def _make_hf_config(**kwargs) -> PretrainedConfig:
    """Create a mock PretrainedConfig with optional overrides."""
    defaults = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return PretrainedConfig(**defaults)


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
    """None overrides fall back directly to the architecture mapping."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
def test_callable_overrides_reach_the_draft_config():
    """A callable override (config-to-config transform) composes with the
    architecture-mapping override and is applied to the draft config."""

    def shrink(hf_config: PretrainedConfig) -> PretrainedConfig:
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

    def record(hf_config: PretrainedConfig) -> PretrainedConfig:
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
    """Verify Inkling MTP overrides expose all draft depths without clamping."""
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


def _module_level_shrink(hf_config: PretrainedConfig) -> PretrainedConfig:
    """Helper transform to shrink layers for pickling tests."""
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
    """Helper to construct an MTP SpeculativeConfig for index share testing."""
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
    """Verify that index_share_for_mtp_iteration can be overridden on draft
    hf_config."""
    speculative_config = _make_mtp_speculative_config(override, checkpoint_value)
    assert (
        speculative_config.draft_model_config.hf_config.index_share_for_mtp_iteration
        is expected
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("model_param", [None, "draft_model"])
def test_mtp_num_speculative_tokens_default_from_n_predict(model_param: str | None):
    """Verify that when num_speculative_tokens is not provided, SpeculativeConfig
    derives draft model from target if omitted, and defaults num_speculative_tokens
    from the draft model config's n_predict without raising TypeError."""
    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=1,
    )
    draft_model_config = MagicMock(
        model=model_param or "target_model_weights",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target_model",
        model_weights="target_model_weights",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        spec_config = SpeculativeConfig(
            model=model_param,
            method="mtp",
            num_speculative_tokens=None,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )

    expected_model = model_param or "target_model_weights"
    assert spec_config.model == expected_model
    assert spec_config.num_speculative_tokens == 1
    assert spec_config.method == "mtp"


@pytest.mark.cpu_test
def test_mtp_num_speculative_tokens_warning_when_defaulted_above_one(caplog):
    """Verify that when num_speculative_tokens is omitted but defaults to
    n_predict > 1 for MTP, the multiple-forward warning is emitted."""
    import logging

    vllm_logger = logging.getLogger("vllm")
    old_propagate = vllm_logger.propagate
    vllm_logger.propagate = True

    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=2,
    )
    draft_model_config = MagicMock(
        model="target_model_weights",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target_model",
        model_weights="target_model_weights",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    try:
        with (
            patch(
                "vllm.config.speculative.ModelConfig",
                return_value=draft_model_config,
            ),
            caplog.at_level("WARNING"),
        ):
            spec_config = SpeculativeConfig(
                model=None,
                method="mtp",
                num_speculative_tokens=None,
                target_model_config=target_model_config,
                target_parallel_config=ParallelConfig(),
            )
    finally:
        vllm_logger.propagate = old_propagate

    assert spec_config.num_speculative_tokens == 2
    assert any(
        "Enabling num_speculative_tokens > 1 will run multiple times" in record.message
        for record in caplog.records
    )


@pytest.mark.cpu_test
def test_mtp_num_speculative_tokens_missing_n_predict_raises_value_error():
    """Verify that when num_speculative_tokens is omitted and draft config
    does not specify n_predict, a clear ValueError is raised instead of TypeError."""
    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=None,
    )
    draft_model_config = MagicMock(
        model="target_model_weights",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target_model",
        model_weights="target_model_weights",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with (
        patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config),
        pytest.raises(
            ValueError,
            match="num_speculative_tokens` was not provided",
        ),
    ):
        SpeculativeConfig(
            model=None,
            method="mtp",
            num_speculative_tokens=None,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize("model_param", [None, "draft_model"])
@pytest.mark.parametrize("block_size_key", ["block_size", "dspark_block_size"])
def test_dspark_num_speculative_tokens_default_from_block_size(
    model_param: str | None, block_size_key: str
):
    """Verify that when num_speculative_tokens is omitted, DSpark configurations
    providing only block_size or dspark_block_size default num_speculative_tokens
    properly without raising ValueError, across both explicit and omitted models."""
    draft_hf_config = _make_hf_config(
        architectures=["Qwen3OmniDSparkModel"],
        model_type="qwen3",
        **{block_size_key: 5},
    )
    expected_model = model_param or "target_model"
    draft_model_config = MagicMock(
        model=expected_model,
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target_model",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with (
        patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config),
        patch("vllm.config.speculative._validate_qwen3_omni_dspark"),
    ):
        spec_config = SpeculativeConfig(
            model=model_param,
            method="dspark",
            num_speculative_tokens=None,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )

    assert spec_config.model == expected_model
    assert spec_config.num_speculative_tokens == 5
    assert spec_config.method == "dspark"
