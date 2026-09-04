# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 MTP speculative decoding config overrides."""

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


def _mtp_config(model_type: str) -> PretrainedConfig:
    """Create a top-level MTP configuration with mtp_num_hidden_layers."""
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArch"],
        mtp_num_hidden_layers=1,
    )


def _multimodal_wrapper_mtp_config(
    model_type: str, mtp_layers: int = 1
) -> PretrainedConfig:
    """Construct a multimodal wrapper config with mtp_num_hidden_layers
    in text_config."""
    text_config = PretrainedConfig(
        model_type="qwen3_5_moe_text" if "moe" in model_type else "qwen3_5_text",
        architectures=["SomeArch"],
        num_attention_heads=32,
        mtp_num_hidden_layers=mtp_layers,
    )
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArchForConditionalGeneration"],
        text_config=text_config,
    )


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants must map to the same MTP architectures.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(
    model_type: str, expected_arch: str
) -> None:
    """Verify that text-only config variants map to the expected MTP architectures."""
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_extracts_n_predict_from_multimodal_wrapper(
    model_type: str, expected_arch: str
) -> None:
    """Verify that multimodal wrapper checkpoints with mtp_num_hidden_layers
    in text_config resolve n_predict and architecture correctly."""
    cfg = SpeculativeConfig.hf_config_override(
        _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    )
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 2


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_top_level_precedence_over_nested_text_config(
    model_type: str, expected_arch: str
) -> None:
    """Verify that an explicit top-level mtp_num_hidden_layers takes precedence
    over a nested text_config value."""
    cfg = _multimodal_wrapper_mtp_config(model_type, mtp_layers=2)
    cfg.mtp_num_hidden_layers = 3
    overridden = SpeculativeConfig.hf_config_override(cfg)
    assert overridden.model_type == "qwen3_5_mtp"
    assert overridden.architectures == [expected_arch]
    assert overridden.n_predict == 3
