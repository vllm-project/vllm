# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 MTP speculative decoding config overrides."""

from pathlib import Path
from typing import Any

import pytest
from transformers import (
    AutoConfig,
    PreTrainedConfig,
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

from vllm.config import ModelConfig, ParallelConfig
from vllm.config.speculative import SpeculativeConfig

_CHECKPOINTS = {
    "qwen3_5": "Qwen/Qwen3.8-27B",
    "qwen3_5_moe": "Qwen/Qwen3.6-35B-A3B",
}


def _mtp_config(model_type: str) -> PreTrainedConfig:
    """Create a top-level MTP configuration with mtp_num_hidden_layers."""
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "architectures": ["SomeArch"],
        "mtp_num_hidden_layers": 1,
    }
    return PreTrainedConfig(**kwargs)


def _multimodal_wrapper_mtp_config(
    model_type: str, mtp_layers: int = 1
) -> PreTrainedConfig:
    """Download a multimodal wrapper config via AutoConfig.from_pretrained
    and configure mtp_num_hidden_layers in text_config.

    Uses real-world Hugging Face Hub checkpoints:
    - Dense (qwen3_5):     Qwen/Qwen3.8-27B
    - MoE   (qwen3_5_moe): Qwen/Qwen3.6-35B-A3B
    """
    repo = _CHECKPOINTS[model_type]
    config: PreTrainedConfig = AutoConfig.from_pretrained(repo)
    text_config = config.get_text_config()
    text_config.mtp_num_hidden_layers = mtp_layers
    return config


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


@pytest.mark.parametrize(
    "model_id,expected_arch",
    [
        ("Qwen/Qwen3.8-27B", "Qwen3_5MTP"),
        ("Qwen/Qwen3.6-35B-A3B", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_downloads_real_hf_hub_configs(
    model_id: str, expected_arch: str
) -> None:
    """Verify that unmodified real-world checkpoints downloaded via
    AutoConfig.from_pretrained resolve n_predict=1 from text_config."""
    hf_config = AutoConfig.from_pretrained(model_id)
    cfg = SpeculativeConfig.hf_config_override(hf_config)
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


@pytest.mark.parametrize("multimodal", [False, True], ids=["text", "multimodal"])
@pytest.mark.parametrize(
    ("mtp_layers", "num_spec_tokens", "index_share", "multi_module"),
    [
        (1, 2, None, False),
        (2, 1, None, False),
        (2, 2, None, True),
        (2, 2, True, False),
    ],
)
def test_mtp_num_hidden_layers_selects_multi_module_mtp(
    tmp_path: Path,
    multimodal: bool,
    mtp_layers: int,
    num_spec_tokens: int,
    index_share: bool | None,
    multi_module: bool,
) -> None:
    """Qwen3.5 declares its MTP modules as `mtp_num_hidden_layers` (nested in
    `text_config` when multimodal), not `num_nextn_predict_layers`. MTP index
    sharing is only supported by the single-module speculator."""
    text_config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        mtp_num_hidden_layers=mtp_layers,
    )
    hf_config: PreTrainedConfig
    if multimodal:
        hf_config = Qwen3_5Config(
            text_config=text_config.to_dict(),
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
    else:
        text_config.architectures = ["Qwen3_5ForCausalLM"]
        hf_config = text_config
    hf_config.save_pretrained(tmp_path)

    speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=num_spec_tokens,
        index_share_for_mtp_iteration=index_share,
        target_model_config=ModelConfig(
            model=str(tmp_path), tokenizer_mode="skip", max_model_len=2048
        ),
        target_parallel_config=ParallelConfig(),
    )

    assert speculative_config.use_multi_module_mtp() is multi_module
    assert speculative_config.num_mtp_modules() == mtp_layers
