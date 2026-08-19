# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU MLA head-dimension fail-fast guard.

The CPU MLA decode kernel (csrc/cpu/mla_decode.cpp) only compiles for the
DeepSeek-V2/V3 cache layout (head_dim=576, v_head_dim=512, block_size=16).
Models whose MLA head dimensions differ (e.g. the Kimi-K3-0.40B tiny variant
with head_dim=160, v_head_dim=64) cannot run on the CPU backend at all, so
``CpuPlatform.check_and_update_config`` must reject them at startup with a
clear error instead of crashing deep inside kernel dispatch.

The VllmConfig is built from a synthetic local config.json (no network access,
no weight download), matching the "local config fixture only" pattern used by
the related CPU MLA work in PR #52045.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms.cpu import CpuPlatform

# A minimal, self-contained config that ModelConfig can parse offline.
_SYNTHETIC_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 100,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
}


@pytest.fixture
def vllm_config(tmp_path) -> VllmConfig:
    """A fully-constructed VllmConfig from a synthetic local config.

    No network access and no remote code: the base model is a plain Llama
    config; the guard is exercised by stubbing hf_text_config's MLA fields.
    """
    (tmp_path / "config.json").write_text(json.dumps(_SYNTHETIC_CONFIG))
    cfg = EngineArgs(model=str(tmp_path)).create_engine_config()
    cfg.device_config.device_type = "cpu"
    # Force MLA mode and the tiny-variant head dimensions.
    # ModelConfig.use_mla derives from hf_text_config.kv_lora_rank only when
    # using_transformers_backend() is True; for a plain Llama config it is
    # False, so the MLA branch would never be entered. The stub below makes
    # the guard run with realistic MLA dimensions.
    cfg.model_config.hf_text_config = SimpleNamespace(
        kv_lora_rank=128, qk_rope_head_dim=32, v_head_dim=64
    )
    return cfg


@pytest.fixture
def mla_use_patch():
    """Make ModelConfig.use_mla resolve from hf_text_config.kv_lora_rank.

    use_mla is a read-only property (vllm/config/model.py) that consults
    hf_text_config.kv_lora_rank only under the transformers backend; the
    synthetic Llama config is not MLA-shaped, so we force the property to
    take the kv_lora_rank path for the duration of the test.
    """
    with patch.object(ModelConfig, "using_transformers_backend", return_value=True):
        yield


def _set_dims(vllm_config: VllmConfig, **kwargs) -> None:
    for k, v in kwargs.items():
        setattr(vllm_config.model_config.hf_text_config, k, v)


def test_cpu_mla_deepseek_dimensions_accepted(vllm_config, mla_use_patch):
    """DeepSeek-V2/V3 layout (576/512) passes the guard and keeps running.

    head_dim = kv_lora_rank(512) + qk_rope_head_dim(64) = 576, which is the
    layout the CPU MLA decode kernel (csrc/cpu/mla_decode.cpp) compiles for.
    """
    _set_dims(vllm_config, kv_lora_rank=512, qk_rope_head_dim=64, v_head_dim=512)

    with patch("vllm.platforms.current_platform.device_type", "cpu"):
        # Must not raise; chunked prefill / prefix caching are forced off.
        CpuPlatform.check_and_update_config(vllm_config)
    assert vllm_config.scheduler_config.enable_chunked_prefill is False
    assert vllm_config.cache_config.enable_prefix_caching is False


def test_cpu_mla_unsupported_dimensions_fail_fast(vllm_config, mla_use_patch):
    """Tiny-variant layout (160/64) is rejected with a clear error."""
    with (
        patch("vllm.platforms.current_platform.device_type", "cpu"),
        pytest.raises(ValueError, match="head_dim=160, v_head_dim=64"),
    ):
        CpuPlatform.check_and_update_config(vllm_config)


def test_cpu_mla_missing_dimensions_fail_fast(vllm_config, mla_use_patch):
    """MLA models missing qk_rope_head_dim are rejected via getattr, not crashed."""
    _set_dims(vllm_config, qk_rope_head_dim=None)

    with (
        patch("vllm.platforms.current_platform.device_type", "cpu"),
        pytest.raises(ValueError, match="head_dim=None"),
    ):
        CpuPlatform.check_and_update_config(vllm_config)
