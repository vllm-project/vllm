# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LoRA target guards in Jamba model."""

from unittest.mock import MagicMock

import pytest

from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.models.jamba import JambaForCausalLM, JambaModel

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("target", ["experts", "router", "feed_forward.experts"])
def test_jamba_raises_value_error_for_moe_lora_targets(target):
    vllm_config = MagicMock()
    vllm_config.lora_config.target_modules = [target]
    with pytest.raises(
        ValueError, match="LoRA is not currently supported on Jamba MoE expert layers"
    ):
        JambaModel(vllm_config=vllm_config)


def test_jamba_allows_non_moe_lora_targets():
    vllm_config = MagicMock()
    vllm_config.lora_config.target_modules = ["qkv_proj", "o_proj"]
    vllm_config.model_config.hf_config.num_hidden_layers = 0
    vllm_config.model_config.hf_config.vocab_size = 100
    vllm_config.model_config.hf_config.hidden_size = 64
    vllm_config.model_config.hf_config.rms_norm_eps = 1e-5
    model = JambaModel(vllm_config=vllm_config)
    assert model is not None


def test_jamba_allows_lora_when_target_modules_none():
    vllm_config = MagicMock()
    vllm_config.lora_config.target_modules = None
    vllm_config.model_config.hf_config.num_hidden_layers = 0
    vllm_config.model_config.hf_config.vocab_size = 100
    vllm_config.model_config.hf_config.hidden_size = 64
    vllm_config.model_config.hf_config.rms_norm_eps = 1e-5
    model = JambaModel(vllm_config=vllm_config)
    assert model is not None


def test_jamba_supported_lora_modules_excludes_moe():
    supported = JambaForCausalLM.supported_lora_modules
    assert "experts" not in supported
    assert "router" not in supported
    assert "qkv_proj" in supported
    assert "o_proj" in supported
    assert "in_proj" in supported


def test_get_supported_lora_modules_respects_model_declared_list():
    model = MagicMock(spec=JambaForCausalLM)
    model.supported_lora_modules = JambaForCausalLM.supported_lora_modules
    supported = get_supported_lora_modules(model)
    assert supported == JambaForCausalLM.supported_lora_modules
