# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelArchitectureConfig and its integration with ModelConfig."""

import json
from pathlib import Path

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

BASE_TRUST_REMOTE_CODE_MODELS = {
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "XiaomiMiMo/MiMo-7B-RL",
    # Excluded: Not available online right now
    # "FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1",
    "meituan-longcat/LongCat-Flash-Chat",
}

BASE_MODELS_TO_TEST = [
    "state-spaces/mamba-130m-hf",
    "mistralai/Mamba-Codestral-7B-v0.1",
    # Excluded: terratorch/torchgeo version mismatch in CPU CI environment
    # (NonGeoDataset import error). Tested in model initialization tests.
    # "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    "Zyphra/Zamba2-7B-instruct",
    # FIXME: mosaicml/mpt-7b has been deleted
    # "mosaicml/mpt-7b",
    # FIXME: databricks/dbrx-instruct has been deleted
    # "databricks/dbrx-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "luccafong/deepseek_mtp_main_random",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "tiny-random/qwen3-next-moe",
    "zai-org/GLM-4.5",
    "baidu/ERNIE-4.5-21B-A3B-PT",
    # Models using base convertor
    "lmsys/gpt-oss-20b-bf16",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
] + list(BASE_TRUST_REMOTE_CODE_MODELS)

# (target_model, draft_model, trust_remote_code)
SPECULATIVE_MODELS = [
    ("JackFram/llama-68m", "abhigoyal/vllm-medusa-llama-68m-random", False),
    ("luccafong/deepseek_mtp_main_random", "luccafong/deepseek_mtp_draft_random", True),
    ("eagle618/deepseek-v3-random", "eagle618/eagle-deepseek-v3-random", True),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "yuhuili/EAGLE-LLaMA3-Instruct-8B", True),
    ("meta-llama/Llama-3.1-8B-Instruct", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", True),
]


def _load_groundtruth(filename: str) -> dict:
    """Load groundtruth JSON from the test directory."""
    groundtruth_path = Path(__file__).parent / filename
    with open(groundtruth_path) as f:
        return json.load(f)


def _assert_model_arch_config(
    model_config, expected: dict, check_head_size: bool = True
):
    """Assert model_arch_config matches expected values."""
    model_arch_config = model_config.model_arch_config
    assert model_arch_config.architectures == expected["architectures"]
    assert model_arch_config.model_type == expected["model_type"]
    assert model_arch_config.text_model_type == expected["text_model_type"]
    assert model_arch_config.hidden_size == expected["hidden_size"]
    assert (
        model_arch_config.total_num_hidden_layers == expected["total_num_hidden_layers"]
    )
    assert (
        model_arch_config.total_num_attention_heads
        == expected["total_num_attention_heads"]
    )
    assert model_arch_config.vocab_size == expected["vocab_size"]
    assert model_arch_config.total_num_kv_heads == expected["total_num_kv_heads"]
    assert model_arch_config.num_experts == expected["num_experts"]
    assert model_arch_config.is_deepseek_mla == expected["is_deepseek_mla"]

    torch_dtype = ModelArchConfigConvertorBase.get_torch_dtype(
        model_config.hf_config, model_config.model, revision=model_config.revision
    )
    assert str(torch_dtype) == expected["dtype"]

    if check_head_size:
        assert model_arch_config.head_size == expected["head_size"]


def _assert_model_config_methods(
    model_config, expected: dict, check_head_size: bool = True
):
    """Assert model_config methods return expected values."""
    assert model_config.architectures == expected["architectures"]
    assert model_config.get_vocab_size() == expected["vocab_size"]
    assert model_config.get_hidden_size() == expected["hidden_size"]
    assert model_config.get_total_num_kv_heads() == expected["total_num_kv_heads"]
    assert model_config.get_num_experts() == expected["num_experts"]
    assert (
        model_config.get_total_num_hidden_layers()
        == expected["total_num_hidden_layers"]
    )

    if check_head_size:
        assert model_config.get_head_size() == expected["head_size"]


@pytest.mark.parametrize("model", BASE_MODELS_TO_TEST)
def test_base_model_arch_config(model: str):
    """Test model architecture config for base models."""
    groundtruth = _load_groundtruth("base_model_arch_groundtruth.json")
    expected = groundtruth[model]

    model_config = ModelConfig(
        model, trust_remote_code=model in BASE_TRUST_REMOTE_CODE_MODELS
    )

    _assert_model_arch_config(model_config, expected)
    _assert_model_config_methods(model_config, expected)


@pytest.mark.parametrize(
    "target_model,draft_model,trust_remote_code", SPECULATIVE_MODELS
)
def test_draft_model_arch_config(
    target_model: str, draft_model: str, trust_remote_code: bool
):
    """Test model architecture config for draft/speculative models."""
    groundtruth = _load_groundtruth("draft_model_arch_groundtruth.json")
    expected = groundtruth[draft_model]

    target_model_config = ModelConfig(target_model, trust_remote_code=trust_remote_code)
    speculative_config = SpeculativeConfig(
        model=draft_model,
        num_speculative_tokens=1,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
    )
    model_config = speculative_config.draft_model_config

    # For medusa models, head_size may cause division by zero before
    # model_arch_config was introduced, so we conditionally check it
    check_head_size = isinstance(expected["head_size"], int)

    _assert_model_arch_config(model_config, expected, check_head_size=check_head_size)
    _assert_model_config_methods(
        model_config, expected, check_head_size=check_head_size
    )
