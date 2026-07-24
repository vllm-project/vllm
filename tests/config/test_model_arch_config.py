# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelArchitectureConfig and its integration with ModelConfig."""

import json
from pathlib import Path

import pytest
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

BASE_TRUST_REMOTE_CODE_MODELS = {
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "XiaomiMiMo/MiMo-7B-RL",
    "stepfun-ai/Step-3.5-Flash",
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
        model_config.hf_config,
        model_config.model,
        revision=model_config.revision,
        config_format="hf",
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


def test_head_size_falls_back_when_head_dim_is_zero():
    """Regression test for configs that materialize missing head_dim as 0."""
    hf_config = PretrainedConfig(
        model_type="deepseek_vl_v2",
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=10,
        head_dim=0,
        kv_lora_rank=None,
    )

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_head_size() == 128


def test_legacy_modelopt_config_without_producer_is_normalized():
    quantization_config = {
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": [],
            "modelopt_quant_config": {"quant_cfg": {}},
        }
    }
    hf_config = PretrainedConfig(quantization_config=quantization_config)

    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_quantization_config()["quant_method"] == "modelopt_fp4"


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


def test_gemma4_head_size_heterogeneous_config():
    """Gemma 4 declares dual head dimensions. `get_head_size` must return the
    largest so attention backends allocate buffers large enough for every
    layer — both for legacy configs (`global_head_dim`) and for heterogeneous
    configs that declare `head_dim` per layer via `per_layer_config`."""
    from transformers import PreTrainedConfig

    from vllm.transformers_utils.model_arch_config_convertor import (
        Gemma4ModelArchConfigConvertor,
    )

    legacy = PretrainedConfig(head_dim=256, global_head_dim=512)
    assert Gemma4ModelArchConfigConvertor(legacy, legacy).get_head_size() == 512

    pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )
    heterogeneous = PreTrainedConfig(
        num_hidden_layers=2, num_attention_heads=8, head_dim=256
    )
    heterogeneous.per_layer_config = {1: {"head_dim": 512}}
    convertor = Gemma4ModelArchConfigConvertor(heterogeneous, heterogeneous)
    assert convertor.get_head_size() == 512


def test_get_model_arch_config_layer_idx():
    """`get_model_arch_config(layer_idx=...)` must resolve per-layer values
    for heterogeneous configs, and must NOT crash for homogeneous ones.

    `per_layer_config` exists (as `None`) on every config via the
    heterogeneity mixin, even when no heterogeneity is declared, so gating
    on `hasattr` instead of the value would subscript `None` for any
    ordinary model as soon as a caller passes `layer_idx`.
    """
    from transformers import PretrainedConfig

    homogeneous = ModelConfig.__new__(ModelConfig)
    homogeneous_hf_config = PretrainedConfig(
        model_type="qwen3", num_attention_heads=8, head_dim=128
    )
    homogeneous.hf_config = homogeneous_hf_config
    homogeneous.hf_text_config = homogeneous_hf_config
    homogeneous.model_arch_config = homogeneous.get_model_arch_config()

    assert homogeneous.get_model_arch_config().head_size == 128
    assert homogeneous.get_model_arch_config(layer_idx=0).head_size == 128

    pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )
    from transformers import PreTrainedConfig

    heterogeneous = ModelConfig.__new__(ModelConfig)
    heterogeneous_hf_config = PreTrainedConfig(
        model_type="gemma4_unified",
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=256,
    )
    heterogeneous_hf_config.per_layer_config = {
        1: {"head_dim": 512, "num_key_value_heads": 1}
    }
    heterogeneous.hf_config = heterogeneous_hf_config
    heterogeneous.hf_text_config = heterogeneous_hf_config
    heterogeneous.model_arch_config = heterogeneous.get_model_arch_config()

    sliding = heterogeneous.get_model_arch_config(layer_idx=0)
    full = heterogeneous.get_model_arch_config(layer_idx=1)
    assert sliding.head_size == 256
    assert sliding.total_num_kv_heads == 8
    assert full.head_size == 512
    assert full.total_num_kv_heads == 1

    # TP-division arithmetic for get_num_kv_heads is covered by
    # test_heterogeneous_attention in test_backend.py; this only needs to
    # confirm layer_idx routing, so use TP=1 to avoid the multi-GPU
    # validation ParallelConfig(tensor_parallel_size>1) requires.
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    assert heterogeneous.get_head_size(layer_idx=0) == 256
    assert heterogeneous.get_head_size(layer_idx=1) == 512
    assert heterogeneous.get_num_kv_heads(parallel_config, layer_idx=0) == 8
    assert heterogeneous.get_num_kv_heads(parallel_config, layer_idx=1) == 1
