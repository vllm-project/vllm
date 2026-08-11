# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ModelArchitectureConfig and its integration with ModelConfig."""

import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from transformers import PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.config.model_arch import ModelArchitectureConfig
from vllm.transformers_utils.configs.gemma4 import gemma4_layer_config
from vllm.transformers_utils.model_arch_config_convertor import (
    Gemma4ModelArchConfigConvertor,
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


@pytest.mark.parametrize(
    "attribute",
    [
        "num_experts_per_tok",
        "num_experts_per_token",
        "top_k_experts",
        "moe_topk",
        "moe_top_k",
    ],
)
def test_num_experts_per_tok_aliases(attribute: str):
    hf_config = PretrainedConfig(**{attribute: 4})
    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)
    model_config = cast(
        ModelConfig,
        SimpleNamespace(
            model_arch_config=SimpleNamespace(
                num_experts_per_token=convertor.get_num_experts_per_token()
            )
        ),
    )

    assert ModelConfig.get_num_experts_per_tok(model_config) == 4


def test_num_experts_per_tok_none_is_normalized():
    hf_config = PretrainedConfig(top_k_experts=None)
    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.get_num_experts_per_token() == 0


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


def _layer(**overrides) -> ModelArchitectureConfig:
    fields = dict(
        architectures=["X"],
        model_type="x",
        text_model_type=None,
        hidden_size=64,
        total_num_hidden_layers=3,
        total_num_attention_heads=8,
        head_size=16,
        vocab_size=32,
        total_num_kv_heads=4,
        num_experts=0,
        num_experts_per_token=0,
        quantization_config=None,
        is_deepseek_mla=False,
        is_mm_prefix_lm=False,
        rswa_window=None,
        derived_max_model_len_and_key=(1.0, None),
    )
    return ModelArchitectureConfig(**(fields | overrides))


def test_from_layers_collapses_to_the_largest_layer():
    """The whole-model config must size buffers for every layer."""
    arch = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32, total_num_kv_heads=8)]
    )

    assert (arch.head_size, arch.total_num_kv_heads) == (32, 8)
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]
    assert [arch[i].total_num_kv_heads for i in range(3)] == [4, 4, 8]
    # A layer view is itself homogeneous, so reading from it cannot recurse.
    assert arch[0].per_layer_overrides is None


def test_layer_view_follows_later_edits_to_the_whole_model_config():
    """`model_arch_config` is edited after construction in a few places."""
    arch = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32)]
    )

    arch.is_mm_prefix_lm = True

    assert all(arch[i].is_mm_prefix_lm for i in range(3))
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]


def test_uniform_layers_stay_homogeneous():
    """A checkpoint can vary attributes vLLM never reads."""
    arch = ModelArchitectureConfig.from_layers([_layer()] * 3)

    assert arch.per_layer_overrides is None
    assert arch[1] is arch


def test_getitem_rejects_negative_indices():
    """Wrapping would only misbehave on heterogeneous models, so reject both."""
    homogeneous = _layer()
    heterogeneous = ModelArchitectureConfig.from_layers(
        [_layer(), _layer(), _layer(head_size=32)]
    )

    for arch in (homogeneous, heterogeneous):
        with pytest.raises(IndexError):
            arch[-1]


@pytest.mark.parametrize(
    "varying",
    [
        # `bool` is an `int`; collapsing this one with `max` would make `use_mla`
        # true model wide and silently discard every per-layer KV head count.
        {"is_deepseek_mla": True},
        {"quantization_config": {"quant_method": "fp8"}},
        {"rswa_window": 512},
    ],
)
def test_from_layers_rejects_fields_with_no_whole_model_value(varying: dict):
    with pytest.raises(ValueError, match="varies across layers"):
        ModelArchitectureConfig.from_layers([_layer(), _layer(), _layer(**varying)])


def test_from_layers_rejects_a_layer_count_mismatch():
    """Draft configs can inherit a target's per-layer spec."""
    with pytest.raises(ValueError, match="per-layer configs"):
        ModelArchitectureConfig.from_layers([_layer(), _layer(head_size=32)])


def _gemma4_text_config(**overrides) -> Gemma4TextConfig:
    """A six layer Gemma4 whose last layer is wider than the rest."""
    fields = dict(
        num_hidden_layers=6,
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=32,
        # Transformers forces the last layer to full attention.
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    return Gemma4TextConfig(**(fields | overrides))


def test_gemma4_head_dims_vary_by_layer_type():
    """Gemma4's full attention layers are wider than its sliding ones.

    Transformers >= 5.15.0 says so in the config; this exercises the convertor
    building the same per-layer view from the flat attributes used before that.
    """
    text_config = _gemma4_text_config(
        num_global_key_value_heads=8, attention_k_eq_v=True
    )

    arch = Gemma4ModelArchConfigConvertor(text_config, text_config).convert()

    assert (arch.head_size, arch.total_num_kv_heads) == (32, 8)
    assert [arch[i].head_size for i in range(6)] == [16] * 5 + [32]
    assert [arch[i].total_num_kv_heads for i in range(6)] == [4] * 5 + [8]
    # The model files resolve each layer through the same helper, so the KV cache
    # vLLM allocates and the projections the model builds cannot disagree.
    assert [gemma4_layer_config(text_config, i).head_dim for i in range(6)] == [
        arch[i].head_size for i in range(6)
    ]


def test_gemma4_without_global_kv_heads():
    """`num_global_key_value_heads` defaults to `None`, not to a head count."""
    text_config = _gemma4_text_config()

    arch = Gemma4ModelArchConfigConvertor(text_config, text_config).convert()

    assert arch.total_num_kv_heads == 4
    assert [arch[i].head_size for i in range(6)] == [16] * 5 + [32]


def test_gemma4_layer_count_comes_from_num_hidden_layers():
    """`dummy_hf_overrides` shrinks the stack but leaves `layer_types` long."""
    text_config = _gemma4_text_config()
    text_config.num_hidden_layers = 3

    arch = Gemma4ModelArchConfigConvertor(text_config, text_config).convert()

    assert arch.total_num_hidden_layers == 3
    # Only sliding layers survive the truncation, so nothing varies.
    assert arch.per_layer_overrides is None


def test_gemma4_uniform_head_dims_are_homogeneous():
    text_config = _gemma4_text_config(global_head_dim=16)

    arch = Gemma4ModelArchConfigConvertor(text_config, text_config).convert()

    assert arch.per_layer_overrides is None
    assert arch[3] is arch


class _HeterogeneousConfig(PretrainedConfig):
    """A heterogeneous config with no convertor of its own.

    Mirrors the parts vLLM uses: per-layer configs are shallow copies with the
    varying attributes applied and heterogeneity stripped, so they do not recurse."""

    is_heterogeneous = True

    def __init__(self, per_layer: dict[str, list], **kwargs):
        # Set before `super().__init__` because it validates `per_layer_config`
        self._per_layer = per_layer
        super().__init__(**kwargs)

    @property
    def per_layer_config(self) -> list[PretrainedConfig]:
        layers = []
        for i in range(self.num_hidden_layers):
            layer = copy(self)
            layer.is_heterogeneous = False
            for name, values in self._per_layer.items():
                setattr(layer, name, values[i])
            layers.append(layer)
        return layers


def test_transformers_heterogeneous_config_is_resolved_per_layer():
    hf_config = _HeterogeneousConfig(
        per_layer={"head_dim": [16, 16, 32], "num_key_value_heads": [4, 4, 8]},
        num_hidden_layers=3,
        hidden_size=64,
        num_attention_heads=8,
    )

    arch = ModelArchConfigConvertorBase(hf_config, hf_config).convert()

    assert (arch.head_size, arch.total_num_kv_heads) == (32, 8)
    assert [arch[i].head_size for i in range(3)] == [16, 16, 32]
    assert [arch[i].total_num_kv_heads for i in range(3)] == [4, 4, 8]


def test_heterogeneous_config_varying_nothing_vllm_reads():
    """Transformers prunes nothing for us: the diff has to notice."""
    hf_config = _HeterogeneousConfig(
        per_layer={"bos_token_id": [1, 2, 3]},
        num_hidden_layers=3,
        hidden_size=64,
        num_attention_heads=8,
        head_dim=16,
    )

    arch = ModelArchConfigConvertorBase(hf_config, hf_config).convert()

    assert arch.per_layer_overrides is None
    assert arch[2] is arch


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
