# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E initialization tests for AnyModel.

Downloads the real HF config (not weights), injects AnyModel overrides,
and initializes with load_format="dummy". Requires GPU and network access.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .utils import dummy_hf_overrides

_NUM_LAYERS = 4


def _dense_block_configs(hf_config: PretrainedConfig) -> list[dict]:
    text_cfg = hf_config.get_text_config()
    isize = getattr(text_cfg, "intermediate_size", 1024)
    kv = getattr(
        text_cfg,
        "num_key_value_heads",
        getattr(text_cfg, "num_attention_heads", 4),
    )
    return [
        {"attention": {"no_op": False}, "ffn": {"no_op": False}},
        {
            "attention": {"no_op": False},
            "ffn": {"no_op": False, "intermediate_size": max(64, isize // 2)},
        },
        {
            "attention": {
                "no_op": False,
                "num_key_value_heads": max(1, kv // 2),
            },
            "ffn": {"no_op": False},
        },
        {
            "attention": {"no_op": True},
            "ffn": {
                "no_op": False,
                "intermediate_size": max(64, isize // 2),
            },
        },
    ]


def _moe_block_configs(
    hf_config: PretrainedConfig,
    *,
    experts_field: str = "num_local_experts",
    expert_size_field: str | None = None,
) -> list[dict]:
    text_cfg = hf_config.get_text_config()
    n_experts = getattr(text_cfg, experts_field, 4)
    exp_isize = (
        getattr(text_cfg, expert_size_field, None) if expert_size_field else None
    )
    reduced = max(2, n_experts // 2)
    moe_override: dict[str, Any] = {"num_local_experts": reduced}
    if exp_isize is not None:
        moe_override["expert_intermediate_dim"] = max(64, exp_isize // 2)
    return [
        {"attention": {"no_op": False}, "ffn": {"no_op": False}},
        {
            "attention": {"no_op": False},
            "ffn": {"no_op": False, "moe": moe_override},
        },
        {"attention": {"no_op": False}, "ffn": {"no_op": False}},
        {"attention": {"no_op": True}, "ffn": {"no_op": False}},
    ]


def _nemotronh_block_configs(hf_config: PretrainedConfig) -> list[dict]:
    kv = getattr(hf_config, "num_key_value_heads", 4)
    return [
        {"attention": {"no_op": False}, "ffn": {"no_op": False}},
        {"attention": {"no_op": False}, "ffn": {"no_op": False}},
        {
            "attention": {
                "no_op": False,
                "num_key_value_heads": max(1, kv // 2),
            },
            "ffn": {"no_op": False},
        },
        {"attention": {"no_op": False}, "ffn": {"no_op": True}},
    ]


@dataclass
class _Case:
    id: str
    base_arch: str
    hf_model: str
    make_block_configs: Any
    trust_remote_code: bool = False
    max_model_len: int = 1024
    attention_backend: str | None = None
    extra_hf_overrides: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.id


_CASES: list[_Case] = [
    # Dense
    _Case(
        "llama",
        "LlamaForCausalLM",
        "meta-llama/Llama-3.2-1B-Instruct",
        _dense_block_configs,
    ),
    _Case(
        "mistral",
        "MistralForCausalLM",
        "mistralai/Mistral-7B-Instruct-v0.1",
        _dense_block_configs,
    ),
    _Case(
        "qwen2", "Qwen2ForCausalLM", "Qwen/Qwen2-0.5B-Instruct", _dense_block_configs
    ),
    _Case("qwen3", "Qwen3ForCausalLM", "Qwen/Qwen3-8B", _dense_block_configs),
    # MoE
    _Case(
        "qwen2moe",
        "Qwen2MoeForCausalLM",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        partial(
            _moe_block_configs,
            experts_field="num_experts",
            expert_size_field="moe_intermediate_size",
        ),
    ),
    _Case(
        "mixtral",
        "MixtralForCausalLM",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        partial(_moe_block_configs, experts_field="num_local_experts"),
    ),
    _Case(
        "gptoss",
        "GptOssForCausalLM",
        "lmsys/gpt-oss-20b-bf16",
        partial(_moe_block_configs, experts_field="num_local_experts"),
        attention_backend="TRITON_ATTN",
    ),
    # Hybrid
    _Case(
        "nemotronh",
        "NemotronHForCausalLM",
        "nvidia/Nemotron-H-8B-Base-8K",
        _nemotronh_block_configs,
        trust_remote_code=True,
        extra_hf_overrides={"hybrid_override_pattern": "*-*-"},
    ),
    # Multimodal
    _Case(
        "qwen3vl",
        "Qwen3VLForConditionalGeneration",
        "Qwen/Qwen3-VL-4B-Instruct",
        _dense_block_configs,
        max_model_len=4096,
    ),
]


def _anymodel_hf_overrides(
    hf_config: PretrainedConfig,
    *,
    case: _Case,
) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=case.base_arch)
    text_cfg = hf_config.get_text_config()
    text_cfg.num_hidden_layers = _NUM_LAYERS
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = case.base_arch
    for key, val in case.extra_hf_overrides.items():
        setattr(text_cfg, key, val)
    text_cfg.block_configs = case.make_block_configs(hf_config)
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update({"num_layers": 1, "num_hidden_layers": 1})
    return hf_config


def _kv_cache_stub(self, vllm_config):
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        [10 * GiB_bytes],
    )
    return 1, 0, generate_scheduler_kv_cache_config(kv_cache_configs)


def _get_model(llm: LLM):
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model


@create_new_process_for_each_test()
def _run_anymodel_e2e(case: _Case):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    attention_config = (
        {"backend": case.attention_backend} if case.attention_backend else None
    )
    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        LLM(
            case.hf_model,
            load_format="dummy",
            hf_overrides=partial(_anymodel_hf_overrides, case=case),
            trust_remote_code=case.trust_remote_code,
            max_model_len=case.max_model_len,
            enforce_eager=False,
            gpu_memory_utilization=0.80,
            attention_config=attention_config,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
def test_anymodel_e2e(case: _Case):
    _run_anymodel_e2e(case)


_PARITY_MODEL = "Qwen/Qwen2-0.5B-Instruct"
_PARITY_BASE_ARCH = "Qwen2ForCausalLM"


def _identity_anymodel_overrides(
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    text_cfg = hf_config.get_text_config()
    n_layers = getattr(text_cfg, "num_hidden_layers", 1)
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = _PARITY_BASE_ARCH
    text_cfg.block_configs = [
        {"attention": {"no_op": False}, "ffn": {"no_op": False}}
        for _ in range(n_layers)
    ]
    return hf_config


@create_new_process_for_each_test()
def _run_anymodel_parity():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sampling = SamplingParams(temperature=0, max_tokens=64)
    prompt = "The capital of France is"

    base_llm = LLM(_PARITY_MODEL, enforce_eager=False, gpu_memory_utilization=0.4)
    base_text = base_llm.generate([prompt], sampling)[0].outputs[0].text
    del base_llm
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()

    anymodel_llm = LLM(
        _PARITY_MODEL,
        hf_overrides=_identity_anymodel_overrides,
        enforce_eager=False,
        gpu_memory_utilization=0.4,
    )
    anymodel_text = anymodel_llm.generate([prompt], sampling)[0].outputs[0].text
    assert base_text == anymodel_text, (
        f"Parity mismatch:\n  base:     {base_text!r}\n  anymodel: {anymodel_text!r}"
    )


def test_anymodel_weight_loading_parity():
    _run_anymodel_parity()


_REDUCTION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
_REDUCTION_ARCH = "LlamaForCausalLM"


def _base_llama_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=_REDUCTION_ARCH)
    hf_config.get_text_config().num_hidden_layers = _NUM_LAYERS
    return hf_config


def _reduced_anymodel_overrides(
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    hf_config = dummy_hf_overrides(hf_config, model_arch=_REDUCTION_ARCH)
    text_cfg = hf_config.get_text_config()
    text_cfg.num_hidden_layers = _NUM_LAYERS
    hf_config.architectures = ["AnyModel"]
    hf_config.base_architecture = _REDUCTION_ARCH
    isize = getattr(text_cfg, "intermediate_size", 1024)
    kv = getattr(
        text_cfg,
        "num_key_value_heads",
        getattr(text_cfg, "num_attention_heads", 4),
    )
    half_i, half_kv = max(64, isize // 2), max(1, kv // 2)
    text_cfg.block_configs = [
        {
            "attention": {"no_op": False},
            "ffn": {"no_op": False, "intermediate_size": half_i},
        },
        {
            "attention": {"no_op": False, "num_key_value_heads": half_kv},
            "ffn": {"no_op": False, "intermediate_size": half_i},
        },
        {
            "attention": {"no_op": True},
            "ffn": {"no_op": False, "intermediate_size": half_i},
        },
        {"attention": {"no_op": True}, "ffn": {"no_op": True}},
    ]
    return hf_config


@create_new_process_for_each_test()
def _run_anymodel_size_reduction():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    common = dict(load_format="dummy", enforce_eager=True, gpu_memory_utilization=0.80)
    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        base_llm = LLM(_REDUCTION_MODEL, hf_overrides=_base_llama_overrides, **common)
    base_params = sum(p.numel() for p in _get_model(base_llm).parameters())
    del base_llm
    cleanup_dist_env_and_memory()

    with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
        any_llm = LLM(
            _REDUCTION_MODEL, hf_overrides=_reduced_anymodel_overrides, **common
        )
    any_params = sum(p.numel() for p in _get_model(any_llm).parameters())
    assert any_params < base_params, (
        f"AnyModel should have fewer params: {any_params:,} >= {base_params:,}"
    )


def test_anymodel_size_reduction():
    _run_anymodel_size_reduction()


_NAS_CONFIG_PATH = Path(__file__).resolve().parents[0] / "fixtures" / "nas_config.json"


@create_new_process_for_each_test()
def _run_puzzletron_nas_config():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm.model_executor.models.anymodel import (
        NoOpAttention,
        NoOpMLP,
        NoOpNorm,
    )

    with open(_NAS_CONFIG_PATH) as f:
        config = json.load(f)

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "config.json").write_text(json.dumps(config))
        with patch.object(V1EngineCore, "_initialize_kv_caches", _kv_cache_stub):
            llm = LLM(
                tmp,
                tokenizer="meta-llama/Llama-3.2-1B-Instruct",
                load_format="dummy",
                enforce_eager=True,
                max_model_len=1024,
                gpu_memory_utilization=0.90,
            )

    model = _get_model(llm)
    layers = model.model.layers
    bcs = config["block_configs"]
    assert len(layers) == len(bcs) == config["num_hidden_layers"]

    hs = config["hidden_size"]
    hd = config["head_dim"]
    nh = config["num_attention_heads"]
    gkv = config["num_key_value_heads"]

    for i, (layer, bc) in enumerate(zip(layers, bcs)):
        a_noop = bc["attention"]["no_op"]
        f_noop = bc["ffn"]["no_op"]

        assert isinstance(layer.self_attn, NoOpAttention) == a_noop, f"L{i} attn"
        assert isinstance(layer.input_layernorm, NoOpNorm) == a_noop, f"L{i} norm"
        assert isinstance(layer.mlp, NoOpMLP) == f_noop, f"L{i} ffn"
        assert isinstance(layer.post_attention_layernorm, NoOpNorm) == f_noop, (
            f"L{i} ffn norm"
        )

        if not a_noop:
            kv = bc["attention"].get("num_key_value_heads") or gkv
            assert layer.self_attn.qkv_proj.weight.shape == (
                (nh + 2 * kv) * hd,
                hs,
            ), f"L{i} qkv"

        if not f_noop:
            isize = bc["ffn"]["intermediate_size"]
            assert layer.mlp.down_proj.weight.shape == (hs, isize), f"L{i} down"
            assert layer.mlp.gate_up_proj.weight.shape == (
                2 * isize,
                hs,
            ), f"L{i} gate_up"


def test_puzzletron_nas_config():
    _run_puzzletron_nas_config()
