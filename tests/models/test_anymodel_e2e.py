# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E initialization tests for AnyModel across all 9 supported architectures.

Each test downloads the real HF config (not weights), injects AnyModel-specific
overrides (``architectures``, ``base_architectures``, ``block_configs``), and
initialises the model with ``load_format="dummy"`` (random weights).  This
validates the full AnyModel → base-model → patch pipeline for every registered
architecture without requiring real checkpoints.

``test_generate_anymodel_configs`` produces example ``config.json`` files
(one per architecture) that can be shared with the Puzzletron team as a
reference for the expected checkpoint format.

GPU access is required for the E2E initialization tests.
Network access is required for all tests (to download HF configs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoConfig, PretrainedConfig

from vllm import LLM
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .utils import dummy_hf_overrides

# ---------------------------------------------------------------------------
# Block-config generators
# ---------------------------------------------------------------------------

_NUM_LAYERS = 4


def _dense_block_configs(hf_config: PretrainedConfig) -> list[dict]:
    """Heterogeneous block_configs for dense (non-MoE) architectures.

    Layout (4 layers):
      L0  normal
      L1  reduced intermediate_size
      L2  reduced num_key_value_heads
      L3  attention no-op + reduced FFN
    """
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
            "ffn": {"no_op": False, "intermediate_size": max(64, isize // 2)},
        },
    ]


def _moe_block_configs(
    hf_config: PretrainedConfig,
    *,
    experts_field: str = "num_local_experts",
    expert_size_field: str | None = None,
) -> list[dict]:
    """Heterogeneous block_configs for MoE architectures.

    Layout (4 layers):
      L0  normal
      L1  reduced expert count + expert intermediate size
      L2  normal
      L3  attention no-op
    """
    text_cfg = hf_config.get_text_config()
    n_experts = getattr(text_cfg, experts_field, 4)
    exp_isize = None
    if expert_size_field:
        exp_isize = getattr(text_cfg, expert_size_field, None)

    reduced = max(2, n_experts // 2)
    moe_override: dict[str, Any] = {"num_local_experts": reduced}
    if exp_isize is not None:
        moe_override["expert_intermediate_size"] = max(64, exp_isize // 2)

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
    """Block configs for NemotronH hybrid with pattern '*-*-'.

    L0 (*) attention layer — normal
    L1 (-) MLP layer — normal
    L2 (*) attention layer — reduced KV heads
    L3 (-) MLP layer — ffn no-op
    """
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


# ---------------------------------------------------------------------------
# Architecture test case descriptors
# ---------------------------------------------------------------------------


@dataclass
class _AnyModelE2ECase:
    """Describes one architecture to test end-to-end with AnyModel."""

    id: str
    base_arch: str
    hf_model: str
    make_block_configs: Any  # callable(hf_config) -> list[dict]
    trust_remote_code: bool = False
    max_model_len: int = 1024
    attention_backend: str | None = None
    extra_hf_overrides: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return self.id


_CASES: list[_AnyModelE2ECase] = [
    # ---- Dense ----
    _AnyModelE2ECase(
        id="llama",
        base_arch="LlamaForCausalLM",
        hf_model="meta-llama/Llama-3.2-1B-Instruct",
        make_block_configs=_dense_block_configs,
    ),
    _AnyModelE2ECase(
        id="mistral",
        base_arch="MistralForCausalLM",
        hf_model="mistralai/Mistral-7B-Instruct-v0.1",
        make_block_configs=_dense_block_configs,
    ),
    _AnyModelE2ECase(
        id="qwen2",
        base_arch="Qwen2ForCausalLM",
        hf_model="Qwen/Qwen2-0.5B-Instruct",
        make_block_configs=_dense_block_configs,
    ),
    _AnyModelE2ECase(
        id="qwen3",
        base_arch="Qwen3ForCausalLM",
        hf_model="Qwen/Qwen3-8B",
        make_block_configs=_dense_block_configs,
    ),
    # ---- MoE ----
    _AnyModelE2ECase(
        id="qwen2moe",
        base_arch="Qwen2MoeForCausalLM",
        hf_model="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        make_block_configs=partial(
            _moe_block_configs,
            experts_field="num_experts",
            expert_size_field="moe_intermediate_size",
        ),
    ),
    _AnyModelE2ECase(
        id="mixtral",
        base_arch="MixtralForCausalLM",
        hf_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        make_block_configs=partial(
            _moe_block_configs,
            experts_field="num_local_experts",
        ),
    ),
    _AnyModelE2ECase(
        id="gptoss",
        base_arch="GptOssForCausalLM",
        hf_model="lmsys/gpt-oss-20b-bf16",
        make_block_configs=partial(
            _moe_block_configs,
            experts_field="num_local_experts",
        ),
        attention_backend="TRITON_ATTN",
    ),
    # ---- Hybrid ----
    _AnyModelE2ECase(
        id="nemotronh",
        base_arch="NemotronHForCausalLM",
        hf_model="nvidia/Nemotron-H-8B-Base-8K",
        make_block_configs=_nemotronh_block_configs,
        trust_remote_code=True,
        extra_hf_overrides={"hybrid_override_pattern": "*-*-"},
    ),
    # ---- Multimodal ----
    _AnyModelE2ECase(
        id="qwen3vl",
        base_arch="Qwen3VLForConditionalGeneration",
        hf_model="Qwen/Qwen3-VL-4B-Instruct",
        make_block_configs=_dense_block_configs,
        max_model_len=4096,
    ),
]


# ---------------------------------------------------------------------------
# hf_overrides factory
# ---------------------------------------------------------------------------


def _anymodel_hf_overrides(
    hf_config: PretrainedConfig,
    *,
    case: _AnyModelE2ECase,
) -> PretrainedConfig:
    """Apply AnyModel overrides on top of the standard dummy shrinking."""
    hf_config = dummy_hf_overrides(
        hf_config,
        model_arch=case.base_arch,
    )

    text_cfg = hf_config.get_text_config()
    text_cfg.num_hidden_layers = _NUM_LAYERS

    hf_config.architectures = ["AnyModel"]
    hf_config.base_architectures = [case.base_arch]

    for key, val in case.extra_hf_overrides.items():
        setattr(text_cfg, key, val)

    text_cfg.block_configs = case.make_block_configs(hf_config)

    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update(
            {
                "num_layers": 1,
                "num_hidden_layers": 1,
            }
        )

    return hf_config


# ---------------------------------------------------------------------------
# KV-cache stub (avoids running forward pass)
# ---------------------------------------------------------------------------


def _initialize_kv_caches_stub(self, vllm_config):
    kv_cache_specs = self.model_executor.get_kv_cache_specs()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        [10 * GiB_bytes],
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    return 1, 0, scheduler_kv_cache_config


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


@create_new_process_for_each_test()
def _run_anymodel_e2e(case: _AnyModelE2ECase):
    hf_overrides_fn = partial(_anymodel_hf_overrides, case=case)

    attention_config = (
        {"backend": case.attention_backend} if case.attention_backend else None
    )

    with patch.object(
        V1EngineCore, "_initialize_kv_caches", _initialize_kv_caches_stub
    ):
        LLM(
            case.hf_model,
            load_format="dummy",
            hf_overrides=hf_overrides_fn,
            trust_remote_code=case.trust_remote_code,
            max_model_len=case.max_model_len,
            enforce_eager=True,
            gpu_memory_utilization=0.80,
            attention_config=attention_config,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
def test_anymodel_e2e(case: _AnyModelE2ECase):
    """Verify that AnyModel initialises correctly for each architecture."""
    _run_anymodel_e2e(case)


# ---------------------------------------------------------------------------
# Example config.json generation (for Puzzletron reference)
# ---------------------------------------------------------------------------


def _full_dense_block_configs(text_cfg: PretrainedConfig) -> list[dict]:
    """Realistic heterogeneous block_configs for a dense model.

    Uses the real model dimensions.  Roughly:
      - first 60 %  normal
      - next  20 %  reduced intermediate_size (half)
      - next  10 %  reduced num_key_value_heads (half)
      - last  10 %  attention no-op + reduced FFN
    """
    n = getattr(text_cfg, "num_hidden_layers", 32)
    isize = getattr(text_cfg, "intermediate_size", 14336)
    kv = getattr(
        text_cfg,
        "num_key_value_heads",
        getattr(text_cfg, "num_attention_heads", 32),
    )
    half_isize = max(64, isize // 2)
    half_kv = max(1, kv // 2)

    cut1 = int(n * 0.6)
    cut2 = int(n * 0.8)
    cut3 = int(n * 0.9)

    cfgs: list[dict] = []
    for i in range(n):
        if i < cut1:
            cfgs.append({"attention": {"no_op": False}, "ffn": {"no_op": False}})
        elif i < cut2:
            cfgs.append(
                {
                    "attention": {"no_op": False},
                    "ffn": {"no_op": False, "intermediate_size": half_isize},
                }
            )
        elif i < cut3:
            cfgs.append(
                {
                    "attention": {
                        "no_op": False,
                        "num_key_value_heads": half_kv,
                    },
                    "ffn": {"no_op": False},
                }
            )
        else:
            cfgs.append(
                {
                    "attention": {"no_op": True},
                    "ffn": {"no_op": False, "intermediate_size": half_isize},
                }
            )
    return cfgs


def _full_moe_block_configs(
    text_cfg: PretrainedConfig,
    *,
    experts_field: str = "num_local_experts",
    expert_size_field: str | None = None,
) -> list[dict]:
    """Realistic heterogeneous block_configs for an MoE model."""
    n = getattr(text_cfg, "num_hidden_layers", 32)
    n_experts = getattr(text_cfg, experts_field, 8)
    exp_isize = (
        getattr(text_cfg, expert_size_field, None) if expert_size_field else None
    )

    reduced_experts = max(2, n_experts // 2)
    moe_override: dict[str, Any] = {"num_local_experts": reduced_experts}
    if exp_isize is not None:
        moe_override["expert_intermediate_size"] = max(64, exp_isize // 2)

    cut1 = int(n * 0.6)
    cut2 = int(n * 0.8)

    cfgs: list[dict] = []
    for i in range(n):
        if i < cut1:
            cfgs.append({"attention": {"no_op": False}, "ffn": {"no_op": False}})
        elif i < cut2:
            cfgs.append(
                {
                    "attention": {"no_op": False},
                    "ffn": {"no_op": False, "moe": moe_override},
                }
            )
        else:
            cfgs.append({"attention": {"no_op": True}, "ffn": {"no_op": False}})
    return cfgs


def _full_nemotronh_block_configs(text_cfg: PretrainedConfig) -> list[dict]:
    """Realistic block_configs for NemotronH, matching pattern."""
    pattern = getattr(text_cfg, "hybrid_override_pattern", "*-")
    n = len(pattern)
    kv = getattr(text_cfg, "num_key_value_heads", 8)
    half_kv = max(1, kv // 2)
    cut = int(n * 0.75)

    cfgs: list[dict] = []
    for i in range(n):
        if i < cut:
            cfgs.append({"attention": {"no_op": False}, "ffn": {"no_op": False}})
        elif pattern[i] == "*":
            cfgs.append(
                {
                    "attention": {
                        "no_op": False,
                        "num_key_value_heads": half_kv,
                    },
                    "ffn": {"no_op": False},
                }
            )
        else:
            cfgs.append({"attention": {"no_op": False}, "ffn": {"no_op": True}})
    return cfgs


_CONFIG_GENERATORS: dict[str, Any] = {
    "llama": _full_dense_block_configs,
    "mistral": _full_dense_block_configs,
    "qwen2": _full_dense_block_configs,
    "qwen3": _full_dense_block_configs,
    "qwen2moe": partial(
        _full_moe_block_configs,
        experts_field="num_experts",
        expert_size_field="moe_intermediate_size",
    ),
    "mixtral": partial(
        _full_moe_block_configs,
        experts_field="num_local_experts",
    ),
    "gptoss": partial(
        _full_moe_block_configs,
        experts_field="num_local_experts",
    ),
    "nemotronh": _full_nemotronh_block_configs,
    "qwen3vl": _full_dense_block_configs,
}


def _generate_config(case: _AnyModelE2ECase, output_dir: Path) -> Path:
    """Download real HF config, apply AnyModel overrides, write JSON."""
    hf_config = AutoConfig.from_pretrained(
        case.hf_model,
        trust_remote_code=case.trust_remote_code,
    )
    text_cfg = hf_config.get_text_config()

    for key, val in case.extra_hf_overrides.items():
        setattr(text_cfg, key, val)

    gen = _CONFIG_GENERATORS[case.id]
    block_configs = gen(text_cfg)

    hf_config.architectures = ["AnyModel"]
    hf_config.base_architectures = [case.base_arch]
    text_cfg.block_configs = block_configs

    out_dir = output_dir / case.id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.json"

    config_dict = json.loads(hf_config.to_json_string())
    with open(out_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    return out_path


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.id)
def test_generate_anymodel_configs(case: _AnyModelE2ECase, tmp_path: Path):
    """Generate example AnyModel config.json for each architecture.

    The output is written to a temp directory.  Run with ``-s`` to see
    the paths printed, or use ``--basetemp=./generated_test_assets``
    to collect them in a known location::

        pytest tests/models/test_anymodel_e2e.py -k generate -s \\
               --basetemp=./generated_test_assets
    """
    config_path = _generate_config(case, tmp_path)
    assert config_path.exists()

    with open(config_path) as f:
        data = json.load(f)

    assert data["architectures"] == ["AnyModel"]
    assert data["base_architectures"] == [case.base_arch]

    text_data = data.get("text_config", data)
    assert "block_configs" in text_data or "block_configs" in data
    block_configs = text_data.get("block_configs", data.get("block_configs"))
    expected_layers = text_data.get("num_hidden_layers", data.get("num_hidden_layers"))
    assert len(block_configs) == expected_layers

    print(f"\n  [{case.id}] config written to: {config_path}")
