# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer/module policy unit tests (ported from the experimental tree) plus
attach-count reproduction against module-name fixtures of real checkpoints."""

import json
from pathlib import Path

import pytest
import torch.nn as nn

from vllm.model_executor.dual_precision.policy_layers import (
    format_layer_indices,
    is_quantized_shadow_layer,
    plan_shadow_attachment,
    resolve_bf16_layer_indices,
    should_attach_int4_shadow,
    transformer_layer_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "weight_name",
    ["qweight", "weight_packed", "w13_weight_packed", "w2_weight_packed"],
)
def test_recognize_quantized_shadow_weight_formats(weight_name: str):
    layer = nn.Module()
    setattr(layer, weight_name, object())

    assert is_quantized_shadow_layer(layer)


@pytest.mark.parametrize(
    "weight_name",
    ["weight_global_scale", "weight_scale_2", "w13_weight_global_scale"],
)
def test_nvfp4_scale_names_mark_a_quantized_shadow(weight_name):
    layer = nn.Module()
    # NVFP4 packs into the plain ``weight`` name, so only a scale can mark it.
    layer.weight = object()
    setattr(layer, weight_name, object())

    assert is_quantized_shadow_layer(layer)


def test_plain_weight_is_not_a_quantized_shadow():
    layer = nn.Module()
    layer.weight = object()

    assert not is_quantized_shadow_layer(layer)


def test_default_bf16_layer_policy_on_64_layers():
    indices = resolve_bf16_layer_indices("first:3,last:3", 64)

    assert indices == frozenset({0, 1, 2, 61, 62, 63})
    assert format_layer_indices(indices) == "0-2,61-63"


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("none", frozenset()),
        ("0,4-6,63", frozenset({0, 4, 5, 6, 63})),
        ("first:0,last:2", frozenset({62, 63})),
        ("first:2,1-3", frozenset({0, 1, 2, 3})),
    ],
)
def test_resolve_bf16_layer_policy(policy: str, expected: frozenset[int]):
    assert resolve_bf16_layer_indices(policy, 64) == expected


@pytest.mark.parametrize(
    "policy",
    ["", "all", "first:-1", "last:65", "7-4", "64", "1,,2"],
)
def test_reject_invalid_bf16_layer_policy(policy: str):
    with pytest.raises(ValueError):
        resolve_bf16_layer_indices(policy, 64)


def test_extract_transformer_layer_index():
    assert (
        transformer_layer_index("language_model.model.layers.63.mlp.gate_up_proj") == 63
    )
    assert transformer_layer_index("language_model.lm_head") is None


def test_default_policy_only_attaches_middle_transformer_blocks():
    bf16_layers = resolve_bf16_layer_indices("first:3,last:3", 64)

    assert not should_attach_int4_shadow(
        "language_model.model.layers.2.mlp.gate_up_proj", bf16_layers
    )
    assert should_attach_int4_shadow(
        "language_model.model.layers.3.mlp.gate_up_proj", bf16_layers
    )
    assert should_attach_int4_shadow(
        "language_model.model.layers.60.mlp.down_proj", bf16_layers
    )
    assert not should_attach_int4_shadow(
        "language_model.model.layers.61.mlp.down_proj", bf16_layers
    )
    assert not should_attach_int4_shadow("language_model.lm_head", bf16_layers)


def test_mlp_only_policy_and_unknown_policy():
    none = frozenset()
    assert should_attach_int4_shadow("model.layers.4.mlp.down_proj", none, "mlp_only")
    assert not should_attach_int4_shadow(
        "model.layers.4.self_attn.qkv_proj", none, "mlp_only"
    )
    with pytest.raises(ValueError, match="mlp_only"):
        should_attach_int4_shadow("model.layers.4.mlp.down_proj", none, "attention")


def _load_fixture(name: str) -> tuple[list[str], set[str]]:
    path = FIXTURES / f"{name}_modules.json"
    if not path.exists():
        pytest.skip(f"fixture {path} missing")
    data = json.loads(path.read_text())
    return data["bf16_linear_modules"], set(data["int4_quantized_modules"])


@pytest.mark.parametrize(
    ("fixture", "num_layers", "bf16_layers", "module_policy", "expected"),
    [
        # Qwen3.5-9B BF16 + Intel AutoRound INT4 shadow, every headline run:
        # Archived log wording: "Loaded 286 GPTQ shadow linear layers;
        # attached 152 ... left 134". The same counts hold for the NVFP4
        # shadow, so the live line no longer says GPTQ.
        ("qwen3_5_9b_autoround", 32, "none", "all", (152, 0, 134)),
        # Nemotron-Nano-9B-v2 + RedHatAI w4a16 (stage0 weight audit):
        # "Loaded 139 ...; attached 112 ... left 27" (27 = mamba conv1d).
        ("nemotron_nano_9b_v2_w4a16", 56, "none", "all", (112, 0, 27)),
        # Gemma4 E2B QAT, MLP-only (tail_w4_supplement_seed42 log):
        # "attached 70 ... kept 141 quantized layers in BF16 by policy".
        ("gemma4_e2b_qat_w4a16", 35, "none", "mlp_only", (70, 141, 2)),
        # Default first:3,last:3 on Qwen3.5-9B (archived line: "attached 123
        # ... kept 29"): blocks 0-2 and 29-30 are GatedDeltaNet (5 linears),
        # block 31 is full attention (4 linears).
        ("qwen3_5_9b_autoround", 32, "first:3,last:3", "all", (123, 29, 134)),
    ],
)
def test_attach_counts_match_recorded_runs(
    fixture: str,
    num_layers: int,
    bf16_layers: str,
    module_policy: str,
    expected: tuple[int, int, int],
):
    bf16_names, quantized = _load_fixture(fixture)
    indices = resolve_bf16_layer_indices(bf16_layers, num_layers)

    attached, policy_bf16, fallback = plan_shadow_attachment(
        bf16_names, quantized, indices, module_policy
    )

    assert (len(attached), len(policy_bf16), len(fallback)) == expected
    assert len(attached) + len(policy_bf16) + len(fallback) == len(bf16_names)
    assert set(attached).isdisjoint(fallback)
    for name in attached:
        assert transformer_layer_index(name) not in indices
