# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-VL-MoE weight-loader name handling.

Covers the ModelOpt NVFP4 per-expert path added to `load_weights` in
`vllm/model_executor/models/qwen3_vl_moe.py`:

- `_FUSED_EXPERT_RE` matches BMM-fused expert tensor names but excludes
  un-BMM'd per-expert names (which would crash `transpose(-1, -2)` on
  a 0-D scale tensor).
- `_MODELOPT_PEREXP_RE` parses `experts.<proj>.<N>.<suffix>` so the
  loader can remap to `experts.<N>.<proj>.<suffix>`.
"""

import pytest

from vllm.model_executor.models.qwen3_vl_moe import (
    _FUSED_EXPERT_RE,
    _MODELOPT_PEREXP_RE,
)


@pytest.mark.parametrize(
    "name",
    [
        # BF16 bare Parameter form (HF-hosted unquantized checkpoint)
        "model.layers.0.experts.gate_up_proj",
        "model.layers.0.experts.down_proj",
        # Linear-wrapped fused form
        "model.layers.0.experts.gate_up_proj.weight",
        "model.layers.0.experts.gate_up_proj.bias",
        # NVFP4-fused expert tensor names (Llama4 / GPT-OSS style)
        "model.layers.0.experts.down_proj.weight_packed",
        "model.layers.0.experts.down_proj.weight_scale",
        "model.layers.0.experts.down_proj.weight_scale_2",
    ],
)
def test_fused_expert_re_matches(name):
    assert _FUSED_EXPERT_RE.search(name), (
        f"Expected {name!r} to match the BMM-fused expert pattern"
    )


@pytest.mark.parametrize(
    "name",
    [
        # ModelOpt NVFP4 un-BMM'd form (pre-remap)
        "model.layers.0.experts.down_proj.0.weight_packed",
        "model.layers.0.experts.down_proj.7.weight_scale",
        "model.layers.0.experts.down_proj.127.weight_scale_2",
        # Post-remap (after the loader rewrites `.<proj>.<N>.` → `.<N>.<proj>.`)
        "model.layers.0.experts.0.down_proj.weight_packed",
        "model.layers.0.experts.7.gate_proj.weight_packed",
        "model.layers.0.experts.127.up_proj.weight_scale",
    ],
)
def test_fused_expert_re_does_not_match(name):
    assert not _FUSED_EXPERT_RE.search(name), (
        f"Expected {name!r} to NOT match the BMM-fused expert pattern "
        f"(would incorrectly trigger transpose on 0-D scale tensor)"
    )


def test_modelopt_perexp_re_remap_gate_proj():
    name = "model.language_model.layers.3.mlp.experts.gate_proj.7.weight_packed"
    m = _MODELOPT_PEREXP_RE.match(name)
    assert m is not None
    remapped = (
        f"{m.group('prefix')}.{m.group('idx')}.{m.group('proj')}{m.group('suffix')}"
    )
    assert remapped == (
        "model.language_model.layers.3.mlp.experts.7.gate_proj.weight_packed"
    )


def test_modelopt_perexp_re_remap_down_proj_scale_2():
    name = "model.language_model.layers.47.mlp.experts.down_proj.123.weight_scale_2"
    m = _MODELOPT_PEREXP_RE.match(name)
    assert m is not None
    remapped = (
        f"{m.group('prefix')}.{m.group('idx')}.{m.group('proj')}{m.group('suffix')}"
    )
    assert remapped == (
        "model.language_model.layers.47.mlp.experts.123.down_proj.weight_scale_2"
    )


@pytest.mark.parametrize(
    "name",
    [
        # Already-remapped Mixtral-style names must NOT match (would double-remap)
        "model.layers.0.mlp.experts.7.gate_proj.weight_packed",
        # Truly-fused names have no per-expert index
        "model.layers.0.mlp.experts.down_proj.weight_packed",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
        # Unrelated names
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ],
)
def test_modelopt_perexp_re_does_not_match(name):
    assert _MODELOPT_PEREXP_RE.match(name) is None, (
        f"Expected {name!r} to NOT match the ModelOpt per-expert pattern"
    )


def test_modelopt_perexp_re_rejects_non_decimal_index():
    # `\d+` must not match hex-style or alphanumeric segments.
    name = "model.layers.0.experts.gate_proj.0xff.weight_packed"
    assert _MODELOPT_PEREXP_RE.match(name) is None
