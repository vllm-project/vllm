# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `_remap_modelopt_qwen3_vl_moe_name` (issue #40885).

ModelOpt-quantized Qwen3-VL-MoE checkpoints (e.g.
`Code4me2/bu-30b-a3b-preview-NVFP4`) emit per-expert weight tensors
with the projection name before the expert index and prefix the LLM
inside the `language_model.` segment of the
`Qwen3VLMoEForConditionalGeneration` wrapper:

  model.language_model.layers.0.mlp.experts.gate_proj.42.weight

vLLM's `FusedMoE.make_expert_params_mapping` produces weight names
with the expert index *before* the projection:

  model.layers.0.mlp.experts.42.gate_proj.weight

Until the remap was added, loading such a checkpoint either dropped
expert weights silently or — when the substring match in
`load_weights` hit `experts.down_proj` / `experts.gate_up_proj` —
crashed at `transpose(-1, -2)` on a 1-D scale tensor
(`IndexError: Dimension out of range (...) but got -2`).

These tests pin the remap behaviour without needing a real checkpoint.
"""

from __future__ import annotations

import pytest

from vllm.model_executor.models.qwen3_vl_moe import (
    _remap_modelopt_qwen3_vl_moe_name,
)


@pytest.mark.parametrize(
    "modelopt_name,expected_vllm_name",
    [
        # 2-D weight tensor — the common case.
        (
            "model.language_model.layers.0.mlp.experts.gate_proj.42.weight",
            "model.layers.0.mlp.experts.42.gate_proj.weight",
        ),
        (
            "model.language_model.layers.5.mlp.experts.up_proj.7.weight",
            "model.layers.5.mlp.experts.7.up_proj.weight",
        ),
        (
            "model.language_model.layers.23.mlp.experts.down_proj.127.weight",
            "model.layers.23.mlp.experts.127.down_proj.weight",
        ),
        # 0-D / 1-D NVFP4 scales — same swap, all suffixes covered.
        (
            "model.language_model.layers.0.mlp.experts.gate_proj.42.weight_scale",
            "model.layers.0.mlp.experts.42.gate_proj.weight_scale",
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj.0.weight_scale_2",
            "model.layers.0.mlp.experts.0.down_proj.weight_scale_2",
        ),
        (
            "model.language_model.layers.0.mlp.experts.up_proj.99.input_scale",
            "model.layers.0.mlp.experts.99.up_proj.input_scale",
        ),
        (
            "model.language_model.layers.0.mlp.experts.gate_proj.10.input_scale_2",
            "model.layers.0.mlp.experts.10.gate_proj.input_scale_2",
        ),
    ],
)
def test_remap_modelopt_nvfp4_names(
    modelopt_name: str, expected_vllm_name: str
) -> None:
    assert _remap_modelopt_qwen3_vl_moe_name(modelopt_name) == expected_vllm_name


@pytest.mark.parametrize(
    "modelopt_name,expected",
    [
        # Non-expert tensors keep the language_model strip but no
        # projection/expert swap (regex doesn't match).
        (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.language_model.layers.0.input_layernorm.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "lm_head.weight",
            "lm_head.weight",
        ),
    ],
)
def test_remap_strips_language_model_prefix_only(
    modelopt_name: str, expected: str
) -> None:
    assert _remap_modelopt_qwen3_vl_moe_name(modelopt_name) == expected


def test_remap_is_idempotent_on_already_vllm_names() -> None:
    """Passing an already-remapped name through the function again must
    not damage it. Important because the iterator wraps every weight,
    including those from non-ModelOpt (non-prefixed) checkpoints."""
    cases = [
        "model.layers.0.mlp.experts.42.gate_proj.weight",
        "model.layers.0.mlp.experts.42.gate_proj.weight_scale_2",
        "model.embed_tokens.weight",
    ]
    for name in cases:
        once = _remap_modelopt_qwen3_vl_moe_name(name)
        twice = _remap_modelopt_qwen3_vl_moe_name(once)
        assert once == name, f"unexpected change: {name!r} → {once!r}"
        assert twice == once, f"not idempotent: {once!r} → {twice!r}"


def test_remap_does_not_touch_non_matching_paths() -> None:
    """Sanity: text outside the `mlp.experts.<proj>.<digits>.` pattern
    is untouched even if it superficially resembles the regex."""
    cases = [
        # No digits after projection — not an expert weight.
        "model.layers.0.mlp.experts.gate_proj.weight",
        # Digit before projection (already vLLM-shaped, see idempotence).
        "model.layers.0.mlp.experts.42.gate_proj.weight",
        # Projection name appears outside `experts.` context.
        "model.layers.0.self_attn.gate_proj_like_name.weight",
    ]
    for name in cases:
        assert _remap_modelopt_qwen3_vl_moe_name(name) == name
