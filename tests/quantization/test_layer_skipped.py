# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``is_layer_skipped`` covering both the legacy exact-match
default and the substring-match opt-in used by AWQ / FP8 quantization
configs.

The substring-match path is the regression target for issue #21669:
HuggingFace-style ``modules_to_not_convert`` patterns (e.g.
``"linear_attn.in_proj_qkv"``) must skip the corresponding layer even when
the runtime prefix is a fully qualified path
(e.g. ``"language_model.model.layers.0.linear_attn.in_proj_qkv"``).
"""

from types import MappingProxyType

import pytest

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

# ---------------------------------------------------------------------------
# Exact-match path (legacy default, still in use by some non-AWQ callers).
# ---------------------------------------------------------------------------


def test_exact_match_full_path_hit():
    assert (
        is_layer_skipped(
            prefix="model.layers.0.lm_head",
            ignored_layers=["model.layers.0.lm_head"],
        )
        is True
    )


def test_exact_match_short_pattern_misses():
    """Reproduces the bug: a short pattern fails exact match and the layer
    is silently quantized.
    """
    assert (
        is_layer_skipped(
            prefix="language_model.model.layers.0.linear_attn.in_proj_qkv",
            ignored_layers=["linear_attn.in_proj_qkv"],
        )
        is False
    )


# ---------------------------------------------------------------------------
# Substring-match path (default for AWQ family; opted into by FP8 family
# in this PR).
# ---------------------------------------------------------------------------


def test_substr_match_short_pattern_hits():
    """The fix: short pattern matches the runtime prefix via substring,
    so ignored layers are correctly skipped.
    """
    assert (
        is_layer_skipped(
            prefix="language_model.model.layers.0.linear_attn.in_proj_qkv",
            ignored_layers=["linear_attn.in_proj_qkv"],
            skip_with_substr=True,
        )
        is True
    )


def test_substr_match_full_path_still_hits():
    """Backwards compatibility: a fully qualified pattern still works under
    substring match (it is a substring of itself).
    """
    assert (
        is_layer_skipped(
            prefix="model.layers.0.self_attn.q_proj",
            ignored_layers=["model.layers.0.self_attn.q_proj"],
            skip_with_substr=True,
        )
        is True
    )


def test_substr_match_unrelated_misses():
    assert (
        is_layer_skipped(
            prefix="model.layers.0.mlp.down_proj",
            ignored_layers=["linear_attn.in_proj_qkv"],
            skip_with_substr=True,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Fused mapping path (qkv_proj / gate_up_proj). Both shards must agree.
# ---------------------------------------------------------------------------


def test_fused_mapping_substr_hit():
    fused = MappingProxyType({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    assert (
        is_layer_skipped(
            prefix="model.layers.0.self_attn.qkv_proj",
            ignored_layers=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
            ],
            fused_mapping=fused,
            skip_with_substr=True,
        )
        is True
    )


def test_fused_mapping_partial_skip_raises():
    fused = MappingProxyType({"qkv_proj": ["q_proj", "k_proj", "v_proj"]})
    with pytest.raises(ValueError, match="same precision"):
        is_layer_skipped(
            prefix="model.layers.0.self_attn.qkv_proj",
            ignored_layers=["self_attn.q_proj"],
            fused_mapping=fused,
            skip_with_substr=True,
        )


# ---------------------------------------------------------------------------
# Experts path. Preserves the legacy MoE convention where an ignored layer
# can be a coarse prefix of the runtime layer name.
# ---------------------------------------------------------------------------


def test_experts_path_exact_default():
    assert (
        is_layer_skipped(
            prefix="model.layers.0.block_sparse_moe.experts.0.w1",
            ignored_layers=["model.layers.0.block_sparse_moe.experts.0.w1"],
        )
        is True
    )


def test_experts_path_substr():
    assert (
        is_layer_skipped(
            prefix="model.layers.0.block_sparse_moe.experts.0.w1",
            ignored_layers=["block_sparse_moe.experts.0"],
            skip_with_substr=True,
        )
        is True
    )


# ---------------------------------------------------------------------------
# Empty ignored_layers: never skip.
# ---------------------------------------------------------------------------


def test_empty_ignored_layers():
    assert (
        is_layer_skipped(
            prefix="model.layers.0.lm_head",
            ignored_layers=[],
        )
        is False
    )
    assert (
        is_layer_skipped(
            prefix="model.layers.0.lm_head",
            ignored_layers=[],
            skip_with_substr=True,
        )
        is False
    )
