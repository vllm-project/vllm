# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for the DeepSeek V4 WeightsMapper.

Covers issue #42777: the suffix rule ``"head.weight": "lm_head.weight"``
also matched ``lm_head.weight`` because the latter ends with
``head.weight``, producing ``lm_lm_head.weight`` and breaking the
parameter lookup. The same anti-pattern affected ``embed.weight``.

The fix moves both rules from the broad-matching ``orig_to_new_suffix``
to anchored regex entries (``^head\\.weight$`` and ``^embed\\.weight$``)
in ``orig_to_new_regex``, so already-canonical names are no longer
caught.
"""

import pytest

from vllm.model_executor.models.deepseek_v4 import (
    _make_deepseek_v4_weights_mapper,
)


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_legacy_head_weight_is_remapped(expert_dtype: str) -> None:
    """A legacy ``head.weight`` checkpoint key maps to ``lm_head.weight``."""
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    assert mapper.apply_list(["head.weight"]) == ["lm_head.weight"]


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_canonical_lm_head_weight_is_idempotent(expert_dtype: str) -> None:
    """An already-canonical ``lm_head.weight`` passes through unchanged."""
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    assert mapper.apply_list(["lm_head.weight"]) == ["lm_head.weight"]


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_legacy_embed_weight_is_remapped(expert_dtype: str) -> None:
    """A legacy ``embed.weight`` key maps to ``model.embed_tokens.weight``."""
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    assert mapper.apply_list(["embed.weight"]) == ["model.embed_tokens.weight"]


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_canonical_model_embed_tokens_weight_is_idempotent(
    expert_dtype: str,
) -> None:
    """An already-canonical ``model.embed_tokens.weight`` is unchanged."""
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    assert mapper.apply_list(["model.embed_tokens.weight"]) == [
        "model.embed_tokens.weight"
    ]


@pytest.mark.parametrize("expert_dtype", ["fp4", "fp8"])
def test_unrelated_names_unchanged(expert_dtype: str) -> None:
    """Tensor names that don't match any rule are returned as-is."""
    mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
    for name in [
        "model.lm_head.weight",
        "lm_head.bias",
        "some.completely.unrelated.tensor",
    ]:
        assert mapper.apply_list([name]) == [name]


def test_existing_ffn_norm_remap_still_fires() -> None:
    """The ``.ffn_norm.weight`` -> ``.ffn.norm_gate.norm.weight`` rule lives
    in the same mapper; confirm #42777 didn't regress it."""
    mapper = _make_deepseek_v4_weights_mapper("fp4")
    assert mapper.apply_list(["model.layers.0.ffn_norm.weight"]) == [
        "model.layers.0.ffn.norm_gate.norm.weight"
    ]
