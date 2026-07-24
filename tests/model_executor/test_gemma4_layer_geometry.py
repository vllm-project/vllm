# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer attention geometry resolution for the native Gemma4 impls."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.gemma4 import _get_layer_attention_geometry


def _legacy_config(**kwargs):
    """Config in the layout used before huggingface/transformers#47384: full
    attention values live on the global config as `global_head_dim` and
    `num_global_key_value_heads`."""
    defaults = dict(
        layer_types=["sliding_attention", "full_attention"],
        head_dim=256,
        num_key_value_heads=8,
        global_head_dim=512,
    )
    return SimpleNamespace(**{**defaults, **kwargs})


def test_legacy_config_reads_global_full_attention_values():
    config = _legacy_config(attention_k_eq_v=True, num_global_key_value_heads=1)

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (512, 1)


def test_legacy_config_without_k_eq_v_keeps_global_kv_heads():
    """`num_global_key_value_heads` only applies when `attention_k_eq_v` is
    set; otherwise every layer shares `num_key_value_heads`."""
    config = _legacy_config(num_global_key_value_heads=1)

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (512, 8)


def test_heterogeneous_config_reads_per_layer_values():
    """Transformers >= 5.15.0 deleted `global_head_dim` and
    `num_global_key_value_heads` in favour of `per_layer_config`
    (huggingface/transformers#47384), and reading the per-layer attributes off
    the global config raises, so resolution must go through the layer copy."""
    pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        head_dim=256,
        num_key_value_heads=8,
    )
    config.per_layer_config = {1: {"head_dim": 512, "num_key_value_heads": 1}}

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (512, 1)


def test_homogeneous_config_on_new_transformers():
    """Not every Gemma4 variant declares dual head dimensions; a homogeneous
    config still carries the mixin, so the same path must return the global
    values rather than raising or mis-resolving."""
    pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        head_dim=256,
        num_key_value_heads=8,
    )

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (256, 8)


def test_legacy_layout_config_on_new_transformers():
    """Published Gemma 4 checkpoints still use the old config layout, and the
    heterogeneity mixin is present on every config from Transformers 5.15.0
    on. Keying off the mixin instead of the layout resolves such a config
    against a `per_layer_config` that carries no overrides, silently returning
    the sliding attention head_dim for full attention layers."""
    pytest.importorskip(
        "transformers.integrations.heterogeneity.configuration_utils",
        reason="requires transformers with heterogeneous config support",
    )
    from transformers import PreTrainedConfig

    config = PreTrainedConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        head_dim=256,
        num_key_value_heads=8,
        global_head_dim=512,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
    )
    assert not config.is_heterogeneous

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (512, 1)


def test_old_transformers_without_dual_head_dims():
    """A Gemma 4 variant with a single head dimension on Transformers
    < 5.15.0 has neither `global_head_dim` nor `per_layer_config`."""
    config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        head_dim=256,
        num_key_value_heads=8,
    )

    assert _get_layer_attention_geometry(config, 0) == (256, 8)
    assert _get_layer_attention_geometry(config, 1) == (256, 8)
