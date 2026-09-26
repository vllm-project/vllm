# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash model and config behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.qwen3_dflash import (
    _dflash_layer_causal,
    _DFlashAttention,
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
)
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


def _config(num_hidden_layers, layer_types=None, causal_override=None, is_causal=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # DFlash2 stores the explicit attention semantics at the top level.
        (
            _config(
                2,
                layer_types=["sliding_attention"] * 2,
                is_causal=False,
            ),
            True,
        ),
        (
            _config(2, layer_types=["full_attention"] * 2, is_causal=True),
            False,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_honors_top_level_override():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=False,
    )
    assert _dflash_layer_causal(config, 0) is False
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


def test_dflash_target_layers_map_to_post_layer_boundaries():
    vllm_config = _vllm_config(
        dflash_config={"target_layer_ids": [0, 16, 31]},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == (
        1,
        17,
        32,
    )


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )


class _MultipleOf16Backend:
    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return [MultipleOf(16)]


class _FixedBlockBackend(_MultipleOf16Backend):
    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return [16, 32, 64]


def _make_attention(
    *,
    kv_cache_block_size: int | None = None,
    backend=None,
):
    attention = _DFlashAttention.__new__(_DFlashAttention)
    nn.Module.__init__(attention)
    attention.attn_type = AttentionType.DECODER
    attention.sliding_window = None
    attention.kv_cache_dtype = "auto"
    attention.kv_cache_torch_dtype = torch.bfloat16
    attention.num_kv_heads = 16
    attention.head_size = 128
    attention.head_size_v = 128
    attention.kv_cache_block_size = kv_cache_block_size
    attention.attn_backend = backend
    attention.layer_name = "draft"
    return attention


@pytest.mark.parametrize(
    ("kv_cache_block_size", "expected_block_size"),
    [(128, 128), (None, 1152)],
)
def test_dflash_kv_block_size_override(
    kv_cache_block_size: int | None,
    expected_block_size: int,
):
    attention = _make_attention(
        kv_cache_block_size=kv_cache_block_size,
        backend=_MultipleOf16Backend,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1152),
    )

    spec = attention.get_kv_cache_spec(vllm_config)

    assert spec.block_size == expected_block_size


def test_dflash_kv_block_size_rejects_backend_that_requires_splitting():
    attention = _make_attention(
        kv_cache_block_size=128,
        backend=_FixedBlockBackend,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=1152),
    )

    with pytest.raises(NotImplementedError, match="attention_backend=FLASH_ATTN"):
        attention.get_kv_cache_spec(vllm_config)
