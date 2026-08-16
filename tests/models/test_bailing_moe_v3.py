# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for Ling's fused KDA decode weight shadows."""

import gc
import weakref

import pytest
import torch
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.layers.attention import is_deferred_attention_layer
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bailing_moe_v3 import (
    BailingMoeV3KimiDeltaAttention,
    _make_decode_conv1d_weight_loader,
    _make_decode_norm_weight_loader,
)

_CONV_NAMES = ("q_conv1d", "k_conv1d", "v_conv1d")


def _conv1d_modules(
    layer: BailingMoeV3KimiDeltaAttention,
) -> tuple[nn.Conv1d, ...]:
    return tuple(getattr(layer, name) for name in _CONV_NAMES)


def _assert_decode_shadows(
    layer: BailingMoeV3KimiDeltaAttention,
    expected_conv: torch.Tensor | None = None,
    expected_norm: torch.Tensor | None = None,
) -> None:
    if expected_conv is None:
        expected_conv = torch.stack(
            [conv.weight.squeeze(1).transpose(0, 1) for conv in _conv1d_modules(layer)]
        )
    if expected_norm is None:
        expected_norm = layer.o_norm.weight.float()
    torch.testing.assert_close(layer.decode_conv1d_weight, expected_conv)
    torch.testing.assert_close(layer.decode_norm_weight, expected_norm)


def _make_fused_kda_layer() -> BailingMoeV3KimiDeltaAttention:
    layer = BailingMoeV3KimiDeltaAttention.__new__(BailingMoeV3KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.use_fused_kda_decode = True
    layer.conv_size = 2
    layer.head_dim = 4
    layer.projection_size_per_partition = 4

    for name, value in zip(_CONV_NAMES, (1.0, 2.0, 3.0), strict=True):
        conv1d = nn.Conv1d(1, 4, 2, bias=False)
        conv1d.weight.data.fill_(value)
        setattr(layer, name, conv1d)

    layer.o_norm = nn.RMSNorm(4)
    layer.o_norm.weight.data.copy_(torch.arange(4.0))
    layer.register_buffer(
        "decode_conv1d_weight", torch.empty(3, 2, 4), persistent=False
    )
    layer.register_buffer("decode_norm_weight", torch.empty(4), persistent=False)

    layer_ref = weakref.ref(layer)
    for shard_id, conv1d in enumerate(_conv1d_modules(layer)):
        conv1d.weight.weight_loader = _make_decode_conv1d_weight_loader(
            layer_ref,
            shard_id,
            default_weight_loader,
        )
    layer.o_norm.weight.weight_loader = _make_decode_norm_weight_loader(
        layer_ref,
        default_weight_loader,
    )
    return layer


@pytest.mark.parametrize(
    "meta_shadows",
    [False, True],
    ids=["refresh-existing", "materialize-meta"],
)
def test_fused_kda_post_load_refreshes_shadow_buffers(meta_shadows: bool) -> None:
    """Deferred post-load processing refreshes or materializes shadows."""
    layer = _make_fused_kda_layer()
    if meta_shadows:
        layer.decode_conv1d_weight = torch.empty(
            3, 2, 4, device="meta", dtype=layer.q_conv1d.weight.dtype
        )
        layer.decode_norm_weight = torch.empty(4, device="meta", dtype=torch.float32)
    else:
        layer.decode_conv1d_weight.fill_(float("nan"))
        layer.decode_norm_weight.fill_(float("nan"))

    assert is_deferred_attention_layer(layer)
    layer.process_weights_after_loading(torch.bfloat16)

    assert not layer.decode_conv1d_weight.is_meta
    assert not layer.decode_norm_weight.is_meta
    _assert_decode_shadows(layer)
    if meta_shadows:
        loaded_weight = torch.full_like(layer.q_conv1d.weight, 9.0)
        layer.q_conv1d.weight.weight_loader(layer.q_conv1d.weight, loaded_weight)
        _assert_decode_shadows(layer)


def test_fused_kda_loader_updates_live_shadow_in_place() -> None:
    """Reloading must retain CUDA-graph-visible storage and avoid stale buffers."""
    layer = _make_fused_kda_layer()
    layer.refresh_fused_kda_decode_weights()
    shadow_ptr = layer.decode_conv1d_weight.data_ptr()

    loader = layer.q_conv1d.weight.weight_loader
    loaded_weight = torch.full_like(layer.q_conv1d.weight, 7.0)
    loader(layer.q_conv1d.weight, loaded_weight)

    assert layer.decode_conv1d_weight.data_ptr() == shadow_ptr
    torch.testing.assert_close(
        layer.decode_conv1d_weight[0],
        loaded_weight.squeeze(1).transpose(0, 1),
    )

    norm_shadow_ptr = layer.decode_norm_weight.data_ptr()
    loaded_norm = torch.arange(4.0) + 20
    layer.o_norm.weight.weight_loader(layer.o_norm.weight, loaded_norm)
    assert layer.decode_norm_weight.data_ptr() == norm_shadow_ptr
    _assert_decode_shadows(layer)


def test_fused_kda_loader_weakref_does_not_retain_layer() -> None:
    layer = _make_fused_kda_layer()
    layer_ref = weakref.ref(layer)
    param = layer.q_conv1d.weight
    loader = param.weight_loader

    del layer
    gc.collect()

    assert layer_ref() is None
    loaded_weight = torch.full_like(param, 11.0)
    loader(param, loaded_weight)
    torch.testing.assert_close(param, loaded_weight)


def test_fused_kda_layerwise_reload_refreshes_shadows() -> None:
    """Layerwise reload must refresh shadows without changing live storage."""
    layer = _make_fused_kda_layer()
    layer.refresh_fused_kda_decode_weights()
    model_config = ModelConfig()
    conv_shadow_ptr = layer.decode_conv1d_weight.data_ptr()
    norm_shadow_ptr = layer.decode_norm_weight.data_ptr()

    record_metadata_for_reloading(layer)
    initialize_layerwise_reload(layer)
    conv_values = (4.0, 5.0, 6.0)
    for conv1d, value in zip(_conv1d_modules(layer), conv_values, strict=True):
        loaded = torch.full((4, 1, 2), value)
        conv1d.weight.weight_loader(conv1d.weight, loaded)
    loaded_norm = torch.arange(4.0) + 10
    layer.o_norm.weight.weight_loader(layer.o_norm.weight, loaded_norm)

    finalize_layerwise_reload(layer, model_config)

    assert layer.decode_conv1d_weight.data_ptr() == conv_shadow_ptr
    assert layer.decode_norm_weight.data_ptr() == norm_shadow_ptr
    expected_conv = torch.stack([torch.full((2, 4), value) for value in conv_values])
    _assert_decode_shadows(layer, expected_conv, loaded_norm)


def test_fused_kda_disabled_layer_has_no_shadow_buffers() -> None:
    """Unsupported TP/shape configurations must retain the original path only."""
    layer = BailingMoeV3KimiDeltaAttention.__new__(BailingMoeV3KimiDeltaAttention)
    nn.Module.__init__(layer)
    layer.use_fused_kda_decode = False
    layer.register_buffer("decode_conv1d_weight", None, persistent=False)
    layer.register_buffer("decode_norm_weight", None, persistent=False)

    layer.refresh_fused_kda_decode_weights()

    assert layer.decode_conv1d_weight is None
    assert layer.decode_norm_weight is None
