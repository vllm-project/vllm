# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA q|k|v short convs are one fused parameter.

The conv kernels read ``conv1d.weight`` directly, so an RL refit must reach it
through every write path (#55087), and FULL cudagraphs capture its address, so
it must never move.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.glm5next.common import kda
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

CHANNELS, KERNEL = 8, 4


def _checkpoint_weights(seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(CHANNELS, 1, KERNEL, generator=gen) for _ in range(3)]


def _load(conv1d: nn.Module, weights: list[torch.Tensor]) -> None:
    for shard_id, weight in enumerate(weights):
        conv1d.weight.weight_loader(conv1d.weight, weight, shard_id)


def _kernel_view(conv1d: nn.Module) -> torch.Tensor:
    return conv1d.weight.view(conv1d.weight.size(0), conv1d.weight.size(-1))


@pytest.fixture
def conv1d(dist_init) -> nn.Module:
    return kda._fused_qkv_conv1d(KERNEL, CHANNELS, 1, 0, prefix="conv1d")


def test_fused_conv_loads_checkpoint_shards_in_qkv_order(conv1d):
    weights = _checkpoint_weights(seed=0)
    _load(conv1d, [w.view(CHANNELS, KERNEL) for w in weights[:1]] + weights[1:])
    assert conv1d.weight.shape == (3 * CHANNELS, 1, KERNEL)
    assert torch.equal(_kernel_view(conv1d), torch.cat(weights).squeeze(1))


def test_refit_updates_fused_conv_without_moving_it(conv1d):
    """Refits through load_weights (RL weight sync) and through the layerwise
    reload (``reload_weights``) both land at the original address."""
    data_ptr = conv1d.weight.data_ptr()
    _load(conv1d, _checkpoint_weights(seed=0))

    refit = _checkpoint_weights(seed=1)
    _load(conv1d, refit)
    assert torch.equal(_kernel_view(conv1d), torch.cat(refit).squeeze(1))

    reload = _checkpoint_weights(seed=2)
    model = nn.Sequential(conv1d)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    _load(conv1d, reload)
    finalize_layerwise_reload(model, model_config=None)
    assert torch.equal(_kernel_view(conv1d), torch.cat(reload).squeeze(1))
    assert conv1d.weight.data_ptr() == data_ptr


def _decode_layer(conv1d: nn.Module) -> kda.Glm5NextLinearAttention:
    layer = object.__new__(kda.Glm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.conv1d = conv1d
    layer.local_num_heads, layer.head_dim = 1, CHANNELS
    layer.local_projection_size = CHANNELS
    layer.kda_safe_gate, layer.kda_lower_bound = True, -5.0
    layer.A_log = layer.dt_bias = None
    layer._conv_state_dim_first = True
    layer.kv_cache = (torch.zeros(1, 3 * CHANNELS, KERNEL - 1), None)
    return layer


def test_decode_reads_refitted_fused_conv(conv1d, monkeypatch):
    """Decode must hand the conv kernel ``conv1d.weight`` itself, not a copy
    cached by an earlier forward, so refits reach the address FULL cudagraphs
    capture (#55087)."""
    layer = _decode_layer(conv1d)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(1, dtype=torch.int32),
    )
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )
    seen = []

    def conv_update(x, conv_state, weight, *args, **kwargs):
        seen.append(weight)
        return x

    monkeypatch.setattr(kda, "causal_conv1d_update", conv_update)
    monkeypatch.setattr(
        kda, "fused_recurrent_kda", lambda **kwargs: (kwargs["out"], None)
    )

    def decode() -> torch.Tensor:
        layer._forward(
            torch.zeros(1, 3 * CHANNELS),
            torch.zeros(1, 1, 1, CHANNELS),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1, CHANNELS),
        )
        return seen.pop()

    for seed in (0, 1):
        weights = _checkpoint_weights(seed)
        _load(conv1d, weights)
        weight = decode()
        assert weight.data_ptr() == conv1d.weight.data_ptr()
        assert torch.equal(weight, torch.cat(weights).squeeze(1))
