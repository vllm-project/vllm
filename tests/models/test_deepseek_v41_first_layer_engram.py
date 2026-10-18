# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An Engram layer that opens the local stack (model entry, or the first layer
of a later PP stage) injects before its attention pre like every other Engram
layer, instead of taking the broadcast-embedding shortcut."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.deepseek_v41.nvidia.model as dsv41

TOKENS, HC, DIM = 3, 4, 8


class RecordingEngram(nn.Module):
    layer_hash_index = 1

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, hash_ids, token_mask=None):
        self.calls.append((hidden_states.clone(), hash_ids, token_mask))
        # Per-copy offset, so a skipped injection is visible downstream.
        offset = torch.arange(1, HC + 1, dtype=hidden_states.dtype)
        return hidden_states + offset.view(1, HC, 1)


def _layer(engram):
    layer = dsv41.DeepseekV4DecoderLayer.__new__(dsv41.DeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.mhc_stream = None
    layer.fuse_mhc_all_reduce = False
    layer.use_sequence_parallel = False
    layer.engram = engram
    layer.hc_mult = HC
    layer.hidden_size = DIM
    layer.rms_norm_eps = 1e-6
    layer.hc_eps = 1e-6
    layer.hc_post_alpha = 2.0
    layer.hc_sinkhorn_iters = 1
    mix = (2 + HC) * HC
    layer.hc_attn_fn = torch.zeros(mix, HC * DIM)
    layer.hc_attn_fn_broadcast = torch.zeros(mix, DIM)
    layer.hc_attn_scale = layer.hc_ffn_scale = torch.zeros(3)
    layer.hc_attn_base = layer.hc_ffn_base = torch.zeros(mix)
    layer.hc_ffn_fn = torch.zeros(mix, HC * DIM)
    layer.attn_norm = layer.ffn_norm = SimpleNamespace(
        weight=torch.ones(DIM), variance_epsilon=1e-6
    )
    layer.attn = lambda positions, x, _: x
    layer.ffn = lambda x, input_ids, metadata=None: x
    return layer


@pytest.fixture
def pre_calls(monkeypatch):
    calls = []

    def mhc_pre(residual, fn, *args, pre_mix=None, x=None, **kwargs):
        calls.append(
            SimpleNamespace(residual=residual.clone(), fn=fn, pre_mix=pre_mix, x=x)
        )
        # Collapse with the carried pre-mix, or copy 0 at model entry.
        mix = pre_mix if pre_mix is not None else torch.eye(HC)[0].expand(TOKENS, HC)
        layer_input = (mix.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)
        post = torch.zeros(TOKENS, HC, 1)
        comb = torch.zeros(TOKENS, HC, HC)
        return post, comb, layer_input, torch.zeros(TOKENS, HC)

    def shifted_post_pre(x, residual, post_mix, res_mix, *args, pre_mix=None, **kwargs):
        return residual, post_mix, res_mix, x, torch.zeros(TOKENS, HC), None

    monkeypatch.setattr(dsv41, "mhc_pre_delayed_tilelang", mhc_pre)
    monkeypatch.setattr(dsv41, "mhc_shifted_post_pre", shifted_post_pre)
    return calls


def _forward(layer, x, pre_mix=None, hashes=None):
    return layer(
        x,
        torch.arange(TOKENS),
        torch.arange(TOKENS),
        pre_mix=pre_mix,
        engram_hashes=hashes,
        engram_mask=None,
    )


def test_engram_at_model_entry_injects_before_pre(pre_calls):
    engram = RecordingEngram()
    layer = _layer(engram)
    embedding = torch.randn(TOKENS, DIM, dtype=torch.bfloat16)
    hashes = torch.arange(TOKENS * 2 * 5).view(TOKENS, 2, 5)

    _forward(layer, embedding, hashes=hashes)

    assert len(engram.calls) == 1, "Engram skipped on the first local layer"
    seen, seen_hashes, _ = engram.calls[0]
    torch.testing.assert_close(
        seen, embedding.unsqueeze(1).expand(-1, HC, -1), rtol=0, atol=0
    )
    torch.testing.assert_close(seen_hashes, hashes[:, 1], rtol=0, atol=0)
    (call,) = pre_calls
    # The injected copies differ: the full projection over all copies, copy 0
    # selected at model entry.
    assert call.fn is layer.hc_attn_fn and call.x is None and call.pre_mix is None
    offset = torch.arange(1, HC + 1, dtype=torch.bfloat16).view(1, HC, 1)
    torch.testing.assert_close(call.residual, seen + offset, rtol=0, atol=0)


def test_engram_at_pp_stage_entry_injects_with_carried_pre_mix(pre_calls):
    engram = RecordingEngram()
    layer = _layer(engram)
    stream = torch.randn(TOKENS, HC, DIM, dtype=torch.bfloat16)
    carried = torch.softmax(torch.randn(TOKENS, HC), -1)
    hashes = torch.arange(TOKENS * 2 * 5).view(TOKENS, 2, 5)

    _forward(layer, stream, pre_mix=carried, hashes=hashes)

    assert len(engram.calls) == 1, "Engram skipped on the first PP-stage layer"
    torch.testing.assert_close(engram.calls[0][0], stream, rtol=0, atol=0)
    (call,) = pre_calls
    assert call.fn is layer.hc_attn_fn and call.pre_mix is carried
    offset = torch.arange(1, HC + 1, dtype=torch.bfloat16).view(1, HC, 1)
    torch.testing.assert_close(call.residual, stream + offset, rtol=0, atol=0)


def test_first_layer_without_engram_keeps_broadcast_projection(pre_calls):
    layer = _layer(None)
    embedding = torch.randn(TOKENS, DIM, dtype=torch.bfloat16)

    _forward(layer, embedding)

    (call,) = pre_calls
    assert call.fn is layer.hc_attn_fn_broadcast and call.x is embedding
