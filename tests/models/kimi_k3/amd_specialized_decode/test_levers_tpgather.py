# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``levers.tpgather`` (CPU): the draft fc's TP sharding gives the replicated
fc's output, and the gate refuses what it cannot shard."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.amd.specialized_decode.levers import tpgather


class _Fc(nn.Module):
    def __init__(self, n_in, n_out, dtype=torch.bfloat16, bias=False):
        super().__init__()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        g = torch.Generator().manual_seed(0)
        self.input_size, self.output_size, self.params_dtype = n_in, n_out, dtype
        self.weight = nn.Parameter(
            torch.randn(n_out, n_in, generator=g).to(dtype), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(n_out)) if bias else None
        self.quant_method = UnquantizedLinearMethod()


def _patch_linear(monkeypatch, tp):
    from vllm import distributed
    from vllm.model_executor.layers import linear

    monkeypatch.setattr(linear, "ReplicatedLinear", _Fc)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: tp)


def _model(fc, aux=True):
    return SimpleNamespace(model=SimpleNamespace(fc=fc, use_aux_hidden_state=aux))


@pytest.mark.parametrize("tp", [2, 8])
def test_sharded_fc_matches_replicated(monkeypatch, tp):
    fc = _Fc(5 * 64, 128)
    x = torch.randn(3, 5 * 64).bfloat16()
    want = torch.nn.functional.linear(x, fc.weight)
    shards = [tpgather.TPShardedFC(fc, r, tp) for r in range(tp)]
    outs = [fc.quant_method.apply(s, x) for s in shards]
    got = torch.cat(outs, dim=-1)
    torch.testing.assert_close(got, want, rtol=0.02, atol=0.05)
    # forward gathers this rank's columns.
    monkeypatch.setattr(tpgather, "all_gather_last", lambda y: torch.cat(outs, -1))
    torch.testing.assert_close(shards[0](x), got)
    assert shards[0].weight.shape == (128 // tp, 5 * 64)
    assert (shards[0].input_size, shards[0].output_size) == (5 * 64, 128)


def test_draft_fc_gate(monkeypatch):
    _patch_linear(monkeypatch, 8)
    assert tpgather.draft_fc_why_not(_model(_Fc(64, 128))) is None
    assert "no aux-hidden" in tpgather.draft_fc_why_not(_model(_Fc(64, 128), False))
    assert "bias" in tpgather.draft_fc_why_not(_model(_Fc(64, 128, bias=True)))
    assert "split" in tpgather.draft_fc_why_not(_model(_Fc(64, 132)))
    assert "float32" in tpgather.draft_fc_why_not(
        _model(_Fc(64, 128, dtype=torch.float32))
    )
    _patch_linear(monkeypatch, 1)
    assert tpgather.draft_fc_why_not(_model(_Fc(64, 128))) == "TP 1"


def test_shard_draft_fc_swaps_once(monkeypatch):
    from vllm import distributed

    _patch_linear(monkeypatch, 4)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: 1)
    m = _model(_Fc(64, 128))
    w = m.model.fc.weight.data
    assert tpgather.shard_draft_fc(m)
    assert isinstance(m.model.fc, tpgather.TPShardedFC)
    torch.testing.assert_close(m.model.fc.weight.data, w[32:64])
    fc = m.model.fc
    assert tpgather.shard_draft_fc(m) and m.model.fc is fc
