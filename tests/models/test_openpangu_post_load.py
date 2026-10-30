# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenPangu MTP refreshes weight-derived sink tensors after loading."""

import torch
from torch import nn

from vllm.model_executor.models.openpangu_mtp import OpenPanguMTP


class _FakeSinkAttention(nn.Module):
    """Caches a tensor derived from a parameter, like the sink KV."""

    def __init__(self) -> None:
        super().__init__()
        self.param_sink_key = nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        self.sink_key = self.param_sink_key.detach().clone()

    def post_weight_load(self) -> None:
        self.sink_key = self.param_sink_key.detach().clone()


def test_mtp_refreshes_nested_sink_tensors_after_load() -> None:
    model = object.__new__(OpenPanguMTP)
    nn.Module.__init__(model)
    sink = _FakeSinkAttention()
    model.model = nn.Sequential(nn.Sequential(sink))
    loaded = torch.arange(8, dtype=torch.float32).view(2, 4)
    sink.param_sink_key.data.copy_(loaded)

    model.process_weights_after_loading()

    torch.testing.assert_close(sink.sink_key, loaded)


def test_sink_kv_update_follows_the_real_kv_cache(monkeypatch) -> None:
    """``update_sink_kv`` on the real StaticSinkAttention: the KV cache is one
    tensor (empty until bound, as ``Attention.__init__`` leaves it), so a
    load before allocation defers the population to the first forward and a
    reload writes the bound cache at once."""
    from vllm.model_executor.layers.attention.static_sink_attention import (
        StaticSinkAttention,
    )

    layer = object.__new__(StaticSinkAttention)
    nn.Module.__init__(layer)
    layer.kv_cache = torch.tensor([])  # as Attention.__init__ leaves it
    populated: list[torch.Tensor] = []
    monkeypatch.setattr(layer, "populate_sink_kv", populated.append)

    layer.update_sink_kv(torch.ones(1), torch.ones(1))
    assert layer.sink_populated is False and populated == []

    cache = torch.zeros(2, 3)
    StaticSinkAttention.bind_kv_cache(layer, cache)
    layer.update_sink_kv(torch.ones(1), torch.ones(1))
    assert populated == [cache]
