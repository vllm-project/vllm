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
