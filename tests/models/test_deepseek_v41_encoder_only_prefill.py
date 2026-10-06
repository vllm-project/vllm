# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.kv_transfer import dsv41_encoder_only_boundary_layer
from vllm.models.deepseek_v41.nvidia import model as dsv41_model

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Embedding(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(1).expand(-1, 2)


class _RecordingLayer(dsv41_model.DeepseekV4DecoderLayer):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.full_calls = 0
        self.cache_calls = 0
        self.global_cache: torch.Tensor | None = None

    def forward(self, x, *args, **kwargs):
        self.full_calls += 1
        x = x + 1
        state = torch.ones_like(x)
        return x, state, state, state, state, None

    def write_global_cache(self, x, *args, **kwargs):
        self.cache_calls += 1
        self.global_cache = x.clone()
        return x


@pytest.mark.parametrize("decoder_replay_start", [21, 40])
def test_encoder_only_prefill_stops_at_global_cache_boundary(
    monkeypatch, decoder_replay_start
):
    """The 40-layer path runs 20 full layers plus the L20 producer only."""
    monkeypatch.setattr(
        dsv41_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    model = dsv41_model.DeepseekV4Model.__new__(dsv41_model.DeepseekV4Model)
    torch.nn.Module.__init__(model)
    layers = [_RecordingLayer() for _ in range(40)]
    model.embed_tokens = _Embedding()
    model.layers = torch.nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.use_native_mega_moe = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = set()
    model.encoder_only_prefill = True
    model.decoder_replay_start = decoder_replay_start
    model.decoder_replay_layers = Mock()
    model.encoder_only_boundary_layer = dsv41_encoder_only_boundary_layer(
        SimpleNamespace(
            num_hidden_layers=40,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        )
    )

    input_ids = torch.tensor([2, 3])
    output = model(input_ids, torch.arange(2), None)

    expected_boundary_input = model.embed_input_ids(input_ids) + 20
    torch.testing.assert_close(output, expected_boundary_input)
    torch.testing.assert_close(layers[20].global_cache, expected_boundary_input)
    assert sum(layer.full_calls for layer in layers) == 20
    assert sum(layer.cache_calls for layer in layers) == 1
    assert all(layer.full_calls == 0 for layer in layers[20:])
    model.decoder_replay_layers.assert_not_called()
