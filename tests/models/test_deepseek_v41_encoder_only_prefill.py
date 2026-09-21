# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.nvidia import model as dsv41_model

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_official_index_sources_do_not_move_encoder_only_boundary(monkeypatch):
    """Later indexers share L20 K; they must not move the producer cut."""

    class ConstructionReached(Exception):
        pass

    def stop_after_topology_check():
        raise ConstructionReached

    monkeypatch.setattr(dsv41_model, "_use_sequence_parallel", lambda _: False)
    monkeypatch.setattr(dsv41_model.torch.cuda, "Stream", stop_after_topology_check)
    hf_config = SimpleNamespace(
        num_hidden_layers=40,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        engram_layer_ids=[1, 14],
        vocab_size=128,
        hc_eps=1e-6,
        hc_mult=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        kernel_config=SimpleNamespace(moe_backend="CUTLASS"),
        is_dsv41_encoder_only_prefill=True,
    )

    with pytest.raises(ConstructionReached):
        dsv41_model.DeepseekV4Model(vllm_config=vllm_config)


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


def test_encoder_only_prefill_stops_at_global_cache_boundary(monkeypatch):
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
    model.use_mega_moe = False
    model.engram_hash = None
    model.engram_dp_shared_memory = False
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = set()
    model.encoder_only_prefill = True
    model.encoder_only_boundary_layer = 20

    input_ids = torch.tensor([2, 3])
    output = model(input_ids, torch.arange(2), None)

    expected_boundary_input = model.embed_input_ids(input_ids) + 20
    torch.testing.assert_close(output, expected_boundary_input)
    torch.testing.assert_close(layers[20].global_cache, expected_boundary_input)
    assert sum(layer.full_calls for layer in layers) == 20
    assert sum(layer.cache_calls for layer in layers) == 1
    assert all(layer.full_calls == 0 for layer in layers[20:])


def test_global_cache_writer_publishes_same_latent_to_main_and_indexer():
    positions = torch.arange(4)
    hidden_states = torch.randn(4, 8)
    score = torch.randn(4, 8)
    latent = torch.randn(4, 8)
    compressor = Mock(return_value=latent)
    indexer = SimpleNamespace(owns_k=True, insert_cache=Mock())
    attention = SimpleNamespace(
        compressor=compressor,
        indexer=indexer,
        is_kv_source=True,
        layer_id=20,
        aux_stream_list=None,
        ln_events=[None, None],
        indexer_rotary_emb=object(),
        rotary_emb=object(),
        _compressor_kv_score=Mock(return_value=score),
    )

    DeepseekV4Attention.write_global_cache(attention, positions, hidden_states)

    attention._compressor_kv_score.assert_called_once_with(hidden_states)
    compressor.assert_called_once_with(score, positions)
    compressor.insert_cache.assert_called_once_with(
        latent, positions, attention.rotary_emb
    )
    indexer.insert_cache.assert_called_once_with(
        latent, positions, attention.indexer_rotary_emb
    )
