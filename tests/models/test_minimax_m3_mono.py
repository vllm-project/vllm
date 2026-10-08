# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned mono initialization and cache lifetime, without GPU allocation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.models.minimax_m3.amd import mono
from vllm.models.minimax_m3.amd.model import MiniMaxM3Model, MiniMaxM3SparseAttention
from vllm.v1.worker.utils import clear_layer_kv_caches


@pytest.fixture
def model(monkeypatch):
    attn = MiniMaxM3SparseAttention.__new__(MiniMaxM3SparseAttention)
    nn.Module.__init__(attn)
    attn.kv_cache = torch.tensor([])
    attn.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(kv_cache=torch.tensor([]))
    )
    layer = nn.Module()
    layer.self_attn = attn
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module(), nn.Module(), nn.Module(), layer])
    model._mono = None
    monkeypatch.setattr(
        mono, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return model


@pytest.mark.parametrize("main,index", [(False, False), (True, False), (False, True)])
def test_initial_memory_profile_waits_for_both_caches(model, monkeypatch, main, index):
    attn = model.layers[3].self_attn
    if main:
        attn.bind_kv_cache(torch.ones(4))
    if index:
        attn.indexer.index_cache.kv_cache = torch.ones(4)
    construct = Mock()
    monkeypatch.setattr(mono, "M3Mono", construct)
    mono.prepare_model(model, object())
    construct.assert_not_called()


def test_cache_generations_close_and_rebuild(model, monkeypatch):
    attn = model.layers[3].self_attn
    runtimes = [SimpleNamespace(close=Mock()), SimpleNamespace(close=Mock())]
    construct = Mock(side_effect=runtimes)
    monkeypatch.setattr(mono, "M3Mono", construct)
    for runtime in runtimes:
        cache = torch.ones(4)
        attn.bind_kv_cache(cache)
        attn.indexer.index_cache.kv_cache = torch.ones(4)
        mono.prepare_model(model, object())
        assert model._mono is runtime
        attn.kv_cache_k = cache[:2]
        attn.kv_cache_v = cache[2:]
        attn._aiter_sparse_pa_cache_data_ptr = cache.data_ptr()
        with pytest.raises(ValueError, match="detach caches"):
            attn.bind_kv_cache(torch.ones(4))
        assert attn.kv_cache is cache
        runtime.close.assert_not_called()
        # Normal profiling/shutdown has destroyed graphs before this call.
        clear_layer_kv_caches([attn])
        clear_layer_kv_caches([attn])
        runtime.close.assert_called_once()
        assert model._mono is None
        assert attn.kv_cache_k.numel() == attn.kv_cache_v.numel() == 0
        assert attn._aiter_sparse_pa_cache_data_ptr == 0


def test_first_initialization_cannot_happen_inside_capture(model, monkeypatch):
    attn = model.layers[3].self_attn
    attn.bind_kv_cache(torch.ones(4))
    attn.indexer.index_cache.kv_cache = torch.ones(4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="eager forward before graph capture"):
        mono.prepare_model(model, object())
    assert model._mono is None


def test_weight_transfer_refused_before_constructing_layers(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_ATOM_M3_MONO", "1")
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=object()),
        cache_config=None,
        quant_config=None,
        use_v2_model_runner=True,
        weight_transfer_config=object(),
    )
    with pytest.raises(ValueError, match="weight transfer is unsupported"):
        MiniMaxM3Model(vllm_config=config)
