# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned mono initialization and cache lifetime, without GPU allocation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.models.minimax_m3.amd import mono, mono_weights
from vllm.models.minimax_m3.amd.model import MiniMaxM3SparseAttention
from vllm.v1.worker.utils import clear_layer_kv_caches

_model_layers = mono._model_layers


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
    model._mono_config = object()
    monkeypatch.setattr(
        mono, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(mono, "get_tp_group", lambda: SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(mono, "_model_layers", lambda *args: [3])
    monkeypatch.setattr(mono, "layer_specs", lambda *args: ([], []))
    monkeypatch.setattr(mono.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(mono.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        mono.dist,
        "all_gather_object",
        lambda out, value, group: out.__setitem__(0, value),
    )
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


def test_incompatible_weight_on_a_peer_disables_mono_before_allocation(
    model, monkeypatch
):
    attn = model.layers[3].self_attn
    attn.bind_kv_cache(torch.ones(4))
    attn.indexer.index_cache.kv_cache = torch.ones(4)
    monkeypatch.setattr(mono.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        mono.dist,
        "all_gather_object",
        lambda out, value, group: out.__setitem__(
            slice(None), [value, (True, "rank 1: router weights are FP32")]
        ),
    )
    construct = Mock()
    monkeypatch.setattr(mono, "M3Mono", construct)
    mono.prepare_model(model, object())
    construct.assert_not_called()
    assert model._mono is None and model._mono_config is None


def test_pipeline_partition_falls_back_before_accessing_missing_layers(
    model, monkeypatch
):
    model.layers[3] = nn.Module()
    config = SimpleNamespace(
        use_v2_model_runner=True,
        weight_transfer_config=None,
        cache_config=SimpleNamespace(num_gpu_blocks_override=1024),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=2, data_parallel_size=1
        ),
    )
    monkeypatch.setattr(mono, "_model_layers", _model_layers)
    mono.prepare_model(model, config)
    assert model._mono is None and model._mono_config is None


def test_wait_for_peer_cache_without_disabling_mono(model, monkeypatch):
    attn = model.layers[3].self_attn
    attn.bind_kv_cache(torch.ones(4))
    attn.indexer.index_cache.kv_cache = torch.ones(4)
    monkeypatch.setattr(mono.dist, "get_world_size", lambda group: 2)

    def gather(out, value, group):
        out[:] = [value, False if isinstance(value, bool) else None]

    monkeypatch.setattr(mono.dist, "all_gather_object", gather)
    construct = Mock()
    monkeypatch.setattr(mono, "M3Mono", construct)
    mono.prepare_model(model, object())
    construct.assert_not_called()
    assert model._mono is None and model._mono_config is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_must_already_have_supported_weights_and_logits(dtype):
    layer = SimpleNamespace(
        layer_id=3,
        self_attn=None,
        block_sparse_moe=SimpleNamespace(
            gate=SimpleNamespace(weight=torch.empty(1, dtype=dtype), out_dtype=dtype),
            experts=SimpleNamespace(routed_experts=None),
        ),
    )
    with pytest.raises(ValueError, match="BF16 router weights and FP32 logits"):
        mono_weights._layer_specs(layer, None)


@pytest.mark.parametrize("hipb", [False, True])
def test_projection_binding_borrows_shuffled_storage(hipb):
    cls = (
        AiterHipbMMPerTokenFp8ScaledMMLinearKernel
        if hipb
        else AiterPreshuffledPerTokenFp8ScaledMMLinearKernel
    )
    storage = torch.empty(16, 32, dtype=torch.float8_e4m3fn)
    scale = torch.ones(16, 1)
    linear = SimpleNamespace(
        weight=storage.t() if hipb else storage,
        weight_scale=scale.t() if hipb else scale,
        quant_method=SimpleNamespace(fp8_linear=cls.__new__(cls)),
    )
    weight, scales = mono_weights._projection_weights(linear)
    assert weight.shape == storage.shape and weight.is_contiguous()
    assert weight.data_ptr() == linear.weight.data_ptr()
    assert scales.shape == (16,) and scales.data_ptr() == scale.data_ptr()


def test_fp8_dtype_alone_does_not_qualify_a_projection():
    linear = SimpleNamespace(
        weight=torch.empty(16, 32, dtype=torch.float8_e4m3fn),
        quant_method=object(),
    )
    with pytest.raises(ValueError, match="preshuffled"):
        mono_weights._projection_weights(linear)


def test_large_batches_fall_back_without_disabling_later_small_batches(monkeypatch):
    adapter = mono.M3Mono.__new__(mono.M3Mono)
    adapter.fallback_counts = {}
    adapter.attentions = [
        SimpleNamespace(
            layer_name="main",
            indexer=SimpleNamespace(index_cache=SimpleNamespace(prefix="index")),
        )
    ]
    adapter.runtime = SimpleNamespace(prepare_step=Mock())
    adapter.StepMetadata = SimpleNamespace
    for requests, query_len, tokens, expected in (
        (8, 1, 8, False),
        (4, 1, 4, True),
        (8, 4, 32, False),
        (4, 4, 16, True),
        (4, 1, 16, False),
        (1, 1, 1, True),
    ):
        decode = SimpleNamespace(
            decode_query_len=query_len,
            seq_lens=torch.ones(requests, dtype=torch.int32),
            block_table=torch.zeros(requests, 128, dtype=torch.int32),
            page16_block_table=torch.zeros(requests, 128, dtype=torch.int32),
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=requests,
            decode=decode,
            page16_slot_mapping=torch.zeros(tokens, dtype=torch.int64),
        )
        context = SimpleNamespace(
            attn_metadata={"main": metadata, "index": metadata},
            slot_mapping={"index": torch.zeros(tokens, dtype=torch.int64)},
        )
        monkeypatch.setattr(
            mono, "get_forward_context", lambda context=context: context
        )
        adapter.runtime.prepare_step.reset_mock()
        assert adapter.begin_forward(tokens) == expected
        assert adapter.runtime.prepare_step.call_count == int(expected)
