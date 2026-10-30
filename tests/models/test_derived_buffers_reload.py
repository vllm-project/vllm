# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffers that models derive from their weights keep their addresses when the
post-load step reruns (weight reload), with the new values: captured CUDA
graphs keep reading the startup tensors."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


def _param(*shape: int) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape), requires_grad=False)


def _module(**tensors: torch.Tensor) -> nn.Module:
    module = nn.Module()
    for name, tensor in tensors.items():
        setattr(module, name, tensor)
    return module


def _rebuild_keeps_addresses(build, reload_weights, names: list[str], obj):
    build()
    first = {n: getattr(obj, n) for n in names}
    ptrs = {n: t.data_ptr() for n, t in first.items()}
    reload_weights()
    build()
    for n in names:
        assert getattr(obj, n) is first[n], n
        assert getattr(obj, n).data_ptr() == ptrs[n], n
    return first


def test_qwen3_dflash_context_buffers(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash

    monkeypatch.setattr(qwen3_dflash, "_reads_raw_weights", lambda projs: True)
    model = qwen3_dflash.DFlashQwen3Model.__new__(qwen3_dflash.DFlashQwen3Model)
    nn.Module.__init__(model)
    model.hidden_norm = _module(weight=_param(8))
    q_size, kv_size, head_dim = 4, 2, 2
    attns = [
        _module(
            q_size=q_size,
            qkv_proj=_module(weight=_param(q_size + 2 * kv_size, 8), bias=_param(8)),
            k_norm=_module(weight=_param(head_dim)),
        )
        for _ in range(2)
    ]

    def reload_weights():
        for a in attns:
            for t in (a.qkv_proj.weight, a.qkv_proj.bias, a.k_norm.weight):
                t.data.add_(1.0)

    names = ["_fused_kv_weight", "_fused_kv_bias", "_k_norm_weights"]
    _rebuild_keeps_addresses(
        lambda: model._build_context_kv_buffers(attns, has_bias=True),
        reload_weights,
        names,
        model,
    )
    torch.testing.assert_close(
        model._fused_kv_weight,
        torch.cat([a.qkv_proj.weight[q_size:] for a in attns]),
    )
    torch.testing.assert_close(
        model._k_norm_weights, torch.stack([a.k_norm.weight for a in attns])
    )


def test_lilicorr_inference_buffers():
    from vllm.model_executor.models.lilicorr import LiLiCorrHead

    head = LiLiCorrHead.__new__(LiLiCorrHead)
    nn.Module.__init__(head)
    hidden, factor, heads = 4, 3, 2
    head.candidate_topk, head.block_size, head.num_candidate_slots = 2, 4, 3
    head.hidden_size = hidden
    head.relative_slot_bias = _param(heads, 2 * head.block_size - 1)
    head.same_slot_bias = _param(heads)
    head.out_head = _module(weight=_param(factor, hidden), bias=_param(factor))
    head.in_head = _module(weight=_param(factor, hidden), bias=_param(factor))
    head.factor_input_proj = _module(weight=_param(hidden, 3 * hidden))
    head._fused_edge_weight = head._fused_edge_bias = head._attn_bias = None
    head._factor_input_splits = ()

    def reload_weights():
        for p in head.parameters():
            p.data.add_(1.0)

    names = [
        "_attn_bias",
        "_fused_edge_weight",
        "_fused_edge_bias",
        "_rank_frac_col",
        "_is_top1_col",
    ]
    splits = []

    def build():
        head.materialize_inference_buffers(torch.device("cpu"), torch.float32)
        splits.append([s.data_ptr() for s in head._factor_input_splits])

    _rebuild_keeps_addresses(build, reload_weights, names, head)
    assert splits[0] == splits[1]
    torch.testing.assert_close(
        head._fused_edge_weight,
        torch.cat([head.out_head.weight, head.in_head.weight]),
    )
    torch.testing.assert_close(
        head._factor_input_splits[1],
        head.factor_input_proj.weight[:, hidden : 2 * hidden],
    )


def test_muse_spark_residual_coefficients():
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    connection = MuseSpark1xNormPreservingResidualConnection(4)
    with torch.no_grad():
        connection.gate.fill_(0.1)

    def reload_weights():
        with torch.no_grad():
            connection.gate.fill_(0.5)

    _rebuild_keeps_addresses(
        connection.materialize_coefficients,
        reload_weights,
        ["alpha", "beta"],
        connection,
    )
    alpha, beta = connection.coefficients(connection.gate / connection.temperature)
    torch.testing.assert_close(connection.alpha, alpha)
    torch.testing.assert_close(connection.beta, beta)


def test_kimi_k3_specialized_decode_refresh(monkeypatch):
    pytest.importorskip("cutlass")
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.dense import (
        PreparedDenseMLP,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import decode

    # dense layer 0: its padded (TP16) down projection is a copy; MoE layer 1:
    # its packed front is a copy
    dense = PreparedDenseMLP.__new__(PreparedDenseMLP)
    dense.hidden, dense.padded = 4, 6
    dense.down = SimpleNamespace(weight=torch.zeros(3, 6))
    packed = torch.zeros(5)
    refreshed: list[int] = []
    sd = decode.SpecializedDecode.__new__(decode.SpecializedDecode)
    sd.rank, sd.tp = 0, 1
    dense_mlp = _module(down_proj=_module(weight=_param(3, 4)))
    moe_mlp = _module(weight=_param(5))
    sd.layers = [
        SimpleNamespace(is_moe_layer=False, layer_idx=0, mlp=dense_mlp, self_attn=0),
        SimpleNamespace(is_moe_layer=True, layer_idx=1, mlp=moe_mlp, self_attn=1),
    ]
    sd.packed = {1: packed}
    sd.feed_forward = [dense, None]
    core = SimpleNamespace(refresh_weights=refreshed.append)
    sd.attention = [SimpleNamespace(core=core), SimpleNamespace(core=None)]
    monkeypatch.setattr(decode, "pack_front", lambda mlp, rank, tp: mlp.weight * 2)

    down, ptrs = dense.down.weight, (dense.down.weight.data_ptr(), packed.data_ptr())
    sd.refresh_weights()
    assert (dense.down.weight.data_ptr(), sd.packed[1].data_ptr()) == ptrs
    assert dense.down.weight is down
    torch.testing.assert_close(down[:, :4], dense_mlp.down_proj.weight)
    assert (down[:, 4:] == 0).all()  # padding columns stay zero
    torch.testing.assert_close(sd.packed[1], moe_mlp.weight * 2)
    assert refreshed == [0]
