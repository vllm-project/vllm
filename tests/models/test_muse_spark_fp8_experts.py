# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize-on-load of MuseSpark1x online FP8 routed experts (CPU only)."""

import types

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.quantization.online import fp8 as online_fp8
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.muse_spark.nvidia import fp8_moe
from vllm.models.muse_spark.nvidia.fp8_moe import (
    MuseSparkFp8FusedMoE,
    quantize_fp8_rows,
)
from vllm.models.muse_spark.nvidia.muse_moe_runner import MuseSparkRoutedExperts

E, H, INTER, TP = 4, 32, 24, 2
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def online_config(**kwargs) -> OnlineQuantizationConfig:
    return OnlineQuantizationConfig(QuantizationConfigArgs(**kwargs))


def make_layer(tp_rank: int) -> MuseSparkRoutedExperts:
    """The experts layer with only what weight loading touches."""
    layer = MuseSparkRoutedExperts.__new__(MuseSparkRoutedExperts)
    torch.nn.Module.__init__(layer)
    moe = types.SimpleNamespace(
        tp_size=TP,
        tp_rank=tp_rank,
        is_act_and_mul=True,
        tp_shard_with_padding=False,
        moe_parallel_config=types.SimpleNamespace(tp_size=TP),
        num_experts=E,
        w13_num_shards=2,
        has_bias=False,
    )
    layer.moe_config = moe
    layer.quant_config = online_config(moe="fp8_per_channel")
    layer.layer_name = "experts"
    layer.expert_map_manager = types.SimpleNamespace(num_fused_shared_experts=0)
    layer.shared_expert_online_loader = None
    layer._loaded_expert_biases = set()
    layer._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    method = MuseSparkFp8FusedMoE.__new__(MuseSparkFp8FusedMoE)
    method.moe = moe
    method.moe_kernel = None
    method.requantization_source = None  # set by __init__, skipped here
    layer.quant_method = method
    method.create_weights(
        layer, E, H, INTER // TP, torch.bfloat16, weight_loader=layer.weight_loader
    )
    return layer


def checkpoint(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    # Rows six decades apart, as real expert rows can be.
    w13 = (
        torch.randn(E, 2 * INTER, H, generator=g)
        * torch.logspace(-3, 3, 2 * INTER)[:, None]
    )
    w2 = torch.randn(E, H, INTER, generator=g) * torch.logspace(-3, 3, H)[:, None]
    return w13.bfloat16(), w2.bfloat16()


def load(layer: MuseSparkRoutedExperts, w13: torch.Tensor, w2: torch.Tensor) -> None:
    list(layer.load_weights([("w13_weight", w13), ("w2_weight", w2)]))


def expected_shard(w13: torch.Tensor, w2: torch.Tensor, rank: int):
    rows = slice(rank * INTER // TP, (rank + 1) * INTER // TP)
    w1, w3 = w13.float().chunk(2, dim=1)
    return torch.cat([w1[:, rows], w3[:, rows]], dim=1), w2.float()[:, :, rows]


def assert_dequantizes(layer, w13, w2, rank):
    exp13, exp2 = expected_shard(w13, w2, rank)
    for name, exp in (("w13", exp13), ("w2", exp2)):
        deq = getattr(layer, f"{name}_weight").float() * getattr(
            layer, f"{name}_weight_scale"
        )
        tol = 0.07 * exp.abs().amax(-1, keepdim=True)
        assert ((deq - exp).abs() <= tol).all(), name


def test_quantize_rows_round_trip():
    w = torch.randn(8, 64) * torch.logspace(-3, 3, 8)[:, None]
    q, scale = quantize_fp8_rows(w)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (8, 1)
    assert (
        (q.float() * scale - w).abs() <= 0.07 * w.abs().amax(-1, keepdim=True)
    ).all()


def test_tp_shards_quantize_their_own_slice():
    w13, w2 = checkpoint(0)
    for rank in range(TP):
        layer = make_layer(rank)
        load(layer, w13, w2)
        assert layer.w13_weight.dtype == torch.float8_e4m3fn
        assert_dequantizes(layer, w13, w2, rank)
        # One scale per row of this rank's slice: a w1/w3 row is whole, a w2
        # row spans this rank's slice of its input dim. fp8_per_channel floors
        # scales at 1 / (448 * 512).
        for name, exp in zip(("w13", "w2"), expected_shard(w13, w2, rank)):
            amax = exp.abs().amax(-1, keepdim=True)
            expected = (amax / FP8_MAX).clamp_min(1 / (FP8_MAX * 512))
            scale = getattr(layer, f"{name}_weight_scale")
            assert torch.allclose(scale, expected), name


def test_loader_moves_only_this_ranks_slice(monkeypatch):
    """Each rank moves to the device, and quantizes, only its TP slice."""
    shapes = []

    def recording(w):
        shapes.append(tuple(w.shape))  # the tensor moved to param.device
        return quantize_fp8_rows(w)

    monkeypatch.setattr(fp8_moe, "quantize_fp8_rows", recording)
    load(make_layer(1), *checkpoint(0))
    w1_or_w3, w2 = (INTER // TP, H), (H, INTER // TP)
    assert shapes == [w1_or_w3] * (2 * E) + [w2] * E


def test_moe_only_config_resolves_experts_to_fp8_per_channel():
    prefix = "model.layers.1.feed_forward.experts"
    config = online_config(moe="fp8_per_channel")
    assert config.args.linear is None
    resolved = config.resolve_quant_method_cls(MuseSparkRoutedExperts, prefix)
    assert resolved is not None
    assert resolved[-1] is online_fp8.Fp8PtpcOnlineMoEMethod
    ignored = online_config(moe="fp8_per_channel", ignore=[prefix])
    assert ignored.resolve_quant_method_cls(MuseSparkRoutedExperts, prefix) is None


def test_direct_reload_requantizes_in_place():
    """In-process RL weight sync: plain load_weights, then a weight-sync guard
    reruns process_weights_after_loading and rejects new tensors."""
    layer = make_layer(1)
    load(layer, *checkpoint(0))
    layer.quant_method.moe_kernel = object()  # built once at startup
    before = {n: (p, p.data_ptr()) for n, p in layer.named_parameters()}
    w13, w2 = checkpoint(1)
    load(layer, w13, w2)
    layer.quant_method.process_weights_after_loading(layer)
    for n, p in layer.named_parameters():
        assert before[n] == (p, p.data_ptr()), n
    assert_dequantizes(layer, w13, w2, 1)


def test_layerwise_reload_requantizes_into_kernel_tensors():
    """Native weight sync: the layerwise reload must process the layer once
    every expert has loaded and copy into the startup tensors."""
    layer = make_layer(0)
    load(layer, *checkpoint(0))
    layer.quant_method.moe_kernel = object()
    record_metadata_for_reloading(layer)
    ptrs = {n: p.data_ptr() for n, p in layer.named_parameters()}
    initialize_layerwise_reload(layer)
    assert layer.w13_weight.is_meta
    w13, w2 = checkpoint(2)
    load(layer, w13, w2)
    # Processed during loading, not deferred to finalize.
    assert not layer.w13_weight.is_meta
    finalize_layerwise_reload(layer, None)
    assert {n: p.data_ptr() for n, p in layer.named_parameters()} == ptrs
    assert_dequantizes(layer, w13, w2, 0)


def test_chunked_quantizer_breaks_layerwise_reload(monkeypatch):
    """Why quantize_fp8_rows avoids copy_: with fp8_per_channel's chunked helper
    the layerwise reload over-counts the loaded elements and processes the
    layer early."""

    def chunked(w):
        scale = online_fp8._fp8_channel_scale(
            fp8_moe.weight_amax(w, dim=-1, keepdim=True)
        )
        return online_fp8._fp8_quant_per_channel(w, scale), scale

    monkeypatch.setattr(fp8_moe, "quantize_fp8_rows", chunked)
    with pytest.raises(AssertionError):
        test_layerwise_reload_requantizes_into_kernel_tensors()
