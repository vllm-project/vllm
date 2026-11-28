# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construction tests for Cohere2Moe swiglu_limit plumbing.

These do not need a GPU. Heavy linear / attention / MoE kernels and the
CustomOp activations are stubbed so the test only checks which activation
and FusedMoEFactory kwargs the model definition constructs.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.model_executor.models import cohere2_moe as cohere


class _DummyLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _DummyMoE(nn.Module):
    pass


class _DummyAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _PlainActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _ClampActivation(nn.Module):
    def __init__(self, swiglu_limit, alpha=1.0, beta=0.0, **kwargs):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.alpha = alpha
        self.beta = beta


def _base_config(**overrides):
    cfg = dict(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        norm_topk_prob=True,
        mlp_layer_types=["dense"],
        layer_norm_eps=1e-5,
        max_position_embeddings=32,
        rope_parameters={"rope_theta": 10000, "rope_type": "default"},
        sliding_window=None,
        layer_types=["full_attention"],
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@pytest.fixture
def stub_heavy_ops(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_factory(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _DummyMoE()

    monkeypatch.setattr(cohere, "MergedColumnParallelLinear", _DummyLinear)
    monkeypatch.setattr(cohere, "RowParallelLinear", _DummyLinear)
    monkeypatch.setattr(cohere, "QKVParallelLinear", _DummyLinear)
    monkeypatch.setattr(cohere, "GateLinear", _DummyLinear)
    monkeypatch.setattr(cohere, "Attention", _DummyAttention)
    monkeypatch.setattr(cohere, "get_rope", lambda *args, **kwargs: object())
    monkeypatch.setattr(cohere, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(cohere, "FusedMoEFactory", fake_factory)
    monkeypatch.setattr(cohere, "SiluAndMul", _PlainActivation)
    monkeypatch.setattr(cohere, "SiluAndMulWithClamp", _ClampActivation)
    return captured


def test_mlp_without_swiglu_limit_uses_silu_and_mul(stub_heavy_ops):
    mlp = cohere.Cohere2MoeMLP(_base_config())
    assert isinstance(mlp.act_fn, _PlainActivation)
    assert mlp.swiglu_limit is None


def test_mlp_with_swiglu_limit_uses_clamped_silu(stub_heavy_ops):
    mlp = cohere.Cohere2MoeMLP(_base_config(), swiglu_limit=7.0)
    assert isinstance(mlp.act_fn, _ClampActivation)
    assert mlp.act_fn.swiglu_limit == 7.0
    assert mlp.act_fn.alpha == 1.0
    assert mlp.act_fn.beta == 0.0


def test_moe_without_swiglu_limit_forwards_none(stub_heavy_ops):
    captured = stub_heavy_ops
    block = cohere.Cohere2Moe(_base_config())
    assert captured["swiglu_limit"] is None
    assert isinstance(block.shared_experts.act_fn, _PlainActivation)


def test_moe_with_swiglu_limit_reaches_factory_and_shared_mlp(stub_heavy_ops):
    captured = stub_heavy_ops
    block = cohere.Cohere2Moe(_base_config(swiglu_limit=7.0))
    assert captured["swiglu_limit"] == 7.0
    assert isinstance(block.shared_experts.act_fn, _ClampActivation)
    assert block.shared_experts.act_fn.swiglu_limit == 7.0


def test_dense_decoder_layer_without_swiglu_limit(stub_heavy_ops):
    layer = cohere.Cohere2MoeDecoderLayer(
        _base_config(),
        prefix="model.layers.0",
    )
    assert isinstance(layer.mlp.act_fn, _PlainActivation)


def test_dense_decoder_layer_with_swiglu_limit(stub_heavy_ops):
    layer = cohere.Cohere2MoeDecoderLayer(
        _base_config(swiglu_limit=7.0),
        prefix="model.layers.0",
    )
    assert isinstance(layer.mlp.act_fn, _ClampActivation)
    assert layer.mlp.act_fn.swiglu_limit == 7.0


def test_sparse_decoder_layer_forwards_limit_to_fused_moe(stub_heavy_ops):
    captured = stub_heavy_ops
    layer = cohere.Cohere2MoeDecoderLayer(
        _base_config(swiglu_limit=7.0, mlp_layer_types=["sparse"]),
        prefix="model.layers.0",
    )
    assert captured["swiglu_limit"] == 7.0
    assert isinstance(layer.mlp.shared_experts.act_fn, _ClampActivation)
