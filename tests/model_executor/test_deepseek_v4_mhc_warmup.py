# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.warmup.deepseek_v4_mhc_warmup as mhc_warmup


class _Norm:
    def __init__(self, hidden_size: int, eps: float):
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps


class _NvidiaDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4
        self.hc_mult = 2
        self.rms_norm_eps = 1e-5
        self.hc_eps = 2e-5
        self.hc_post_alpha = 2.0
        self.hc_sinkhorn_iters = 3

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = torch.nn.Parameter(
            torch.empty(mix_hc, hc_dim), requires_grad=False
        )
        self.hc_attn_scale = torch.nn.Parameter(torch.empty(3), requires_grad=False)
        self.hc_attn_base = torch.nn.Parameter(torch.empty(mix_hc), requires_grad=False)
        self.hc_attn_fn_broadcast = None
        self.hc_ffn_fn = torch.nn.Parameter(
            torch.empty(mix_hc, hc_dim), requires_grad=False
        )
        self.hc_ffn_scale = torch.nn.Parameter(torch.empty(3), requires_grad=False)
        self.hc_ffn_base = torch.nn.Parameter(torch.empty(mix_hc), requires_grad=False)
        self.attn_norm = _Norm(self.hidden_size, 3e-5)
        self.ffn_norm = _Norm(self.hidden_size, 4e-5)


_NvidiaDecoderLayer.__name__ = "DeepseekV4DecoderLayer"


class _CustomOpDecoderLayer(_NvidiaDecoderLayer):
    def __init__(self):
        super().__init__()
        del self.attn_norm
        del self.ffn_norm
        self.pre_calls = 0
        self.post_calls = 0

    def hc_pre(self, residual, fn, scale, base):
        self.pre_calls += 1
        num_tokens = residual.shape[0]
        post_mix = torch.empty(num_tokens, self.hc_mult, 1)
        res_mix = torch.empty(num_tokens, self.hc_mult, self.hc_mult)
        layer_input = torch.empty(num_tokens, self.hidden_size, dtype=torch.bfloat16)
        return layer_input, post_mix, res_mix

    def hc_post(self, layer_input, residual, post_mix, res_mix):
        self.post_calls += 1
        return residual


_CustomOpDecoderLayer.__name__ = "DeepseekV4DecoderLayer"


class _NvidiaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.hc_mult = 2
        self.rms_norm_eps = 1e-5
        self.hc_eps = 2e-5
        self.hc_head_fn = torch.nn.Parameter(torch.empty(2, 8), requires_grad=False)
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.hc_head_base = torch.nn.Parameter(torch.empty(2), requires_grad=False)


_NvidiaModel.__name__ = "DeepseekV4Model"


class _CustomOpModel(_NvidiaModel):
    def __init__(self):
        super().__init__()
        self.head_calls = 0

    def hc_head_op(self, hidden_states, fn, scale, base, rms_eps, hc_eps):
        self.head_calls += 1
        return torch.empty(
            hidden_states.shape[0],
            self.config.hidden_size,
            dtype=torch.bfloat16,
        )


def test_find_first_mhc_layer_accepts_nvidia_implementation():
    model = torch.nn.Module()
    model.layer = _NvidiaDecoderLayer()

    assert mhc_warmup._find_first_mhc_layer(model) is model.layer


def test_find_first_mhc_layer_keeps_custom_op_implementation():
    model = torch.nn.Module()
    model.layer = _CustomOpDecoderLayer()

    assert mhc_warmup._find_first_mhc_layer(model) is model.layer


def test_warmup_nvidia_layer_uses_forward_tilelang_ops(monkeypatch):
    layer = _NvidiaDecoderLayer()
    layer.hc_attn_fn_broadcast = torch.empty(
        (2 + layer.hc_mult) * layer.hc_mult,
        layer.hidden_size,
    )
    call_order = []
    pre_calls = []
    broadcast_calls = []
    fused_calls = []
    post_calls = []

    def mhc_pre(*args, **kwargs):
        call_order.append("pre")
        pre_calls.append((args, kwargs))
        residual = args[0]
        num_tokens = residual.shape[0]
        return (
            torch.empty(num_tokens, layer.hc_mult, 1),
            torch.empty(num_tokens, layer.hc_mult, layer.hc_mult),
            torch.empty(num_tokens, layer.hidden_size, dtype=torch.bfloat16),
        )

    def mhc_pre_broadcast(*args, **kwargs):
        call_order.append("broadcast")
        broadcast_calls.append((args, kwargs))
        residual = args[0]
        num_tokens = residual.shape[0]
        return (
            torch.empty(
                num_tokens,
                layer.hc_mult,
                layer.hidden_size,
                dtype=torch.bfloat16,
            ),
            torch.empty(num_tokens, layer.hc_mult, 1),
            torch.empty(num_tokens, layer.hc_mult, layer.hc_mult),
            torch.empty(num_tokens, layer.hidden_size, dtype=torch.bfloat16),
        )

    def mhc_fused_post_pre(*args, **kwargs):
        call_order.append("fused")
        fused_calls.append((args, kwargs))
        residual = args[1]
        num_tokens = residual.shape[0]
        return (
            torch.empty_like(residual),
            torch.empty(num_tokens, layer.hc_mult, 1),
            torch.empty(num_tokens, layer.hc_mult, layer.hc_mult),
            torch.empty(num_tokens, layer.hidden_size, dtype=torch.bfloat16),
        )

    def mhc_post(*args, **kwargs):
        call_order.append("post")
        post_calls.append((args, kwargs))
        return torch.empty_like(args[1])

    monkeypatch.setattr(
        mhc_warmup,
        "_get_tilelang_mhc_ops",
        lambda: (
            mhc_pre,
            mhc_pre_broadcast,
            mhc_fused_post_pre,
            mhc_post,
            lambda *args: None,
        ),
        raising=False,
    )

    mhc_warmup._warmup_layer_mhc(layer, [1, 3])

    assert call_order == ["broadcast"] * 2 + ["pre", "fused", "post"] * 2
    assert [call[0][0].shape for call in broadcast_calls] == [(1, 4), (3, 4)]
    assert all(call[0][1] is layer.hc_attn_fn for call in broadcast_calls)
    assert all(
        call[1]["fn_broadcast"] is layer.hc_attn_fn_broadcast
        for call in broadcast_calls
    )
    assert all(
        call[1]["norm_weight"].data_ptr() == layer.attn_norm.weight.data_ptr()
        for call in broadcast_calls
    )
    assert all(
        call[1]["norm_eps"] == layer.attn_norm.variance_epsilon
        for call in broadcast_calls
    )
    assert [call[0][0].shape[0] for call in pre_calls] == [1, 3]
    assert all(call[0][1] is layer.hc_attn_fn for call in pre_calls)
    assert all(
        call[1]["norm_weight"].data_ptr() == layer.attn_norm.weight.data_ptr()
        for call in pre_calls
    )
    assert all(
        call[1]["norm_eps"] == layer.attn_norm.variance_epsilon for call in pre_calls
    )
    assert all(
        call[0][4:9]
        == (
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
        )
        for call in pre_calls
    )

    assert all(call[0][4] is layer.hc_ffn_fn for call in fused_calls)
    assert all(
        call[1]["norm_weight"].data_ptr() == layer.ffn_norm.weight.data_ptr()
        for call in fused_calls
    )
    assert all(
        call[1]["norm_eps"] == layer.ffn_norm.variance_epsilon for call in fused_calls
    )
    assert all(
        call[0][7:12]
        == (
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
        )
        for call in fused_calls
    )
    assert all(call[1]["n_splits"] == 1 for call in fused_calls)
    assert all(call[1]["tile_n"] == 1 for call in fused_calls)
    assert len(post_calls) == 2


def test_warmup_custom_op_layer_keeps_existing_path(monkeypatch):
    layer = _CustomOpDecoderLayer()

    def fail_if_tilelang_loaded():
        raise AssertionError("custom-op layers must not load NVIDIA TileLang ops")

    monkeypatch.setattr(
        mhc_warmup,
        "_get_tilelang_mhc_ops",
        fail_if_tilelang_loaded,
        raising=False,
    )

    mhc_warmup._warmup_layer_mhc(layer, [1, 3])

    assert layer.pre_calls == 4
    assert layer.post_calls == 4


def test_warmup_nvidia_hc_head_uses_forward_tilelang_op(monkeypatch):
    model = _NvidiaModel()
    calls = []

    def hc_head(*args):
        calls.append(args)
        hidden_states = args[0]
        return torch.empty(
            hidden_states.shape[0],
            model.config.hidden_size,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(
        mhc_warmup,
        "_get_tilelang_mhc_ops",
        lambda: (None, None, None, None, hc_head),
        raising=False,
    )

    mhc_warmup._warmup_hc_head(model, [1, 3])

    assert [call[0].shape[0] for call in calls] == [1, 3]
    assert all(call[1] is model.hc_head_fn for call in calls)
    assert all(call[4:] == (model.rms_norm_eps, model.hc_eps) for call in calls)


def test_warmup_custom_hc_head_keeps_existing_path(monkeypatch):
    model = _CustomOpModel()

    def fail_if_tilelang_loaded():
        raise AssertionError("custom-op models must not load NVIDIA TileLang ops")

    monkeypatch.setattr(
        mhc_warmup,
        "_get_tilelang_mhc_ops",
        fail_if_tilelang_loaded,
        raising=False,
    )

    mhc_warmup._warmup_hc_head(model, [1, 3])

    assert model.head_calls == 2
