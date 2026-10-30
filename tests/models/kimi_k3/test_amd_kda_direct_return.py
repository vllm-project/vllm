# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

import vllm.models.kimi_k3.amd.kda as amd_kda
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.amd.linear import KimiDecoderLayer


class _ReturningAttention(nn.Module):
    def __init__(self, result: torch.Tensor) -> None:
        super().__init__()
        self.result = result

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.result


class _TupleProjection(nn.Module):
    def __init__(self, result: torch.Tensor) -> None:
        super().__init__()
        self.result = result

    def forward(self, _: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.result, None


def _make_layer(self_attn: nn.Module) -> KimiDecoderLayer:
    layer = object.__new__(KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = self_attn
    return layer


def test_kda_self_attention_returns_projection_storage_directly() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)
    layer = _make_layer(_ReturningAttention(projected))

    output = layer._run_self_attn(torch.arange(4), hidden_states)

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


def test_k3_kda_forward_returns_output_projection_storage() -> None:
    hidden_states = torch.randn(4, 8)
    projected = torch.randn_like(hidden_states)

    def fill_core_output(**kwargs: torch.Tensor) -> None:
        kwargs["core_attn_out"].fill_(1)

    attention = SimpleNamespace(
        local_projection_size=4,
        head_dim=2,
        local_num_heads=2,
        in_proj_padding=0,
        in_proj_qkvgfab=_TupleProjection(torch.randn(4, 20)),
        f_b_proj=_TupleProjection(torch.randn(4, 4)),
        o_proj=_TupleProjection(projected),
        _forward=fill_core_output,
    )

    output = KimiK3DeltaAttention.forward(attention, hidden_states, torch.arange(4))

    assert output is projected
    assert output.data_ptr() == projected.data_ptr()


# Stand-ins for the KDA kernels: simple functions of their inputs that honour
# out=, so a test can follow which rows go where without the GPU kernels.


def _conv_update(x, *_, out, **__):
    return out.copy_(x * 2)


def _qkv_conv(x, *_, **__):
    y = x.t() * 3
    kd = y.shape[1] // 3
    return y[:, :kd], y[:, kd : 2 * kd], y[:, 2 * kd :]


def _with_out(o, out):
    if out is None:
        return o, None
    return out.copy_(o), None


def _recurrent(q, k, v, raw_g, raw_beta, out=None, **_):
    return _with_out(q + k * v + raw_g + raw_beta[..., None], out)


def _chunk(q, k, v, raw_g, raw_beta, out=None, **_):
    return _with_out(q * k - v + 2 * raw_g + raw_beta[..., None], out)


def _mixed_spec_metadata(num_spec_tokens):
    """Two spec decodes of 3 tokens, then a 5-token prefill."""
    spec_rows = torch.arange(6)
    return amd_kda.GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=5,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=6,
        num_actual_tokens=11,
        has_initial_state=torch.tensor([True]),
        spec_query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        non_spec_state_indices_tensor=torch.tensor([7]),
        spec_sequence_masks=torch.tensor([True, True, False]),
        # The view path must not read the token indices
        spec_token_indx=None if num_spec_tokens else spec_rows,
        non_spec_token_indx=None if num_spec_tokens else torch.arange(6, 11),
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32),
        uniform_spec_sequence_length=3,
    )


def _run_mixed_spec_forward(monkeypatch, num_spec_tokens):
    num_heads, head_dim, num_tokens = 2, 4, 11
    proj_size = num_heads * head_dim
    torch.manual_seed(0)
    # mixed_qkv is a column view of the projection output, as in the model
    projection = torch.randn(num_tokens, 3 * proj_size + 5)
    mixed_qkv = projection[:, : 3 * proj_size]
    g1 = torch.randn(1, num_tokens, num_heads, head_dim)
    g2 = torch.randn(num_tokens, num_heads, head_dim)
    beta = torch.randn(1, num_tokens, num_heads)
    core_attn_out = torch.zeros(1, num_tokens, num_heads, head_dim)
    conv_state = torch.zeros(8, 3 * proj_size, 3)
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2).contiguous()

    metadata = _mixed_spec_metadata(num_spec_tokens)
    monkeypatch.setattr(
        amd_kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"kda": metadata}),
    )
    monkeypatch.setattr(amd_kda, "causal_conv1d_update", _conv_update)
    monkeypatch.setattr(amd_kda, "fused_recurrent_kda", _recurrent)
    monkeypatch.setattr(amd_kda, "chunk_kda_prefill", _chunk)

    layer = SimpleNamespace(
        prefix="kda",
        kv_cache=(conv_state, torch.zeros(8, num_heads, head_dim, head_dim)),
        use_aiter_decode=False,
        decode_conv1d_weight=None,
        decode_norm_weight=None,
        conv1d=SimpleNamespace(weight=torch.randn(3 * proj_size, 1, 4), bias=None),
        local_projection_size=proj_size,
        head_dim=head_dim,
        A_log=None,
        dt_bias=None,
        gate_lower_bound=None,
        _fused_qkv_conv_fn=_qkv_conv,
        kda_prefill_backend=None,
        o_norm=lambda x, _: x,
    )
    KimiK3DeltaAttention._forward(layer, mixed_qkv, g1, g2, beta, core_attn_out)
    return core_attn_out


def test_leading_spec_rows_split_with_views(monkeypatch):
    """With num_spec_tokens set, the layer splits the batch with views and both
    paths write core_attn_out directly; the result matches the gather path."""
    gathered = _run_mixed_spec_forward(monkeypatch, num_spec_tokens=None)
    viewed = _run_mixed_spec_forward(monkeypatch, num_spec_tokens=6)

    assert gathered.abs().sum() > 0
    torch.testing.assert_close(viewed, gathered, rtol=0, atol=0)
