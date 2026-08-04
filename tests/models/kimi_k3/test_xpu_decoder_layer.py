# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.models.kimi_k3.xpu import linear as kimi_xpu
from vllm.models.kimi_k3.xpu.ops.attn_res import attn_res
from vllm.platforms import current_platform


class _ResidualNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return hidden_states + 1
        return hidden_states + residual + 1, residual + hidden_states


class _Scale(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


class _WeightedNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = 1e-5


class _Projection(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, hidden_size))


class _TupleIdentity(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states, None


class _ConstantProjection(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.output, None


class _FakeMLA(nn.Module):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.full((1, 1), 4.0)


def test_mla_output_gate_is_applied_before_output_projection() -> None:
    wrapper = object.__new__(MultiHeadLatentAttentionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.q_lora_rank = None
    wrapper.kv_lora_rank = 1
    wrapper.qk_rope_head_dim = 1
    wrapper.qk_nope_head_dim = 1
    wrapper.qk_head_dim = 2
    wrapper.v_head_dim = 1
    wrapper.num_heads = 1
    wrapper.kv_a_proj_with_mqa = _ConstantProjection(torch.ones(1, 2))
    wrapper.q_proj = _ConstantProjection(torch.ones(1, 2))
    wrapper.kv_a_layernorm = nn.Identity()
    wrapper.rotary_emb = None
    wrapper.indexer = None
    wrapper.is_sparse = False
    wrapper.skip_topk = False
    wrapper.dcp_q_replicate = False
    wrapper.mla_attn = _FakeMLA()
    wrapper.g_proj = _ConstantProjection(torch.zeros(1, 1))
    wrapper.o_proj = _TupleIdentity()

    output = wrapper(torch.tensor([0]), torch.ones(1, 2))

    torch.testing.assert_close(output, torch.full((1, 1), 2.0))


def test_xpu_decoder_layer_runs_attention_then_mlp() -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = False
    layer.input_layernorm = _ResidualNorm()
    layer.post_attention_layernorm = _ResidualNorm()
    layer.mlp = _Scale()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states + 3,
        layer,
    )

    hidden_states = torch.tensor([[1.0, 2.0]])
    output, prefix_sum, residual = layer(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        residual=None,
    )

    torch.testing.assert_close(output, torch.tensor([[14.0, 18.0]]))
    assert prefix_sum is None
    torch.testing.assert_close(residual, torch.tensor([[6.0, 8.0]]))


def test_xpu_decoder_layer_uses_three_attn_res_states(monkeypatch) -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.prev_valid_blocks = 1
    layer.block_write_idx = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = _WeightedNorm(2)
    layer.post_attention_layernorm = _WeightedNorm(2)
    layer.self_attention_res_norm = _WeightedNorm(2)
    layer.mlp_res_norm = _WeightedNorm(2)
    layer.self_attention_res_proj = _Projection(2)
    layer.mlp_res_proj = _Projection(2)
    layer.mlp = _Scale()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states + 3,
        layer,
    )
    calls: list[tuple[torch.Tensor, torch.Tensor | None, int, int]] = []

    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        *args: object,
        num_blocks: int,
        block_write_idx: int,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        calls.append((prefix, delta, num_blocks, block_write_idx))
        if delta is not None:
            prefix.add_(delta)
        return prefix + 100

    monkeypatch.setattr(kimi_xpu, "attn_res", fake_attn_res)
    hidden_states = torch.tensor([[1.0, 2.0]])
    prefix_sum = torch.tensor([[10.0, 20.0]])
    residual = torch.tensor([[[7.0, 8.0]]])

    output, updated_prefix, updated_residual = layer(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        prefix_sum=prefix_sum,
        residual=residual,
    )

    assert len(calls) == 2
    assert calls[0][1] is hidden_states
    assert calls[0][2:] == (1, -1)
    assert calls[1][1] is not None
    assert calls[1][2:] == (1, -1)
    torch.testing.assert_close(updated_prefix, torch.tensor([[125.0, 147.0]]))
    torch.testing.assert_close(output, torch.tensor([[450.0, 494.0]]))
    assert updated_residual is residual


def test_xpu_decoder_layer_resets_prefix_after_block_write(monkeypatch) -> None:
    layer = object.__new__(kimi_xpu.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.prev_valid_blocks = 0
    layer.is_block_write_layer = True
    layer.post_attention_layernorm = _WeightedNorm(2)
    layer.mlp_res_norm = _WeightedNorm(2)
    layer.mlp_res_proj = _Projection(2)
    call: dict[str, object] = {}

    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        *args: object,
        num_blocks: int,
        block_write_idx: int,
        **kwargs: object,
    ) -> torch.Tensor:
        del args, kwargs
        call.update(
            prefix=prefix,
            delta=delta,
            blocks=blocks,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
        )
        return prefix + 100

    monkeypatch.setattr(kimi_xpu, "attn_res", fake_attn_res)
    attention_output = torch.tensor([[3.0, 4.0]])
    old_prefix = torch.tensor([[1.0, 2.0]])
    residual = torch.zeros(1, 1, 2)

    output, prefix_sum, updated_residual = layer._post_attn_norm(
        attention_output, residual, old_prefix
    )

    assert call["prefix"] is attention_output
    assert call["delta"] is None
    assert call["num_blocks"] == 1
    assert call["block_write_idx"] == -1
    assert prefix_sum is attention_output
    assert updated_residual is residual
    torch.testing.assert_close(output, torch.tensor([[103.0, 104.0]]))


def _reference_attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    if delta is not None:
        prefix.add_(delta)
    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(prefix)
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if output_norm_weight is not None:
        output = F.rms_norm(
            output, (hidden_size,), output_norm_weight, output_norm_eps
        )
    return output


@pytest.mark.skipif(
    not current_platform.is_xpu(),
    reason="XPU AttnRes requires XPU",
)
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "block_capacity", "hidden_size"),
    [
        pytest.param(1, 1, 2, 128, id="decode-single"),
        pytest.param(17, 4, 6, 1024, id="decode-multiple-blocks"),
        pytest.param(320, 8, 10, 7168, id="prefill-full"),
    ],
)
def test_xpu_attn_res_matches_reference(
    num_tokens: int,
    num_blocks: int,
    block_capacity: int,
    hidden_size: int,
) -> None:
    eps = 1e-5
    device = torch.device("xpu")
    prefix = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    blocks = torch.randn(
        num_tokens,
        block_capacity,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device=device, dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    delta = torch.randn_like(prefix)
    output_norm_weight = 1 + 0.1 * torch.randn_like(norm_weight)
    block_write_idx = num_blocks
    expected_prefix = prefix.clone()
    expected_blocks = blocks.clone()
    expected = _reference_attn_res(
        expected_prefix,
        delta,
        expected_blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        eps,
    )

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        eps,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()