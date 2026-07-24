# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models import gemma4 as gemma4_mod


class _FakeColumnParallelLinear(nn.Module):

    def __init__(
        self,
        _hidden_size: int,
        output_size: int,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        shape = (*x.shape[:-1], self.output_size)
        return torch.zeros(shape, dtype=x.dtype, device=x.device), None


class _FakeQKVParallelLinear(nn.Module):
    forward_calls = 0

    def __init__(
        self,
        _hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.output_size = (
            total_num_heads * head_size
            + 2 * total_num_kv_heads * head_size
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        type(self).forward_calls += 1
        shape = (*x.shape[:-1], self.output_size)
        return torch.zeros(shape, dtype=x.dtype, device=x.device), None


class _FakeRowParallelLinear(nn.Module):

    def __init__(self, _input_size: int, _output_size: int, **_kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


class _FakeAttention(nn.Module):

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.last_shapes: tuple[torch.Size, torch.Size, torch.Size] | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        self.last_shapes = (q.shape, k.shape, v.shape)
        return q


def test_gemma4_k_eq_v_uses_separate_qk_proj(monkeypatch):
    monkeypatch.setattr(gemma4_mod, "ColumnParallelLinear", _FakeColumnParallelLinear)
    monkeypatch.setattr(gemma4_mod, "QKVParallelLinear", _FakeQKVParallelLinear)
    monkeypatch.setattr(gemma4_mod, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(gemma4_mod, "Attention", _FakeAttention)
    monkeypatch.setattr(
        gemma4_mod,
        "get_rope",
        lambda *args, **kwargs: (lambda positions, q, k: (q, k)),
    )
    monkeypatch.setattr(gemma4_mod, "extract_layer_index", lambda _prefix: 5)

    config = SimpleNamespace(
        attention_bias=False,
        rms_norm_eps=1e-6,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=1024,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
        num_hidden_layers=6,
        num_kv_shared_layers=0,
    )

    _FakeQKVParallelLinear.forward_calls = 0
    attn = gemma4_mod.Gemma4Attention(
        config=config,
        hidden_size=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=16,
        max_position_embeddings=128,
        use_k_eq_v=True,
        prefix="model.layers.5.self_attn",
    )

    assert attn.qkv_proj is None
    assert hasattr(attn, "q_proj")
    assert hasattr(attn, "k_proj")

    positions = torch.arange(3)
    hidden_states = torch.zeros(3, 32)
    output = attn(positions, hidden_states)

    assert _FakeQKVParallelLinear.forward_calls == 0
    assert output.shape == (3, 64)
    assert attn.attn.last_shapes == (
        torch.Size([3, 64]),
        torch.Size([3, 16]),
        torch.Size([3, 16]),
    )
