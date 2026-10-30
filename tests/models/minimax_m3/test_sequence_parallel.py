# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.minimax_m3.nvidia import model as m3_model


class _IdentityNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingAttention(nn.Module):
    """Stands in for attention + o_proj reduce-scatter at TP2."""

    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return torch.nn.functional.pad(hidden_states, (0, 0, 0, 1))[:2]


class _RecordingMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def defers_finalize(self, num_tokens: int) -> bool:
        return False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


def _vllm_config(moe_backend: str, tp: int, pp: int, ep: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
        ),
    )


@pytest.mark.parametrize(
    ("moe_backend", "tp", "pp", "ep", "expected"),
    [
        ("deep_gemm_mega_moe", 4, 1, True, True),
        ("deep_gemm_mega_moe", 1, 1, True, False),
        ("deep_gemm_mega_moe", 4, 2, True, False),
        ("deep_gemm_mega_moe", 4, 1, False, False),
        ("auto", 4, 1, True, False),
    ],
)
def test_sequence_parallel_requires_mega_moe_tp_ep_without_pp(
    moe_backend: str, tp: int, pp: int, ep: bool, expected: bool
):
    vllm_config = _vllm_config(moe_backend, tp, pp, ep)
    assert m3_model._use_sequence_parallel(vllm_config) is expected


def test_decoder_layer_keeps_residual_stream_sequence_sharded(monkeypatch):
    # TP2: the all-gather doubles the local shard.
    monkeypatch.setattr(
        m3_model, "sp_all_gather", lambda tensor: torch.cat([tensor, tensor])
    )
    layer = object.__new__(m3_model.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.fuse_input_allreduce = False
    layer.is_moe_layer = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingAttention()
    layer.block_sparse_moe = _RecordingMoE()

    positions = torch.arange(3)
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    residual = None
    for _ in range(2):
        hidden_states, residual = layer(positions, hidden_states, residual)
        # Attention sees every token; the residual stream and MoE stay sharded.
        assert layer.self_attn.num_tokens == 3
        assert layer.block_sparse_moe.num_tokens == 2
        assert hidden_states.shape == residual.shape == (2, 2)

    # No all-reduce is pending for the next layer's input norm to fuse.
    assert not layer.ffn_all_reduce_deferred
