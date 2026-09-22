# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash sequence-parallel MoE layout.

With DP > 1, TP > 1 and expert parallelism, ``Glm5NextModel`` shards the token
dimension across the TP group once at the model entry and every layer runs its
MLP on that shard. A module that still does tensor-parallel collectives there
sums the partial results of *different* tokens held by different ranks, so the
dense MLP of the first ``first_k_dense_replace`` layers must hold replicated
weights and run without any collective, like the shared experts already do.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.models.glm5next.common import model as glm_model
from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

HIDDEN, INTERMEDIATE, TP_SIZE = 8, 16, 2


class _Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)


def _fake_tensor_parallel_world(monkeypatch) -> None:
    """Pretend to be rank 1 of a TP=2 group whose all-reduce must never run."""

    def _forbidden_all_reduce(*args, **kwargs):
        raise AssertionError("tensor-parallel all-reduce on a sequence-parallel shard")

    for module in (linear, parameter):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
        )
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        linear, "tensor_model_parallel_all_reduce", _forbidden_all_reduce
    )


def _reference_mlp(mlp: glm_model.Glm5NextMLP, x: torch.Tensor) -> torch.Tensor:
    gate_up = x @ mlp.gate_up_proj.weight.T
    gate, up = gate_up.split(INTERMEDIATE, dim=-1)
    return (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.T


def _assert_runs_locally_on_shard(mlp: glm_model.Glm5NextMLP) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp.to(device)
    assert mlp.gate_up_proj.weight.shape == (2 * INTERMEDIATE, HIDDEN)
    assert mlp.down_proj.weight.shape == (HIDDEN, INTERMEDIATE)
    torch.manual_seed(0)
    for param in mlp.parameters():
        param.data.normal_()
    shard = torch.randn(3, HIDDEN, device=device)
    torch.testing.assert_close(mlp(shard), _reference_mlp(mlp, shard))


def test_dense_mlp_runs_locally_on_the_sequence_parallel_shard(
    monkeypatch, default_vllm_config
):
    _fake_tensor_parallel_world(monkeypatch)

    sharded = glm_model.Glm5NextMLP(HIDDEN, INTERMEDIATE, "silu")
    assert sharded.down_proj.tp_size == TP_SIZE
    assert sharded.gate_up_proj.weight.shape == (2 * INTERMEDIATE // TP_SIZE, HIDDEN)

    _assert_runs_locally_on_shard(
        glm_model.Glm5NextMLP(HIDDEN, INTERMEDIATE, "silu", is_sequence_parallel=True)
    )


@pytest.mark.parametrize("use_sequence_parallel_moe", [True, False])
def test_dense_layer_mlp_follows_the_sequence_parallel_layout(
    monkeypatch, default_vllm_config, use_sequence_parallel_moe: bool
):
    _fake_tensor_parallel_world(monkeypatch)
    monkeypatch.setattr(glm_model, "Glm5NextLinearAttention", _Attention)
    monkeypatch.setattr(glm_model, "Glm5NextMLAAttention", _Attention)

    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mhc=False,
    )
    vllm_config = SimpleNamespace(
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=use_sequence_parallel_moe
        ),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
    )

    layer = glm_model.Glm5NextDecoderLayer(vllm_config, config, layer_idx=0)

    assert isinstance(layer.mlp, glm_model.Glm5NextMLP)
    assert layer.self_attn.o_proj.reduce_results is not use_sequence_parallel_moe
    if use_sequence_parallel_moe:
        _assert_runs_locally_on_shard(layer.mlp)
    else:
        assert layer.mlp.down_proj.tp_size == TP_SIZE
