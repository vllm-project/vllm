# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import mtp as kimi_mtp
from vllm.models.kimi_k3.nvidia.ops import sequence_parallel as sp_ops
from vllm.platforms import current_platform


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _Projection(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, hidden_size),
            requires_grad=False,
        )


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, None, hidden_states * 3


def _mock_sequence_parallel_collectives(monkeypatch):
    monkeypatch.setattr(
        kimi_model,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


@pytest.mark.parametrize(
    ("num_tokens", "is_padding", "tp_rank", "expected"),
    [
        (1, None, 0, [False]),
        (1, None, 1, [True]),
        (5, None, 2, [False, True]),
        (5, None, 3, [True, True]),
        (5, [False, True, False, False, False], 0, [False, True]),
    ],
)
def test_sp_padding_mask_marks_added_rows(
    monkeypatch,
    num_tokens: int,
    is_padding: list[bool] | None,
    tp_rank: int,
    expected: list[bool],
):
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: tp_rank)

    hidden_states = torch.empty(num_tokens, 2)
    padding = torch.tensor(is_padding) if is_padding is not None else None
    actual = sp_ops.sp_padding_mask(padding, hidden_states)

    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    ("data_parallel_size", "expected"),
    [
        (1, False),
        (2, True),
    ],
)
def test_moe_sequence_parallel_requires_data_parallel(
    monkeypatch,
    data_parallel_size: int,
    expected: bool,
):
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    assert parallel_config.use_sequence_parallel_moe is expected


def test_kimi_decoder_layer_keeps_moe_states_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = False
    layer.use_sequence_parallel = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = kimi_model.sp_shard(full_hidden_states)
    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2

    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_attn_residual_states_stay_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.use_sequence_parallel = True
    layer.prev_valid_blocks = 0
    layer.block_write_idx = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attention_res_norm = _IdentityNorm()
    layer.mlp_res_norm = _IdentityNorm()
    layer.self_attention_res_proj = _Projection()
    layer.mlp_res_proj = _Projection()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)
    monkeypatch.setattr(
        kimi_model,
        "attn_res",
        lambda prefix_sum, hidden_states, *args, **kwargs: (
            prefix_sum if hidden_states is None else prefix_sum + hidden_states
        ),
    )

    prefix_sum = kimi_model.sp_shard(torch.arange(6, dtype=torch.float32).view(3, 2))
    block_residual = torch.zeros(2, 1, 2)
    hidden_states, prefix_sum, block_residual = layer(
        positions=torch.arange(3),
        hidden_states=None,
        prefix_sum=prefix_sum,
        residual=block_residual,
    )

    assert hidden_states.shape == prefix_sum.shape == (2, 2)
    assert block_residual.shape == (2, 1, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_mtp_restores_sequence_parallel_output(monkeypatch):
    layer = object.__new__(kimi_mtp.KimiK3MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = nn.Identity()
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())

    final_norm = Mock(side_effect=lambda hidden_states: hidden_states + 1)
    object.__setattr__(
        layer,
        "shared_head",
        SimpleNamespace(norm=final_norm),
    )

    monkeypatch.setattr(
        kimi_mtp,
        "fused_mtp_input",
        lambda positions, inputs_embeds, *args: inputs_embeds,
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    logits_hidden_states, hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected_hidden_states = torch.cat(
        [sharded_states * 5, sharded_states * 5],
        dim=0,
    )[:3]
    torch.testing.assert_close(hidden_states, expected_hidden_states)
    torch.testing.assert_close(logits_hidden_states, expected_hidden_states + 1)
    final_norm.assert_called_once()
    torch.testing.assert_close(final_norm.call_args.args[0], expected_hidden_states)


def test_sp_all_gather_uses_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    expected = torch.cat([hidden_states, hidden_states])
    custom_all_gather = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_all_gather=custom_all_gather,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", fallback)

    output = sp_ops.sp_all_gather(hidden_states)

    torch.testing.assert_close(output, expected)
    custom_all_gather.assert_called_once_with(hidden_states)
    fallback.assert_not_called()


def test_sp_reduce_scatter_uses_custom_kernel_after_padding(monkeypatch):
    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    expected = torch.arange(4, dtype=torch.float32).view(2, 2)
    custom_reduce_scatter = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_reduce_scatter=custom_reduce_scatter,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_reduce_scatter", fallback)

    output = sp_ops.sp_reduce_scatter(hidden_states)

    torch.testing.assert_close(output, expected)
    padded = custom_reduce_scatter.call_args.args[0]
    assert padded.shape == (4, 2)
    torch.testing.assert_close(padded[:3], hidden_states)
    torch.testing.assert_close(padded[3], torch.zeros(2))
    fallback.assert_not_called()


def test_sp_collectives_fall_back_without_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=None),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    all_gather = Mock(return_value=hidden_states)
    reduce_scatter = Mock(return_value=hidden_states)
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(
        sp_ops,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )

    torch.testing.assert_close(sp_ops.sp_all_gather(hidden_states), hidden_states)
    torch.testing.assert_close(sp_ops.sp_reduce_scatter(hidden_states), hidden_states)
    all_gather.assert_called_once_with(hidden_states, 0)
    reduce_scatter.assert_called_once_with(hidden_states, 0)
