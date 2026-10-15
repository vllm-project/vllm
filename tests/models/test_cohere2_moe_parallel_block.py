# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Cohere2 MoE parallel block (attention || MLP).

Covers the single TP all-reduce of attention + MLP. Decoder layers are built
without ``__init__`` around stub branches, so the tests need no checkpoint or
GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import cohere2_moe


class _PassthroughNorm(torch.nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, residual


class _Branch(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.calls = 0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.output


def _make_decoder(
    *,
    attention: torch.nn.Module,
    mlp: torch.nn.Module,
    mlp_output_is_reduced: bool = False,
) -> cohere2_moe.Cohere2MoeDecoderLayer:
    layer = cohere2_moe.Cohere2MoeDecoderLayer.__new__(
        cohere2_moe.Cohere2MoeDecoderLayer
    )
    torch.nn.Module.__init__(layer)
    layer.input_layernorm = _PassthroughNorm()
    layer.self_attn = attention
    layer.mlp = mlp
    layer.mlp_output_is_reduced = mlp_output_is_reduced
    return layer


def _record_all_reduce(monkeypatch, tp_size: int = 4) -> list[torch.Tensor]:
    """Replace the TP all-reduce with a sum over ``tp_size`` identical ranks."""
    inputs: list[torch.Tensor] = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        inputs.append(tensor)
        return tensor * tp_size

    monkeypatch.setattr(cohere2_moe, "tensor_model_parallel_all_reduce", all_reduce)
    return inputs


@pytest.mark.parametrize(
    ("num_shared_experts", "tp_size", "skip_final_all_reduce", "expected"),
    [
        pytest.param(0, 4, True, (False, False), id="tp4-routed-only"),
        pytest.param(0, 4, False, (False, True), id="tp4-routed-only-all2all"),
        pytest.param(1, 4, False, (True, True), id="tp4-shared-experts"),
        pytest.param(0, 1, False, (True, True), id="tp1"),
    ],
)
def test_moe_defers_all_reduce_only_without_shared_experts(
    monkeypatch,
    num_shared_experts: int,
    tp_size: int,
    skip_final_all_reduce: bool,
    expected: tuple[bool, bool],
) -> None:
    """Only a routed-only TP MoE asks FusedMoE to skip its all-reduce, and it
    stays reduced when the runner resolves the request away (e.g. all2all)."""
    factory_kwargs: dict[str, object] = {}

    def make_experts(**kwargs):
        factory_kwargs.update(kwargs)
        return SimpleNamespace(
            moe_config=SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce)
        )

    monkeypatch.setattr(
        cohere2_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        cohere2_moe, "GateLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        cohere2_moe, "Cohere2MoeMLP", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(cohere2_moe, "FusedMoEFactory", make_experts)
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=1,
        intermediate_size=4,
        num_shared_experts=num_shared_experts,
    )

    moe = cohere2_moe.Cohere2Moe(config)

    assert (factory_kwargs["reduce_results"], moe.output_is_reduced) == expected


@pytest.mark.parametrize("mlp_output_is_reduced", [False, True])
def test_parallel_block_reduces_attention_plus_mlp_once(
    monkeypatch, mlp_output_is_reduced: bool
) -> None:
    """Attention + MLP share one all-reduce unless the MLP output is already
    reduced, in which case only attention is reduced."""
    hidden_states = torch.ones(2, 4)
    attention_output = torch.full_like(hidden_states, 2)
    mlp_output = torch.full_like(hidden_states, 3)
    layer = _make_decoder(
        attention=_Branch(attention_output),
        mlp=_Branch(mlp_output),
        mlp_output_is_reduced=mlp_output_is_reduced,
    )
    all_reduce_inputs = _record_all_reduce(monkeypatch)

    output, residual = layer(
        positions=torch.arange(hidden_states.shape[0]),
        hidden_states=hidden_states,
        residual=None,
    )

    assert len(all_reduce_inputs) == 1
    if mlp_output_is_reduced:
        torch.testing.assert_close(all_reduce_inputs[0], attention_output)
        expected_block = 4 * attention_output + mlp_output
    else:
        torch.testing.assert_close(all_reduce_inputs[0], attention_output + mlp_output)
        expected_block = 4 * (attention_output + mlp_output)
    torch.testing.assert_close(output, hidden_states + expected_block)
    torch.testing.assert_close(residual, hidden_states)
