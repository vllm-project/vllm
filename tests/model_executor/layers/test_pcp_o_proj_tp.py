# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import PCPOProjRowParallelLinear


class _FakeWork:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _FakePCPGroup:
    def __init__(self, rank: int) -> None:
        self.rank_in_group = rank
        self.world_size = 2
        self.device_group = object()
        self.other_output: torch.Tensor | None = None

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self.other_output is not None
        return tensor + self.other_output


def _make_layer(
    monkeypatch, pcp_rank: int
) -> tuple[PCPOProjRowParallelLinear, _FakePCPGroup]:
    pcp_group = _FakePCPGroup(pcp_rank)
    config_scope = object()
    monkeypatch.setattr(linear_module, "get_pcp_group", lambda: pcp_group)
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(linear_module, "get_current_vllm_config", lambda: config_scope)
    monkeypatch.setattr(
        linear_module.UnquantizedLinearMethod,
        "apply",
        lambda _self, layer, x, bias=None: F.linear(x, layer.weight, bias),
    )
    layer = PCPOProjRowParallelLinear(
        input_size=4,
        output_size=3,
        params_dtype=torch.float32,
        reduce_results=False,
    )
    return layer, pcp_group


def test_prefill_waits_for_async_full_weight_gather(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=1)
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)
    torch.testing.assert_close(layer.weight, full_weight[:, 2:])

    work = _FakeWork()

    def _all_gather_into_tensor(output, input_, group, async_op):
        torch.testing.assert_close(input_, full_weight[:, 2:].transpose(0, 1))
        output.copy_(full_weight.transpose(0, 1))
        assert group is layer.pcp_group.device_group
        assert async_op
        return work

    monkeypatch.setattr(
        torch.distributed, "all_gather_into_tensor", _all_gather_into_tensor
    )

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    layer.prefetch_full_weight_if_needed(has_prefill=True)
    assert not work.waited

    output, output_bias = layer(input_)
    assert work.waited
    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))


def test_forward_requires_explicit_attention_prefetch(monkeypatch):
    layer, _ = _make_layer(monkeypatch, pcp_rank=0)

    with pytest.raises(RuntimeError, match="Attention must call"):
        layer(torch.zeros(1, 4))


def test_decode_slices_features_and_reduces_over_pcp(monkeypatch):
    layer, pcp_group = _make_layer(monkeypatch, pcp_rank=1)
    full_weight = torch.arange(12, dtype=torch.float32).view(3, 4)
    layer.weight_loader_v2(layer.weight, full_weight)

    input_ = torch.arange(8, dtype=torch.float32).view(2, 4)
    pcp_group.other_output = F.linear(input_[:, :2], full_weight[:, :2])

    layer.prefetch_full_weight_if_needed(has_prefill=False)
    output, output_bias = layer(input_)

    assert output_bias is None
    torch.testing.assert_close(output, F.linear(input_, full_weight))
