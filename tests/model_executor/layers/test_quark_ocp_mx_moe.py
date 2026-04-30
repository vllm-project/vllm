# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.quantization.quark import quark_moe


def _make_method() -> quark_moe.QuarkOCP_MX_MoEMethod:
    method = object.__new__(quark_moe.QuarkOCP_MX_MoEMethod)
    method.moe_kernel = None
    method.emulate = False
    method.moe_quant_config = object()
    return method


def _make_layer(apply_router_weight_on_input: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        w13_weight=torch.randn(2, 4, 4),
        w2_weight=torch.randn(2, 4, 4),
        activation=quark_moe.MoEActivation.SILU,
        global_num_experts=2,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=None,
        moe_config=SimpleNamespace(),
    )


def test_quark_ocp_mx_moe_aiter_apply_forwards_router_weight_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    method = _make_method()
    layer = _make_layer()
    x = torch.randn(3, 4)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    expected = torch.randn(3, 4)

    aiter_mock = MagicMock(return_value=expected)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts",  # noqa: E501
        aiter_mock,
    )

    result = method.apply(layer, x, topk_weights, topk_ids, shared_experts_input=None)

    assert result is expected
    aiter_mock.assert_called_once()
    assert aiter_mock.call_args.args == (
        x,
        layer.w13_weight,
        layer.w2_weight,
    )
    assert aiter_mock.call_args.kwargs == {
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": layer.activation,
        "apply_router_weight_on_input": layer.apply_router_weight_on_input,
        "quant_config": method.moe_quant_config,
        "moe_config": layer.moe_config,
        "expert_map": layer.expert_map,
    }


def test_quark_ocp_mx_moe_does_not_runtime_fallback_after_aiter_error(
    monkeypatch: pytest.MonkeyPatch,
):
    method = _make_method()
    layer = _make_layer()
    x = torch.randn(3, 4)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)

    aiter_mock = MagicMock(
        side_effect=RuntimeError("Unsupported kernel config for moe heuristic dispatch")
    )
    fused_mock = MagicMock()

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts",  # noqa: E501
        aiter_mock,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_experts",
        fused_mock,
    )

    with pytest.raises(RuntimeError, match="Unsupported kernel config"):
        method.apply(layer, x, topk_weights, topk_ids, shared_experts_input=None)

    assert method.emulate is False
    aiter_mock.assert_called_once()
    fused_mock.assert_not_called()
