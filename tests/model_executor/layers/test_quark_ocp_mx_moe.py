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
    method.use_rocm_aiter_moe = True
    method.moe_quant_config = object()
    method.moe = SimpleNamespace(disable_inplace=False)
    method.ocp_mx_scheme = "w_mxfp4_a_fp8"
    return method


def _make_layer() -> SimpleNamespace:
    return SimpleNamespace(
        w13_weight=torch.randn(2, 4, 4),
        w2_weight=torch.randn(2, 4, 4),
        activation=quark_moe.MoEActivation.SILU,
        global_num_experts=2,
        apply_router_weight_on_input=False,
        expert_map=None,
        moe_config=SimpleNamespace(),
    )


def test_quark_ocp_mx_moe_falls_back_for_unsupported_aiter_dispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    method = _make_method()
    layer = _make_layer()
    x = torch.randn(3, 4)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    expected = torch.randn(3, 4)

    aiter_mock = MagicMock(
        side_effect=RuntimeError("Unsupported kernel config for moe heuristic dispatch")
    )
    fused_mock = MagicMock(return_value=expected)
    warning_mock = MagicMock()

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts",  # noqa: E501
        aiter_mock,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_experts",
        fused_mock,
    )
    monkeypatch.setattr(quark_moe.logger, "warning_once", warning_mock)

    result = method.apply(layer, x, topk_weights, topk_ids, shared_experts_input=None)

    assert result is expected
    assert method.emulate is True
    assert method.use_rocm_aiter_moe is False
    aiter_mock.assert_called_once()
    fused_mock.assert_called_once()
    assert fused_mock.call_args.args == (
        x,
        layer.w13_weight,
        layer.w2_weight,
    )
    assert fused_mock.call_args.kwargs == {
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "inplace": True,
        "activation": layer.activation,
        "global_num_experts": layer.global_num_experts,
        "apply_router_weight_on_input": layer.apply_router_weight_on_input,
        "expert_map": layer.expert_map,
        "quant_config": method.moe_quant_config,
    }
    warning_mock.assert_called_once()
    assert "Unsupported kernel config for moe heuristic dispatch" in str(
        warning_mock.call_args
    )


def test_quark_ocp_mx_moe_preserves_unrelated_aiter_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    method = _make_method()
    layer = _make_layer()
    x = torch.randn(3, 4)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32)

    aiter_mock = MagicMock(side_effect=RuntimeError("different aiter failure"))
    fused_mock = MagicMock()

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts",  # noqa: E501
        aiter_mock,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.fused_experts",
        fused_mock,
    )

    with pytest.raises(RuntimeError, match="different aiter failure"):
        method.apply(layer, x, topk_weights, topk_ids, shared_experts_input=None)

    assert method.emulate is False
    assert method.use_rocm_aiter_moe is True
    aiter_mock.assert_called_once()
    fused_mock.assert_not_called()
