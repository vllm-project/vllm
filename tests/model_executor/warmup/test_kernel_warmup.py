# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.warmup import kernel_warmup as warmup_module
from vllm.models.inkling.nvidia.moe import InklingGate


class _WarmupModule(torch.nn.Module):
    def __init__(self, shapes: tuple[tuple[int, int], ...]) -> None:
        super().__init__()
        self.shapes = shapes

    def _get_ll_bf16_warmup_shapes(self) -> tuple[tuple[int, int], ...]:
        return self.shapes


def test_ll_bf16_warmup_skips_models_without_router(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as ll_bf16

    is_available = Mock(return_value=True)
    warmup = Mock()
    monkeypatch.setattr(ll_bf16, "is_available", is_available)
    monkeypatch.setattr(ll_bf16.ll_bf16_gemm_kernel, "warmup", warmup)

    warmup_module._warmup_ll_bf16_router_gemm(torch.nn.Linear(8, 4))

    is_available.assert_not_called()
    warmup.assert_not_called()


def test_ll_bf16_warmup_uses_unique_model_shapes(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as ll_bf16

    model = torch.nn.Sequential(
        _WarmupModule(((32, 8),)),
        _WarmupModule(((16, 4), (32, 8))),
    )
    monkeypatch.setattr(ll_bf16, "is_available", lambda: True)
    warmup = Mock()
    monkeypatch.setattr(ll_bf16.ll_bf16_gemm_kernel, "warmup", warmup)

    warmup_module._warmup_ll_bf16_router_gemm(model)

    warmup.assert_called_once_with(
        shapes=((16, 4), (32, 8)),
        m_values=warmup_module._LL_BF16_WARMUP_M_RANGE,
    )


def test_gate_linear_reports_enabled_ll_bf16_shape() -> None:
    gate = SimpleNamespace(
        allow_ll_bf16_gemm=True,
        weight=torch.empty(8, 16, dtype=torch.bfloat16),
    )

    assert GateLinear._get_ll_bf16_warmup_shapes(gate) == ((16, 8),)

    gate.allow_ll_bf16_gemm = False
    assert GateLinear._get_ll_bf16_warmup_shapes(gate) == ()


def test_custom_gate_reports_only_bf16_shape() -> None:
    gate = SimpleNamespace(weight=torch.empty(8, 16, dtype=torch.bfloat16))
    assert InklingGate._get_ll_bf16_warmup_shapes(gate) == ((16, 8),)

    gate.weight = gate.weight.float()
    assert InklingGate._get_ll_bf16_warmup_shapes(gate) == ()
