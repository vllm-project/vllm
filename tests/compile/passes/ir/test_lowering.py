# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect

import pytest
import torch
from torch import Tensor, nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir import register_op
from vllm.ir.op import IrOp
from vllm.platforms import current_platform

from ...backend import TestBackend
from .config import LoweringTestConfig


def assert_ops_lowered(
    lowering_pass: VllmIRLoweringPass, op: IrOp, provider: str | None, expected: int
):
    impls = list(lowering_pass.selected_impls.get(op.name, {}).values())
    lowered_count = impls.count(provider) if provider else len(impls)
    assert lowered_count == expected


def assert_total_ops_lowered(lowering_pass: VllmIRLoweringPass, expected: int):
    lowered_count = sum(
        [len(op_lowering) for op_lowering in lowering_pass.selected_impls.values()]
    )
    assert lowered_count == expected


# ==========================================
# UNIT TESTS
# ==========================================

FAKE_DEVICE_SUPPORTED = True


@register_op
def _fake_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    dtype: torch.dtype,
    variance_size: int | None = None,
) -> Tensor:
    return torch.zeros_like(x)


def _fake_device_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    dtype: torch.dtype,
    variance_size: int | None = None,
) -> Tensor:
    return torch.zeros_like(x)


@register_op
def _fake_rms_norm_1(
    x: Tensor,
    weight: Tensor | None,
    dtype: torch.dtype,
    variance_size: int | None = None,
) -> Tensor:
    return torch.zeros_like(x)


fake_device_rms_norm = _fake_rms_norm.register_impl(
    "fake_device",
    supports_args=lambda x, weight, dtype, variance_size=None: (
        FAKE_DEVICE_SUPPORTED
        and variance_size is None
        and dtype != torch.float8_e4m3fn
        and x.dtype == weight.dtype
    ),
)(_fake_device_rms_norm)

fake_device_rms_norm_1 = _fake_rms_norm_1.register_impl(
    "fake_device",
    supports_args=lambda x, weight, dtype, variance_size=None: (
        FAKE_DEVICE_SUPPORTED and variance_size is None
    ),
)(_fake_device_rms_norm)


def test_lowering(default_vllm_config):
    torch.set_default_device(current_platform.device_type)

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    x = torch.randn((12, 50), dtype=torch.bfloat16)
    weight = torch.randn((50,), dtype=torch.bfloat16)

    with ir.enable_torch_wrap(True):

        class CustomModel(nn.Module):
            def forward(
                self,
                x: Tensor,
                weight: Tensor | None,
                dtype: torch.dtype,
                variance_size: int | None = None,
            ) -> Tensor:
                y = _fake_rms_norm_1(x, weight, dtype, variance_size)
                return _fake_rms_norm(y, weight, dtype, variance_size)

        model = CustomModel()
        compiled_model = torch.compile(model, fullgraph=True, backend=backend)

        # check priority is followed
        with (
            _fake_rms_norm.set_priority(["fake_device", "native"]),
            _fake_rms_norm_1.set_priority(["native", "fake_device"]),
        ):
            torch._dynamo.reset()
            compiled_model(x, weight, torch.bfloat16)
            assert_ops_lowered(lowering_pass, _fake_rms_norm, "fake_device", 1)
            assert_ops_lowered(lowering_pass, _fake_rms_norm_1, "native", 1)
            assert_total_ops_lowered(lowering_pass, 2)

        with _fake_rms_norm.set_priority(["native", "fake_device"]):
            torch._dynamo.reset()
            compiled_model(x, weight, torch.bfloat16)
            assert_ops_lowered(lowering_pass, _fake_rms_norm, "native", 1)
            assert_total_ops_lowered(lowering_pass, 2)

        # check supports_args is respected
        with (
            _fake_rms_norm.set_priority(["fake_device", "native"]),
            _fake_rms_norm_1.set_priority(["fake_device", "native"]),
        ):
            torch._dynamo.reset()
            weight_f32 = weight.to(torch.float32)
            compiled_model(x, weight_f32, torch.bfloat16)
            assert_ops_lowered(lowering_pass, _fake_rms_norm, "native", 1)
            assert_ops_lowered(lowering_pass, _fake_rms_norm_1, "fake_device", 1)
            assert_total_ops_lowered(lowering_pass, 2)


# ==========================================
# PER OP TESTS
# ==========================================


@pytest.mark.parametrize(
    "op_name, provider, inputs, unbacked_idx",
    LoweringTestConfig.get_test_inputs(),
)
def test_per_op_lowering(
    default_vllm_config,
    op_name,
    provider,
    inputs,
    unbacked_idx,
):
    """
    test supports_all_args and implementation don't specialize on batch size
    of inputs
    """
    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)

    op = IrOp.registry[op_name]

    class OpImplModel(nn.Module):
        def forward(self, *args, **kwargs):
            return op(*args, **kwargs)

    model = OpImplModel()
    compiled_model = torch.compile(model, backend=backend, fullgraph=True)
    input = inputs[0]

    # ensure op compiles without break
    with ir.enable_torch_wrap(True), op.set_priority([provider]):
        torch._dynamo.reset()
        compiled_model(*input)
        assert_ops_lowered(lowering_pass, op, provider, 1)

    native_signature = inspect.signature(op.impls["native"].impl_fn)
    arg_pos = {
        name: idx for idx, name in enumerate(native_signature.parameters.keys())
    }

    with ir.enable_torch_wrap(True), op.set_priority([provider]):
        torch._dynamo.reset()
        for name, indices in unbacked_idx.items():
            assert name in arg_pos, "invalid arg name"
            for idx in indices:
                assert isinstance(input[arg_pos[name]], torch.Tensor)
                torch._dynamo.decorators.mark_unbacked(input[arg_pos[name]], idx)
        compiled_model(*input)
