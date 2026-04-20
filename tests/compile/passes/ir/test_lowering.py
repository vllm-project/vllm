# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import Tensor, nn
from torch.testing import assert_close

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


def assert_ops_lowered(
    lowering_pass: VllmIRLoweringPass, op: IrOp, provider: str | None, expected: int
):
    __tracebackhide__ = True
    lowered_count = lowering_pass.recorder.lowered_count(op.name, provider)
    if lowered_count != expected:
        pytest.fail(
            f"Expected {expected} calls to be lowered to {op.name}[{provider}]"
            f", found {lowered_count} instead"
        )


def assert_total_ops_lowered(lowering_pass: VllmIRLoweringPass, expected: int):
    __tracebackhide__ = True
    lowered_count = lowering_pass.recorder.total_lowered()
    if lowered_count != expected:
        pytest.fail(
            f"Expected {expected} calls to be lowered, found {lowered_count} instead"
        )


@register_op
def _custom_add_op(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> Tensor:
    return a + b


@_custom_add_op.register_impl(
    "mask_a", supports_args=lambda a, b, mask, mask_a, mask_b: mask_a
)
def _custom_add_int16(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> Tensor:
    return a.masked_fill(mask, 0) + b


@_custom_add_op.register_impl(
    "mask_b", supports_args=lambda a, b, mask, mask_a, mask_b: mask_b
)
def _custom_add_int64(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> Tensor:
    return a + b.masked_fill(mask, 0)


@_custom_add_op.register_impl(
    "mask_a_and_b", supports_args=lambda a, b, mask, mask_a, mask_b: mask_a and mask_b
)
def _custom_add_int64(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> Tensor:
    return a.masked_fill(mask, 0) + b.masked_fill(mask, 0)


@register_op
def _custom_sub_op(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> torch.Tensor:
    return a - b


@_custom_sub_op.register_impl(
    "mask_a", supports_args=lambda a, b, mask, mask_a, mask_b: mask_a
)
def _custom_sub_int16(
    a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
) -> torch.Tensor:
    return a.masked_fill(mask, 0) - b


class TestOpLoweringPass:
    def test_lowering_single_impl(self, default_vllm_config):
        torch.set_default_device(current_platform.device_type)

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        a = torch.full((4,), 1)
        b = torch.full((4,), 2)
        mask = torch.tensor([1, 1, 0, 0]) >= 1

        with ir.enable_torch_wrap(True):

            class CustomAddModel(nn.Module):
                def forward(
                    self, a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
                ) -> Tensor:
                    return _custom_add_op(a, b, mask, mask_a, mask_b)

            model = CustomAddModel()
            compiled_model = torch.compile(model, fullgraph=True, backend=backend)

            with _custom_add_op.set_priority(["mask_a", "mask_a_and_b", "native"]):
                torch._dynamo.reset()
                out = compiled_model(a, b, mask, True, True)
                # ensure op is lowered to correct impl
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 1)
                # ensure correct code got executed
                assert_close((a.masked_fill(mask, 0) + b), out)
                assert_total_ops_lowered(lowering_pass, 1)

            with _custom_add_op.set_priority(["mask_a_and_b", "mask_a", "native"]):
                torch._dynamo.reset()
                out = compiled_model(a, b, mask, True, True)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a_and_b", 1)
                assert_close((a.masked_fill(mask, 0) + b.masked_fill(mask, 0)), out)
                assert_total_ops_lowered(lowering_pass, 1)

    def test_lowering_multiple_impl(self, default_vllm_config):
        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        a = torch.full((4,), 1)
        b = torch.full((4,), 2)
        mask = torch.tensor([1, 1, 0, 0]) >= 1

        with ir.enable_torch_wrap(True):

            class CustomAddModel2(nn.Module):
                def forward(
                    self, a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
                ) -> Tensor:
                    x = _custom_add_op(a, b, mask, False, True)
                    y = _custom_add_op(a, b, mask, True, False)
                    return _custom_add_op(x, y, mask, mask_a, mask_b)

            model = CustomAddModel2()
            compiled_model = torch.compile(model, fullgraph=True, backend=backend)

            with _custom_add_op.set_priority(
                ["mask_a", "mask_a_and_b", "mask_b", "native"]
            ):
                torch._dynamo.reset()
                compiled_model(a, b, mask, True, True)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 2)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_b", 1)
                assert_total_ops_lowered(lowering_pass, 3)

            with _custom_add_op.set_priority(
                ["mask_a_and_b", "mask_a", "mask_b", "native"]
            ):
                torch._dynamo.reset()
                compiled_model(a, b, mask, True, True)

                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a_and_b", 1)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 1)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_b", 1)
                assert_total_ops_lowered(lowering_pass, 3)

                compiled_model(a, b, mask, True, False)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 2)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_b", 1)
                assert_total_ops_lowered(lowering_pass, 3)

    def test_lowering_multiple_ops(self, default_vllm_config):
        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        a = torch.full((4,), 1)
        b = torch.full((4,), 2)
        mask = torch.tensor([1, 1, 0, 0]) >= 1

        class CustomModel(nn.Module):
            def forward(
                self, a: Tensor, b: Tensor, mask: Tensor, mask_a: bool, mask_b: bool
            ) -> Tensor:
                x = _custom_add_op(a, b, mask, True, False)
                y = _custom_add_op(a, b, mask, mask_a, mask_b)
                return _custom_sub_op(x, y, mask, mask_a, mask_b)

        with ir.enable_torch_wrap(True):
            model = CustomModel()
            compiled_model = torch.compile(model, fullgraph=True, backend=backend)
            with (
                _custom_add_op.set_priority(
                    ["mask_a", "mask_a_and_b", "mask_b", "native"]
                ),
                _custom_sub_op.set_priority(["mask_a"]),
            ):
                torch._dynamo.reset()
                compiled_model(a, b, mask, True, True)

                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 2)
                assert_ops_lowered(lowering_pass, _custom_sub_op, "mask_a", 1)
                assert_total_ops_lowered(lowering_pass, 3)

            with (
                _custom_add_op.set_priority(
                    ["mask_a", "mask_a_and_b", "mask_b", "native"]
                ),
                _custom_sub_op.set_priority(["mask_a"]),
            ):
                torch._dynamo.reset()
                compiled_model(a, b, mask, False, True)

                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_a", 1)
                assert_ops_lowered(lowering_pass, _custom_add_op, "mask_b", 1)
                assert_ops_lowered(lowering_pass, _custom_sub_op, "native", 1)
                assert_total_ops_lowered(lowering_pass, 3)


# cartesian product of op x op_impls
OpImpls = [
    (op.name, op_impl.provider)
    for op in IrOp.registry.values()
    for op_impl in op.impls.values()
]

TEST_INPUTS: dict[str, list[list]] = {
    "rms_norm": [[torch.randn(16), torch.randn(16), 1e-5, None]]
}


@pytest.mark.parametrize("op_name, provider", OpImpls)
def test_op_impl(op_name, provider, default_vllm_config):
    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)
    unlowered_backend = TestBackend()
    torch._dynamo.reset()

    op = IrOp.registry[op_name]
    op_impl = op.impls[provider]

    if not op_impl.supported:
        pytest.skip("Implementation is marked as not supported")

    test_inputs = TEST_INPUTS.get(op_name)
    if not test_inputs:
        pytest.skip("No test inputs provider for the op")

    def _filter_test_inputs(inputs: list[list]) -> list[list]:
        supported_inputs = []
        for input in inputs:
            if op_impl.supports_all_args or op_impl.supports_args(*input):
                supported_inputs.append(input)
        return supported_inputs

    # filter out inputs that the op provider doesnt support
    test_inputs = _filter_test_inputs(test_inputs)
    if not test_inputs:
        pytest.skip("Provider doesnt support any of the test inputs")

    class OpImplModel(nn.Module):
        def forward(self, *args, **kwargs):
            return op(*args, **kwargs)

    model = OpImplModel()
    compiled_model = torch.compile(model, backend=backend, fullgraph=True)
    unlowered_model = torch.compile(model, backend=unlowered_backend, fullgraph=True)
    # ensure implementation is lowered correctly
    with ir.enable_torch_wrap(True), op.set_priority([provider]):
        lowered_output = compiled_model(*test_inputs[0])
        assert_ops_lowered(lowering_pass, op, provider, 1)
        torch._dynamo.reset()

    # ensure supports_all_args and implementation is traceable by dynamo
    with ir.enable_torch_wrap(False), op.set_priority([provider]):
        unwrapped_output = compiled_model(*test_inputs[0])
        torch._dynamo.reset()

    with ir.enable_torch_wrap(True), op.set_priority([provider]):
        unlowered_output = unlowered_model(*test_inputs[0])

    torch.testing.assert_close(lowered_output, unwrapped_output)
    torch.testing.assert_close(lowered_output, unlowered_output)
