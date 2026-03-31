# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.ir.op
from vllm.ir.op import IrOp, IrOpImpl, IrOpInplaceOverload


@vllm.ir.register_op(allow_inplace=True)
def _custom_mm2(x: Tensor, w: Tensor) -> Tensor:
    return x @ w


@_custom_mm2.register_impl("regular")
def _custom_mm2_regular(x: Tensor, w: Tensor) -> Tensor:
    return x @ w + 1


@_custom_mm2.register_impl("inplace", inplace=True)
def _custom_mm2_inplace(x: Tensor, w: Tensor) -> Tensor:
    x.copy_(x @ w + 2)
    return x


class TestInplaceOp:
    def test_registration(self):
        # Test that the inplace op is registered correctly.
        assert "_custom_mm2" in IrOp.registry
        assert IrOp.registry["_custom_mm2"] is _custom_mm2
        assert _custom_mm2.torch_op is torch.ops.vllm_ir._custom_mm2.default
        assert isinstance(_custom_mm2.maybe_inplace, IrOpInplaceOverload)
        assert (
            _custom_mm2.maybe_inplace.torch_op
            is torch.ops.vllm_ir._custom_mm2.maybe_inplace
        )

    def test_inplace_dispatching(self):
        # check that the correct implementation is dispatched based on priority,
        # and inplace semantics hold
        w = torch.randn(3, 3)
        x = torch.randn(2, 3)
        x1 = x.clone()

        with _custom_mm2.set_priority(["regular"]):
            result_regular = _custom_mm2.maybe_inplace(x, w)

        # check that the regular op does not modify x
        torch.testing.assert_close(x, x1)

        with _custom_mm2.set_priority(["inplace"]):
            result_inplace: Tensor = _custom_mm2.maybe_inplace(x, w)

        # check that the inplace op returns x directly
        assert result_inplace.data_ptr() == x.data_ptr()

        torch.testing.assert_close(result_inplace, x1 @ w + 2)
        torch.testing.assert_close(result_regular, x1 @ w + 1)

    def test_default_dispatching(self):
        # check that the correct implementation is dispatched when no priority is set,
        # and ops do not modify inputs
        w = torch.randn(3, 3)
        x = torch.randn(2, 3)
        x1 = x.clone()

        with _custom_mm2.set_priority(["regular"]):
            result_regular = _custom_mm2(x, w)

        with _custom_mm2.set_priority(["inplace"]):
            result_inplace = _custom_mm2(x, w)

        # check that x was not modified by either impl
        torch.testing.assert_close(x, x1)

        torch.testing.assert_close(result_inplace, x1 @ w + 2)
        torch.testing.assert_close(result_regular, x1 @ w + 1)

    @pytest.mark.xfail
    def test_trace(self):
        # Test that the inplace op can be used in a graph.
        def func(x: Tensor, y: Tensor) -> Tensor:
            return _custom_mm2(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        graph = make_fx(func)(x, y)
        assert any(node.target == "custom_mm2" for node in graph.graph.nodes)

        # Test that the inplace op can be used in an IrOpImpl.
        class CustomMM2Impl(IrOpImpl):
            def __init__(self):
                super().__init__("custom_mm2")

            def forward(self, x: Tensor, y: Tensor) -> Tensor:
                return _custom_mm2(x, y)

        impl = CustomMM2Impl()
        result = impl.forward(x, y)
        assert torch.allclose(result, x @ y)
