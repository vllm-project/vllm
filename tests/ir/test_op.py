# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.ir.op
from vllm.ir.op import IrOp

# This should not exist
assert "_custom_add" not in IrOp.registry


@vllm.ir.register_op
def _custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def test_registration_overloads():
    # Calling with decorator
    @vllm.ir.register_op()
    def _custom_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y

    assert _custom_sub.name == "_custom_sub"
    assert _custom_sub is IrOp.registry["_custom_sub"]

    # Custom name
    @vllm.ir.register_op(name="_custom_mul")
    def custom_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y

    assert custom_mul.name == "_custom_mul"
    assert custom_mul is IrOp.registry["_custom_mul"]

    # Direct construction is equivalent
    def _custom_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x / y

    custom_div = IrOp("_custom_div", _custom_div)
    assert custom_div.name == "_custom_div"
    assert custom_div is IrOp.registry["_custom_div"]


class TestIrOpCustomAdd:
    # Registration invariants
    def test_decorated_object(self):
        """Make sure that referring directly to an op is correct"""
        assert isinstance(_custom_add, IrOp)
        assert "_custom_add" in IrOp.registry
        assert _custom_add is IrOp.registry["_custom_add"]

    def test_torch_op_is_registered(self):
        assert hasattr(torch.ops.vllm_ir, "_custom_add")
        assert callable(torch.ops.vllm_ir._custom_add.default)

    # Semantic correctness
    def test_semantics_match_native(self):
        x = torch.randn(4, 5)
        y = torch.randn(4, 5)

        out = _custom_add(x, y)
        ref = x + y

        torch.testing.assert_close(out, ref)

    # -------------------------
    # Implementation registration
    # -------------------------

    def test_register_impl_is_non_intrusive(self):
        @_custom_add.register_impl("dummy_provider")
        def dummy_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y + 123

        assert "dummy_provider" in _custom_add._impls
        assert callable(_custom_add._impls["dummy_provider"])

        x = torch.ones(2, 2)
        y = torch.ones(2, 2)

        # Native semantics must still hold
        torch.testing.assert_close(_custom_add(x, y), x + y)

    def test_schema_contains_tensor_signature(self):
        schema = _custom_add._schema_str

        assert "Tensor" in schema
        assert "-> Tensor" in schema

    # -------------------------
    # FX visibility
    # -------------------------

    def test_fx_sees_single_custom_op(self):
        def fn(x, y):
            return _custom_add(x, y)

        gm = torch.fx.symbolic_trace(fn)

        assert any(
            "vllm_ir._custom_add" in str(n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        )

    def test_dynamo_sees_single_custom_op(self):
        def fn(x, y):
            return _custom_add(x, y)

        fx_gm = make_fx(fn)(torch.randn(2, 2), torch.randn(2, 2))

        assert any(
            "vllm_ir._custom_add" in str(n.target)
            for n in fx_gm.graph.nodes
            if n.op == "call_function"
        )
