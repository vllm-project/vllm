# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.ir.op
from vllm.ir.op import RESERVED_PROVIDERS, IrOp, IrOpImpl

# This should not exist
assert "_custom_add" not in IrOp.registry


@vllm.ir.register_op
def _custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def test_registration_overloads():
    assert all(
        n not in IrOp.registry for n in ["_custom_sub", "_custom_mul", "_custom_div"]
    )

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


def test_no_kw_only_args():
    # kw-only args not supported
    with pytest.raises(ValueError, match="keyword-only arguments"):

        @vllm.ir.register_op
        def _custom_kwarg_op(
            x: torch.Tensor, y: torch.Tensor, *, kwarg: int = 0
        ) -> torch.Tensor:
            return x + y + kwarg

    assert "_custom_kwarg_op" not in IrOp.registry


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

        # Calls native by default
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

        assert "dummy_provider" in _custom_add.impls
        assert isinstance(_custom_add.impls["dummy_provider"], IrOpImpl)

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


@_custom_add.register_impl("impl_a")
def impl_a(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 10


@_custom_add.register_impl("impl_b")
def impl_b(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 20


@_custom_add.register_impl("impl_even", supports_args=lambda a, b: a.size(1) % 2 == 0)
def impl_even(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 50


class TestIrOpImplDispatch:
    def test_register_impl(self):
        assert "impl_a" in _custom_add.impls
        impl = _custom_add.impls["impl_a"]

        assert impl is impl_a
        assert impl.op is _custom_add
        assert impl.provider == "impl_a"
        assert callable(impl.impl_fn)

        # Test duplicate registration rejected
        with pytest.raises(AssertionError):

            @_custom_add.register_impl("impl_a")
            def impl_a_dup(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + 30

        # Check the original impl is still intact
        assert _custom_add.impls["impl_a"] is impl_a

        # Check support all args
        assert impl_a.supports_all_args
        assert impl_b.supports_all_args
        assert not impl_even.supports_all_args

    def test_reserved_provider_rejected(self):
        for provider in RESERVED_PROVIDERS:
            with pytest.raises(AssertionError):

                @_custom_add.register_impl(provider)
                def bad_impl(x, y):
                    return x + y

    def test_set_priority_scoped(self):
        assert _custom_add.get_priority() == []

        with _custom_add.set_priority(["impl_even", "impl_b"]):
            assert _custom_add.get_priority() == ["impl_even", "impl_b"]

            # Check nesting
            with _custom_add.set_priority(["impl_b"]):
                assert _custom_add.get_priority() == ["impl_b"]

            # Restored
            assert _custom_add.get_priority() == ["impl_even", "impl_b"]

            # Check that exception restores priority
            with pytest.raises(ValueError), _custom_add.set_priority(["impl_a"]):
                assert _custom_add.get_priority() == ["impl_a"]
                raise ValueError("test exception")

            # Restored again
            assert _custom_add.get_priority() == ["impl_even", "impl_b"]

        # Restored to empty
        assert _custom_add.get_priority() == []

    def test_dispatch_priority_order(self):
        x = torch.tensor(1, dtype=torch.int32)
        y = torch.tensor(2, dtype=torch.int32)

        with _custom_add.set_priority(["impl_b", "impl_a"]):
            assert _custom_add.dispatch(x, y) is impl_b
            out1 = _custom_add(x, y)
            out2 = torch.ops.vllm_ir._custom_add(x, y)

            with _custom_add.set_priority(["impl_a"]):
                assert _custom_add.dispatch(x, y) is impl_a
                out3 = _custom_add(x, y)
                out4 = torch.ops.vllm_ir._custom_add(x, y)

        # impl_b
        assert out1.item() == 1 + 2 + 20
        assert out2.item() == 1 + 2 + 20
        # impl_a
        assert out3.item() == 1 + 2 + 10
        assert out4.item() == 1 + 2 + 10

    def test_unsupported_impl_filtered(self):
        @_custom_add.register_impl("unsupported", supported=False)
        def impl_bad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y + 999

        x = torch.tensor(1, dtype=torch.int32)
        y = torch.tensor(2, dtype=torch.int32)

        with _custom_add.set_priority(["unsupported", "impl_a"]):
            assert _custom_add.get_priority() == ["impl_a"]
            out = _custom_add(x, y)

        # impl_bad skipped → impl_a
        assert out.item() == 1 + 2 + 10

    def test_supports_args_runtime_dispatch_and_warning(
        self, caplog_vllm: pytest.LogCaptureFixture
    ):
        x1 = torch.ones((2, 2), dtype=torch.int32)
        y1 = torch.full((2, 2), 2, dtype=torch.int32)

        x2 = torch.ones((2, 3), dtype=torch.int32)
        y2 = torch.full((2, 3), 2, dtype=torch.int32)

        with (
            caplog_vllm.at_level(logging.WARNING),
            _custom_add.set_priority(["impl_even"]),
        ):
            # Test the warning about native fallback is logged (before even dispatching)
            assert len(caplog_vllm.records) == 1
            message = caplog_vllm.records[0].message
            assert "_custom_add" in message
            assert "fallback to native" in message
            assert "priority" in message

            # Check dispatching
            assert _custom_add.get_priority() == ["impl_even", "native"]
            assert _custom_add.dispatch(x1, y1) is impl_even
            assert _custom_add.dispatch(x2, y2) is _custom_add.impls["native"]

            out1 = _custom_add(x1, y1)  # size(1) == 2 → impl_even
            out2 = _custom_add(x2, y2)  # size(1) == 3 → native fallback

        # no other warnings
        assert len(caplog_vllm.records) == 1
        assert torch.all(out1 == 1 + 2 + 50)
        assert torch.all(out2 == 1 + 2)

    def test_default_priority(
        self, caplog_vllm: pytest.LogCaptureFixture, disable_log_dedup
    ):
        # Make sure logs are not deduplicated to properly test the warning
        x = torch.tensor([3], dtype=torch.int32)
        y = torch.tensor([4], dtype=torch.int32)

        # No priority set → falls back to native
        assert _custom_add.get_priority() == []
        with caplog_vllm.at_level(logging.WARNING):
            # Native by default
            assert _custom_add.dispatch(x, y) is _custom_add.impls["native"]
            out = _custom_add(x, y)

        # Check dispatching to native by default
        assert out.item() == 3 + 4

        # Check warning
        assert len(caplog_vllm.records) == 2
        message = caplog_vllm.records[0].message.lower()
        assert "_custom_add" in message
        assert "priority not set" in message


@vllm.ir.register_op
def _custom_mm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    tmp = x @ y
    return tmp if bias is None else tmp + bias


def test_default_args():
    # Test that default args are properly applied when dispatching and calling
    @_custom_mm.register_impl("impl_mm", supports_args=lambda x, y, b: True)
    def impl_mm(
        x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        tmp = x @ y
        return tmp + 50 if bias is None else tmp + bias + 100

    x1 = torch.tensor([1, 2], dtype=torch.int32)
    x2 = torch.tensor([3, 4], dtype=torch.int32)

    assert _custom_mm.apply_arg_defaults(x1, x2) == ((x1, x2, None), {})
    assert _custom_mm.apply_arg_defaults(x1, x2, None) == ((x1, x2, None), {})

    # Test that supports_args receives the defaulted args
    assert impl_mm.supports_args(x1, x2)
    with _custom_mm.set_priority(["impl_mm", "native"]):
        assert _custom_mm.dispatch(x1, x2) is impl_mm


def test_bad_impl_registrations():
    # Check bad schema
    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema")
        def impl_mm_bad_schema(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x @ y - 1

    assert "impl_mm_bad_schema" not in _custom_mm.impls

    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema_2")
        def impl_mm_bad_schema_2(
            x: torch.Tensor, y: torch.Tensor, b: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + b - 2

    assert "impl_mm_bad_schema_2" not in _custom_mm.impls

    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema_3")
        def impl_mm_bad_schema_3(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            return x @ y + bias - 5

    assert "impl_mm_bad_schema_3" not in _custom_mm.impls

    # check supports_args with incorrect params
    with pytest.raises(ValueError, match="supports_args must be a callable"):

        @_custom_mm.register_impl("impl_mm_bad_supports_args", supports_args=True)
        def impl_mm_bad_supports_args(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 10

    assert "impl_mm_bad_supports_args" not in _custom_mm.impls

    with pytest.raises(ValueError, match="number of parameters"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_2", supports_args=lambda x, y: True
        )
        def impl_mm_bad_supports_args(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 10

    assert "impl_mm_bad_supports_args_2" not in _custom_mm.impls

    with pytest.raises(ValueError, match="keyword-only parameters"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_3", supports_args=lambda x, y, *, b: True
        )
        def impl_mm_bad_supports_args_2(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 20

    assert "impl_mm_bad_supports_args_3" not in _custom_mm.impls
