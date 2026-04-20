# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx

import vllm.ir.op
from vllm.ir.op import RESERVED_PROVIDERS, IrOp, IrOpImpl

# This should not exist
assert "_custom_add" not in IrOp.registry


class CustomError(Exception):
    pass


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

    # Direct construction does not register directly
    def _custom_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x / y

    custom_div = IrOp("_custom_div", _custom_div, False)
    assert custom_div.name == "_custom_div"
    assert "_custom_div" not in IrOp.registry

    # Duplicate op registration not allowed
    with pytest.raises(AssertionError):

        @vllm.ir.register_op
        def _custom_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x * y - 100


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

    @pytest.mark.parametrize("enable_torch_wrap", [True, False])
    @pytest.mark.parametrize("symbolic_trace", [True, False])
    def test_trace_sees_single_custom_op(
        self, symbolic_trace: bool, enable_torch_wrap: bool
    ):
        def fn(x, y):
            return _custom_add(x, y)

        def find_fn(target: Any, gm: fx.GraphModule):
            return gm.graph.find_nodes(op="call_function", target=target)

        with pytest.raises(CustomError), vllm.ir.enable_torch_wrap(enable_torch_wrap):
            if symbolic_trace:
                gm = torch.fx.symbolic_trace(fn)
            else:
                gm = make_fx(fn)(torch.randn(2, 2), torch.randn(2, 2))

            x1, y1 = torch.rand(5, 4), torch.rand(5, 4)
            out_fx = gm(x1, y1)
            out_eager = fn(x1, y1)

            # raise error to check enable_torch_wrap context restored correctly
            raise CustomError

        # check behavior matches eager in all cases
        torch.testing.assert_close(out_fx, out_eager)

        # check that IR nodes only appear if enable_torch_wrap=True
        ir_nodes = find_fn(torch.ops.vllm_ir._custom_add.default, gm)
        if enable_torch_wrap:
            assert len(ir_nodes) == 1, gm.code
        else:
            assert len(ir_nodes) == 0, gm.code

        # with torch wrapping enabled (default), IR nodes appear
        if symbolic_trace:
            gm = torch.fx.symbolic_trace(fn)
        else:
            gm = make_fx(fn)(torch.randn(2, 2), torch.randn(2, 2))

        ir_nodes = find_fn(torch.ops.vllm_ir._custom_add.default, gm)
        assert len(ir_nodes) == 1, gm.code


@_custom_add.register_impl("impl_a")
def impl_a(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 10


@_custom_add.register_impl("impl_b")
def impl_b(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 20


@_custom_add.register_impl("impl_even", supports_args=lambda x, y: x.size(1) % 2 == 0)
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
            with pytest.raises(CustomError), _custom_add.set_priority(["impl_a"]):
                assert _custom_add.get_priority() == ["impl_a"]
                raise CustomError

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


@vllm.ir.register_op(has_reduction=True)
def _custom_mm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    tmp = x @ y
    return tmp if bias is None else tmp + bias


@_custom_mm.register_impl("impl_mm", supports_args=lambda x, y, bias=None: True)
def impl_mm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    tmp = x @ y
    return tmp + 50 if bias is None else tmp + bias + 100


def test_default_args():
    # Test that default args are properly applied when dispatching and calling
    x1 = torch.tensor([1, 2], dtype=torch.int32)
    x2 = torch.tensor([3, 4], dtype=torch.int32)

    # Test that supports_args receives the defaulted args
    assert impl_mm.supports_args(x1, x2)
    with _custom_mm.set_priority(["impl_mm", "native"]):
        assert _custom_mm.dispatch(x1, x2) is impl_mm


@_custom_add.register_impl(
    "bv_impl_add", batch_invariant=False, supports_args=lambda x, y: True
)
def bv_impl_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y + 70


@_custom_mm.register_impl(
    "bi_impl_mm", batch_invariant=True, supports_args=lambda x, y, bias=None: True
)
def bi_impl_mm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x @ y + 20 + bias


@_custom_mm.register_impl(
    "bv_impl_mm", batch_invariant=False, supports_args=lambda x, y, bias=None: True
)
def bv_impl_mm(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x @ y + 20 + bias


def test_batch_invariant_defaults():
    # _custom_add is batch invariant by default
    assert _custom_add.impls["native"].batch_invariant
    assert _custom_add.impls["impl_even"].batch_invariant
    assert not _custom_add.impls["bv_impl_add"].batch_invariant

    # _custom_mm is not batch invariant by default
    assert _custom_mm.impls["native"].batch_invariant
    assert _custom_mm.impls["bi_impl_mm"].batch_invariant
    assert not _custom_mm.impls["impl_mm"].batch_invariant
    assert not _custom_mm.impls["bv_impl_mm"].batch_invariant


def test_batch_invariant_dispatching():
    # batch invariance off, all ops remain
    with _custom_add.set_priority(
        ["bv_impl_add", "impl_even", "native"], batch_invariant_only=False
    ):
        assert _custom_add.get_priority() == ["bv_impl_add", "impl_even", "native"]

    # batch invariance required, filter ops
    with _custom_add.set_priority(
        ["impl_even", "bv_impl_add", "native"], batch_invariant_only=True
    ):
        assert _custom_add.get_priority() == ["impl_even", "native"]

    # batch invariance off, all ops remain
    with _custom_mm.set_priority(
        ["bv_impl_mm", "bi_impl_mm", "native"], batch_invariant_only=False
    ):
        assert _custom_mm.get_priority() == ["bv_impl_mm", "bi_impl_mm", "native"]

    # batch invariance required, filter ops
    with _custom_mm.set_priority(
        ["bv_impl_mm", "bi_impl_mm", "native"], batch_invariant_only=True
    ):
        assert _custom_mm.get_priority() == ["bi_impl_mm", "native"]


def test_bad_impl_registrations():
    # Check bad schema
    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema")
        def impl_mm_bad_schema(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x @ y - 1

    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema_2")
        def impl_mm_bad_schema_2(
            x: torch.Tensor, y: torch.Tensor, b: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + b - 2

    with pytest.raises(ValueError, match="does not match native schema"):

        @_custom_mm.register_impl("impl_mm_bad_schema_3")
        def impl_mm_bad_schema_3(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            return x @ y + bias - 5

    # check supports_args with incorrect params
    with pytest.raises(ValueError, match="supports_args must be a callable"):

        @_custom_mm.register_impl("impl_mm_bad_supports_args", supports_args=True)
        def impl_mm_bad_supports_args(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 10

    with pytest.raises(ValueError, match="number of parameters"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_2", supports_args=lambda x, y: True
        )
        def impl_mm_bad_supports_args(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 10

    with pytest.raises(ValueError, match="keyword-only parameters"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_3", supports_args=lambda x, y, *, b: True
        )
        def impl_mm_bad_supports_args_2(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 20

    with pytest.raises(ValueError, match="does not match native parameter"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_4", supports_args=lambda x, y, b: True
        )
        def impl_mm_bad_supports_args_4(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 30

    with pytest.raises(ValueError, match="does not match native default"):

        @_custom_mm.register_impl(
            "impl_mm_bad_supports_args_5", supports_args=lambda x, y, bias=1: True
        )
        def impl_mm_bad_supports_args_5(
            x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            return x @ y + 40

    assert set(_custom_mm.impls.keys()) == {
        "bi_impl_mm",
        "bv_impl_mm",
        "impl_mm",
        "native",
    }


IMPL_OOT_SRC = """
import torch

@_custom_mm.register_impl("impl_mm_oot")
def impl_mm_oot(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x @ y - 99
"""


def load_custom_mm_module(file_path: Path):
    spec = importlib.util.spec_from_file_location("_custom_mm_oot", file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)

    # Inject the variable into the module's global namespace
    # This allows the @_custom_mm.register_impl decorator to work
    module._custom_mm = _custom_mm  # type: ignore[attr-defined]

    # Execute the file; this triggers the decorator
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_uuid_and_oot(tmp_path: Path):
    file_path = tmp_path / "_custom_mm_oot.py"
    file_path.write_text(IMPL_OOT_SRC)

    assert "impl_mm_oot" not in _custom_mm.impls
    _ = load_custom_mm_module(file_path)
    assert "impl_mm_oot" in _custom_mm.impls

    uuid = _custom_mm.impls["impl_mm_oot"].uuid()
    del _custom_mm.impls["impl_mm_oot"]

    # Replace file source
    file_path.write_text(IMPL_OOT_SRC + " # added file source")
    assert "impl_mm_oot" not in _custom_mm.impls
    _ = load_custom_mm_module(file_path)
    assert "impl_mm_oot" in _custom_mm.impls

    uuid1 = _custom_mm.impls["impl_mm_oot"].uuid()
    assert uuid1 != uuid
    del _custom_mm.impls["impl_mm_oot"]

    # Back to original
    file_path.write_text(IMPL_OOT_SRC)
    assert "impl_mm_oot" not in _custom_mm.impls
    _ = load_custom_mm_module(file_path)
    assert "impl_mm_oot" in _custom_mm.impls

    uuid2 = _custom_mm.impls["impl_mm_oot"].uuid()
    assert uuid2 == uuid
    assert uuid2 != uuid1
    del _custom_mm.impls["impl_mm_oot"]
