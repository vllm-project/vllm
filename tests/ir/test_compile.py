# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch._dynamo

import vllm.ir.op
from vllm.ir.op import IrOpImplCompiledWrapper


# Create a simple IR op for testing
@vllm.ir.register_op
def _test_add_mul(
    x_a: torch.Tensor, x_b: torch.Tensor, scale: float = 2.0
) -> torch.Tensor:
    """Simple op: (a + b) * scale"""
    return (x_a + x_b) * scale


@pytest.fixture(autouse=True)
def clear_compile_wrapper():
    """Clear the compile wrapper before every test."""
    _test_add_mul.impls["native"].compile_clear()


def test_compile_context_manager_semantics():
    """Test that compile=True in set_priority produces correct results"""
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    scale = 3.0

    # Without compilation
    with _test_add_mul.set_priority(["native"], compile=False):
        assert not isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
        out_no_compile = _test_add_mul(a, b, scale)

    # With compilation
    with _test_add_mul.set_priority(["native"], compile=True):
        assert isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
        out_compile = _test_add_mul(a, b, scale)

    # Both should produce the same result
    torch.testing.assert_close(out_no_compile, out_compile)
    torch.testing.assert_close(out_compile, (a + b) * scale)


def test_no_recompile():
    """Test that dynamic shape is marked correctly and no recompilation happens."""
    torch._dynamo.reset()

    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    scale = 3.0

    # Without compilation
    with _test_add_mul.set_priority(["native"], compile=True):
        out1 = _test_add_mul(a, b, scale)

    torch.testing.assert_close(out1, (a + b) * scale)

    a = torch.randn(10, 5)
    b = torch.randn(10, 5)

    with (
        torch.compiler.set_stance("fail_on_recompile"),
        _test_add_mul.set_priority(["native"], compile=True),
    ):
        out2 = _test_add_mul(a, b, scale)

    torch.testing.assert_close(out2, (a + b) * scale)


def test_compile_inside_custom_op():
    """Test that compiled impl works when called inside a custom op"""

    # Create a custom op that calls our IR op internally
    @torch.library.custom_op("mylib::wrapped_add_mul", mutates_args=())
    def wrapped_add_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _test_add_mul(a, b, scale=3.0)

    # Use compile=True to optimize even though this custom op is opaque

    @wrapped_add_mul.register_fake
    def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    # Test the wrapped op
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)

    assert _test_add_mul.impls["native"]._compiled_wrapper is None
    with _test_add_mul.set_priority(["native"], compile=True):
        result = wrapped_add_mul(a, b)

    torch.testing.assert_close(result, (a + b) * 3.0)
    assert _test_add_mul.impls["native"]._compiled_wrapper is not None


def test_compile_with_custom_impl():
    """Test that compile=True works with custom implementations"""

    @_test_add_mul.register_impl("optimized", compiled=True)
    def optimized_impl(
        x_a: torch.Tensor, x_b: torch.Tensor, scale: float = 2.0
    ) -> torch.Tensor:
        # Alternative implementation
        return torch.mul(torch.add(x_a, x_b), scale) + 4

    a = torch.randn(3, 3)
    b = torch.randn(3, 3)

    # Use the custom implementation with compile=True
    with _test_add_mul.set_priority(["optimized"], compile=True):
        result = _test_add_mul(a, b, scale=1.5)

    torch.testing.assert_close(result, (a + b) * 1.5 + 4)


def test_nested_priority_contexts():
    """Test that nested set_priority contexts work correctly with compile"""
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)

    # Outer context with compile=False
    with _test_add_mul.set_priority(["native"], compile=False):
        assert not isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
        out1 = _test_add_mul(a, b)

        # Inner context with compile=True
        with _test_add_mul.set_priority(["native"], compile=True):
            assert isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
            out2 = _test_add_mul(a, b)

        # Back to compile=False
        assert not isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
        out3 = _test_add_mul(a, b)

    # All should produce the same result
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out2, out3)


def test_compiled_flag():
    """Test that compiled flag controls whether impl can be compiled"""
    # Native impl should have compiled=True
    assert _test_add_mul.impls["native"].compiled is True
    assert _test_add_mul.impls["native"].uncompiled_impl_fn is not None

    # Register impl without compiled flag (defaults to False)
    @_test_add_mul.register_impl("no_compile")
    def no_compile_impl(
        x_a: torch.Tensor, x_b: torch.Tensor, scale: float = 2.0
    ) -> torch.Tensor:
        return (x_a + x_b) * scale

    assert _test_add_mul.impls["no_compile"].compiled is False

    a = torch.randn(4, 5)
    b = torch.randn(4, 5)

    # compile=True with no_compile impl should NOT compile it
    with _test_add_mul.set_priority(["no_compile"], compile=True):
        # Should use the original impl, not compiled wrapper
        assert not isinstance(_test_add_mul.dispatch(a, b), IrOpImplCompiledWrapper)
        out = _test_add_mul(a, b)

    torch.testing.assert_close(out, (a + b) * 2.0)
