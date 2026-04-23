# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Callable

import pytest
import torch

import vllm.kernels  # noqa: F401 to register kernels
from tests.ir.ir_test_utils import _make_simple_model
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir.op import IrOp
from vllm.platforms import current_platform

from ...backend import TestBackend


@contextlib.contextmanager
def _register_test_op(
    name: str,
    native_fn: Callable[..., torch.Tensor],
    input_generator: Callable[[], tuple],
    impls: dict[str, tuple[Callable[..., bool] | None, Callable]],
):
    """Register a test op with implementations, auto-cleanup on exit."""
    op = ir.register_op(name=name)(native_fn)
    op.register_input_generator(input_generator)

    for impl_name, (supports_args, impl_fn) in impls.items():
        op.register_impl(impl_name, supports_args=supports_args)(impl_fn)

    try:
        yield op
    finally:
        if name in IrOp.registry:
            del IrOp.registry[name]


# ============================================================
# Lowering unit tests with fake ops
# ============================================================


class TestFakeOpLowering:
    """
    Lowering unit tests with fake ops to stress-test implementation selection
    through the VllmIRLoweringPass.
    """

    def test_impl_selection_and_fallback(self, default_vllm_config):
        """
        Test that lowering correctly selects implementation based on supports_args,
        and falls back to native when no custom impl matches.
        """
        torch.set_default_device(current_platform.device_type)

        # --- Test 1: dtype-based selection ---
        def _selection_input_gen() -> tuple:
            return (torch.randn(8, 16, dtype=torch.bfloat16),)

        def _selection_native_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def _bf16_impl_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        def _fp32_impl_fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        with _register_test_op(
            "_test_selection_op",
            native_fn=_selection_native_fn,
            input_generator=_selection_input_gen,
            impls={
                "bf16_impl": (lambda x: x.dtype == torch.bfloat16, _bf16_impl_fn),
                "fp32_impl": (lambda x: x.dtype == torch.float32, _fp32_impl_fn),
            },
        ) as selection_op:
            lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
            backend = TestBackend(lowering_pass)
            with selection_op.set_priority(["bf16_impl", "fp32_impl", "native"]):
                real_args = selection_op.generate_inputs()
                model = _make_simple_model(selection_op, real_args)
                x = torch.randn(8, 16, dtype=torch.bfloat16)
                compiled = torch.compile(model, backend=backend, fullgraph=True)
                _ = compiled(x)

            # Verify bf16_impl was selected
            selected = lowering_pass.selected_impls["_test_selection_op"]
            assert "bf16_impl" in selected.values()

        # --- Test 2: fallback to native ---
        def _fallback_input_gen() -> tuple:
            return (torch.randn(8, 16, dtype=torch.bfloat16),)

        def _fallback_native_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def _never_matches_impl_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        with _register_test_op(
            "_test_fallback_op",
            native_fn=_fallback_native_fn,
            input_generator=_fallback_input_gen,
            impls={
                "never_matches": (lambda x: False, _never_matches_impl_fn),
            },
        ) as fallback_op:
            lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
            backend = TestBackend(lowering_pass)
            # Set priority WITHOUT native - it will be auto-appended
            with fallback_op.set_priority(["never_matches"]):
                real_args = fallback_op.generate_inputs()
                model = _make_simple_model(fallback_op, real_args)
                x = torch.randn(8, 16, dtype=torch.bfloat16)
                compiled = torch.compile(model, backend=backend, fullgraph=True)
                _ = compiled(x)

            # Verify native was selected (fallback)
            selected = lowering_pass.selected_impls["_test_fallback_op"]
            assert "native" in selected.values()

    def test_supports_args_validation(self):
        """
        Test supports_args signature validation and batch size specialization detection.
        """

        # --- Test 1: signature mismatch ---
        @ir.register_op(name="_test_sig_op")
        def _test_sig_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        try:
            with pytest.raises(ValueError, match="number of parameters"):

                @_test_sig_op.register_impl("bad_sig", supports_args=lambda x: True)
                def bad_sig_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y
        finally:
            if "_test_sig_op" in IrOp.registry:
                del IrOp.registry["_test_sig_op"]

        # --- Test 2: batch size specialization (negative test) ---
        def _batch_dep_input_gen() -> tuple:
            return (torch.randn(8, 16, dtype=torch.bfloat16),)

        def _batch_dep_native_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def _batch_dep_impl_fn(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        with _register_test_op(
            "_test_batch_dep_op",
            native_fn=_batch_dep_native_fn,
            input_generator=_batch_dep_input_gen,
            impls={
                "batch_dep_impl": (lambda x: x.size(0) == 8, _batch_dep_impl_fn),
            },
        ) as batch_op:
            fake_args = batch_op.generate_symbolic_inputs()
            impl = batch_op.impls["batch_dep_impl"]
            result = impl.supports_args(*fake_args)

            # This should NOT be a bool - it's a SymBool (Eq(u0, 8))
            with pytest.raises(AssertionError, match="isinstance"):
                assert isinstance(result, bool)
