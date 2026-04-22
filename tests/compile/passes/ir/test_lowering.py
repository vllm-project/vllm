# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Callable

import pytest
import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir.op import IrOp
from vllm.platforms import current_platform

from ...backend import TestBackend


def _get_op_provider_pairs() -> list[tuple[str, str]]:
    """Get all (op_name, provider) pairs for parametrization."""
    pairs: list[tuple[str, str]] = []
    for op_name, op in IrOp.registry.items():
        for provider in op.supported_providers():
            pairs.append((op_name, provider))
    return pairs


def _make_simple_model(op: IrOp, real_args: tuple) -> nn.Module:
    """Create a simple model that calls the op with given arguments."""

    class SimpleModel(nn.Module):
        def forward(self, x):
            return op(*real_args)

    return SimpleModel()


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


@pytest.fixture
def lowering_backend(default_vllm_config):
    """Shared lowering backend fixture."""
    torch.set_default_device(current_platform.device_type)
    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    return lowering_pass, TestBackend(lowering_pass)


# ============================================================
# 1. Per-op Lowering tests (parametrized across all ops)
# ============================================================


class TestPerOpLowering:
    """
    Per-op lowering tests: verify all op implementations are lowered correctly
    through the VllmIRLoweringPass.

    These tests ensure:
    1. supports_args is properly executable by Dynamo during lowering
    2. supports_args does not specialize on the batch size (unbacked symint)
    3. The op is replaced by the implementation in the graph
    """

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_op_lowering(self, op_name, provider, default_vllm_config):
        """
        Test that each op implementation can be lowered through the pass.

        Verifies:
        1. supports_args does not crash on unbacked symint (batch-agnostic)
        2. The op is replaced by the implementation in the graph
        3. selected_impls records the correct provider
        """
        op = IrOp.registry[op_name]
        if not op.impls[provider].supported:
            pytest.skip(f"Provider {provider} not supported")

        # Step 1: Verify supports_args works with unbacked symint
        with op.enable_symbolic():
            fake_args = op.generate_inputs()
        impl = op.impls[provider]
        supports_result = impl.supports_args(*fake_args)
        assert isinstance(supports_result, bool), (
            f"supports_args should return bool, got {type(supports_result)}"
        )

        # Step 2: Perform lowering test
        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        torch.set_default_device(current_platform.device_type)

        with (
            op.set_priority([provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            real_args = op.generate_inputs()
            model = _make_simple_model(op, real_args)
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            output = compiled_model(x)

        # Verify lowering succeeded
        assert op_name in lowering_pass.selected_impls, f"Op {op_name} was not lowered"
        selected = lowering_pass.selected_impls[op_name]
        assert len(selected) > 0, f"No instances of {op_name} were lowered"
        assert isinstance(output, torch.Tensor)


# ============================================================
# 2. Lowering unit tests with fake ops
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
            with batch_op.enable_symbolic():
                fake_args = batch_op.generate_inputs()
            impl = batch_op.impls["batch_dep_impl"]
            result = impl.supports_args(*fake_args)

            # This should NOT be a bool - it's a SymBool (Eq(u0, 8))
            with pytest.raises(AssertionError, match="isinstance"):
                assert isinstance(result, bool)


# ============================================================
# 3. E2E correctness tests
# ============================================================


class TestE2ELowering:
    """
    E2E correctness tests, comparing the lowering pipeline with baselines.
    """

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_e2e_correctness(self, op_name, provider, default_vllm_config):
        """
        Compare lowering pipeline output with two baselines:
        1. ir_enable_torch_wrap=False: implementations traced by Dynamo directly
        2. No lowering at all: IR ops remain in the Inductor-produced artifact
        Also verify lowering pass selection matches direct dispatch results.
        """
        op = IrOp.registry[op_name]

        if not op.impls[provider].supported:
            pytest.skip(f"Provider {provider} not supported")

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        torch.set_default_device(current_platform.device_type)
        x = torch.randn(8, 16, dtype=torch.bfloat16)

        # Generate inputs once so all models use the same inputs
        real_args = op.generate_inputs()

        # Get expected result from direct dispatch
        with op.set_priority([provider, "native"]):
            expected_impl = op.dispatch(*real_args)

        # Case 1: lowering enabled with torch_wrap=True
        with op.set_priority([provider, "native"]), ir.enable_torch_wrap(True):
            model = _make_simple_model(op, real_args)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            output = compiled_model(x.clone())

        # Verify lowering pass selection matches direct dispatch
        if op_name in lowering_pass.selected_impls:
            selected = lowering_pass.selected_impls[op_name]
            assert len(selected) > 0
            for node_name, selected_provider in selected.items():
                assert selected_provider == expected_impl.provider, (
                    f"lowering selected {selected_provider} for {node_name}, "
                    f"but direct dispatch got {expected_impl.provider}"
                )

        # Case 2: torch_wrap=False - implementations traced by Dynamo directly
        backend_no_wrap = TestBackend()
        with op.set_priority([provider, "native"]), ir.enable_torch_wrap(False):
            model_no_wrap = _make_simple_model(op, real_args)
            compiled_no_wrap = torch.compile(
                model_no_wrap, backend=backend_no_wrap, fullgraph=True
            )
            output_no_wrap = compiled_no_wrap(x.clone())

        # Case 3: no lowering at all - IR ops remain in Inductor artifact
        backend_no_lowering = TestBackend()
        with op.set_priority([provider, "native"]):
            model_no_lowering = _make_simple_model(op, real_args)
            compiled_no_lowering = torch.compile(
                model_no_lowering, backend=backend_no_lowering, fullgraph=True
            )
            output_no_lowering = compiled_no_lowering(x.clone())

        # Use op-defined tolerances for dtype-aware comparison
        dtype = output.dtype
        tolerance = op.get_tolerance(dtype)
        torch.testing.assert_close(
            output, output_no_wrap, atol=tolerance["atol"], rtol=tolerance["rtol"]
        )
        torch.testing.assert_close(
            output, output_no_lowering, atol=tolerance["atol"], rtol=tolerance["rtol"]
        )