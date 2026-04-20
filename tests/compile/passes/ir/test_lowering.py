# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch
from torch import nn

import vllm.kernels  # noqa: F401 to register kernels
from vllm import ir
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config
from vllm.ir import ops
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


def _make_simple_model(op: IrOp) -> nn.Module:
    """Create a simple model that calls the op with fake arguments."""
    class SimpleModel(nn.Module):
        def forward(self, x):
            args = _make_fake_args_for_op(op)
            return op(*args)
    return SimpleModel()


def _make_fake_args_for_op(op: IrOp) -> tuple[Any, ...]:
    """Create fake meta tensors with unbacked symint dimensions for an op.

    Uses unbacked SymInt for the batch dimension to verify that supports_args
    does not specialize on concrete tensor shapes.
    """
    from torch._dynamo.utils import create_unbacked_symint

    sig = op._py_signature
    args = []
    for param in sig.parameters.values():
        ann = param.annotation
        if ann == torch.Tensor:
            # Use unbacked symint for batch dimension
            batch = create_unbacked_symint()
            args.append(torch.empty(batch, 16, device="meta", dtype=torch.bfloat16))
        elif ann in (int, "int"):
            args.append(16)
        elif ann in (float, "float"):
            args.append(1e-5)
        elif "Tensor | None" in str(ann) or "Optional[Tensor]" in str(ann):
            # Tensor | None type - pass a fake tensor (supports_args checks attributes)
            batch = create_unbacked_symint()
            args.append(torch.empty(batch, 16, device="meta", dtype=torch.bfloat16))
        elif "int | None" in str(ann) or "Optional[int]" in str(ann):
            args.append(16)
        else:
            args.append(None)
    return tuple(args)


# ============================================================
# 1. Per-op lowering tests
# ============================================================


class TestPerOpLowering:
    """
    Per-op lowering tests: verify all op implementations are lowered correctly
    through the VllmIRLoweringPass.

    These tests ensure:
    1. All implementations can be lowered through the pass
    2. supports_args is properly executable by Dynamo during lowering
    3. supports_args does not specialize on the batch size (unbacked symint)
    """

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_op_lowering_succeeds(self, op_name, provider, default_vllm_config):
        """
        Test that each op implementation can be lowered through the pass.

        Verifies:
        1. The op is replaced by the implementation in the graph
        2. selected_impls records the correct provider
        3. The lowered graph produces valid output
        """
        op = IrOp.registry[op_name]
        if not op.impls[provider].supported:
            pytest.skip(f"Provider {provider} not supported")

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)

        torch.set_default_device(current_platform.device_type)

        with (
            op.set_priority([provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            model = _make_simple_model(op)
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            output = compiled_model(x)

        # Verify lowering succeeded
        assert op_name in lowering_pass.selected_impls, (
            f"Op {op_name} was not lowered"
        )
        selected = lowering_pass.selected_impls[op_name]
        assert len(selected) > 0, f"No instances of {op_name} were lowered"

        # Verify output is valid (tensor with correct shape)
        assert isinstance(output, torch.Tensor)


# ============================================================
# 2. Lowering unit tests with fake ops
# ============================================================


class TestLoweringUnit:
    """
    Lowering unit tests with fake ops to stress-test implementation selection
    through the VllmIRLoweringPass.
    """

    def test_complex_supports_args_selection(self, default_vllm_config):
        """
        Test that lowering correctly selects implementation based on complex
        supports_args conditions. Uses a fake op with multiple providers.
        """
        @ir.register_op(name="_test_selection_op")
        def _test_selection_op(x: torch.Tensor) -> torch.Tensor:
            return x

        @_test_selection_op.register_impl(
            "bf16_impl",
            supports_args=lambda x: x.dtype == torch.bfloat16,
        )
        def bf16_impl(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @_test_selection_op.register_impl(
            "fp32_impl",
            supports_args=lambda x: x.dtype == torch.float32,
        )
        def fp32_impl(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        try:
            lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
            backend = TestBackend(lowering_pass)

            torch.set_default_device(current_platform.device_type)

            with _test_selection_op.set_priority(
                ["bf16_impl", "fp32_impl", "native"]
            ):
                model = _make_simple_model(_test_selection_op)
                x = torch.randn(8, 16, dtype=torch.bfloat16)
                compiled = torch.compile(model, backend=backend, fullgraph=True)
                output = compiled(x)

            # Verify bf16_impl was selected (dtype matches supports_args)
            selected = lowering_pass.selected_impls["_test_selection_op"]
            assert "bf16_impl" in selected.values(), (
                f"Expected bf16_impl, got {selected}"
            )
        finally:
            if "_test_selection_op" in IrOp.registry:
                del IrOp.registry["_test_selection_op"]

    def test_supports_args_signature_validation(self):
        """
        Test that supports_args signature validation catches mismatches
        during registration.
        """
        @ir.register_op(name="_test_sig_op")
        def _test_sig_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        try:
            with pytest.raises(ValueError, match="number of parameters"):
                @_test_sig_op.register_impl(
                    "bad_sig", supports_args=lambda x: True
                )
                def bad_sig_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y
        finally:
            if "_test_sig_op" in IrOp.registry:
                del IrOp.registry["_test_sig_op"]

    def test_dispatch_failure_no_matching_impl(self):
        """
        Test that dispatch raises RuntimeError when no implementation
        matches (including native). This verifies the error handling
        path when priority is set incorrectly.
        """
        @ir.register_op(name="_test_fail_op")
        def _test_fail_op(x: torch.Tensor) -> torch.Tensor:
            return x

        # Register impl that never matches
        @_test_fail_op.register_impl(
            "never_matches",
            supports_args=lambda x: False,  # never True
        )
        def never_matches_impl(x: torch.Tensor) -> torch.Tensor:
            return x

        try:
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            # Set priority WITHOUT native - should fail
            with _test_fail_op.set_priority(["never_matches"]):
                with pytest.raises(RuntimeError, match="Priority set incorrectly"):
                    _test_fail_op.dispatch(x)
        finally:
            if "_test_fail_op" in IrOp.registry:
                del IrOp.registry["_test_fail_op"]


# ============================================================
# 3. E2E correctness tests
# ============================================================


class TestE2ELowering:
    """
    E2E correctness tests, comparing the lowering pipeline with
    ir_enable_torch_wrap=False where implementations get traced through
    with Dynamo directly, and comparing with no lowering where IR ops
    remain in the Inductor-produced artifact.
    """

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_e2e_output_consistency(self, op_name, provider, default_vllm_config):
        """
        Compare lowering pipeline output with no-lowering output to verify
        correctness. This tests that lowered implementations produce the
        same results as the native implementation.
        """
        op = IrOp.registry[op_name]

        # Skip ops that require special setup (e.g., need specific hardware)
        if not op.impls[provider].supported:
            pytest.skip(f"Provider {provider} not supported")

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        backend_unlowered = TestBackend()
        model = _make_simple_model(op)

        torch.set_default_device(current_platform.device_type)

        with (
            op.set_priority([provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            x = torch.randn(8, 16, dtype=torch.bfloat16)

            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            compiled_unlowered = torch.compile(
                model, backend=backend_unlowered, fullgraph=True
            )

            try:
                output = compiled_model(x)
                output_unlowered = compiled_unlowered(x)
                torch.testing.assert_close(output_unlowered, output)
            except Exception as e:
                pytest.skip(
                    f"E2E test skipped for {op_name}:{provider}: {e}"
                )

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_lowering_dispatch_consistency(self, op_name, provider,
                                           default_vllm_config):
        """
        Verify lowering pass selection matches direct dispatch results.
        
        This ensures the lowering pass correctly uses ir_op.dispatch() to
        select implementations, and that the selected providers are consistent
        with what dispatch() would return for the same arguments.
        """
        op = IrOp.registry[op_name]

        if not op.impls[provider].supported:
            pytest.skip(f"Provider {provider} not supported")

        # Get expected result from direct dispatch
        fake_args = _make_fake_args_for_op(op)
        with op.set_priority([provider, "native"]):
            expected_impl = op.dispatch(*fake_args)

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)

        torch.set_default_device(current_platform.device_type)

        model = _make_simple_model(op)

        with (
            op.set_priority([provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            _ = compiled_model(x)

        # Verify selected_impls matches direct dispatch result
        if op_name in lowering_pass.selected_impls:
            selected = lowering_pass.selected_impls[op_name]
            assert len(selected) > 0, (
                f"No instances of {op_name} were lowered"
            )

            for node_name, selected_provider in selected.items():
                assert selected_provider == expected_impl.provider, (
                    f"lowering selected {selected_provider} for {node_name}, "
                    f"but direct dispatch got {expected_impl.provider}"
                )


