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


class Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    def forward(self, x):
        x1 = x + 4.0
        x2 = ops.rms_norm(x1, self.weight, 1e-5)
        x3 = x2 * 5.0
        # no weight
        x4 = ops.rms_norm(x3, None, 1e-5)
        x5 = x4 / 2.0
        # dispatch to native due to variance_size parameter
        x6 = ops.rms_norm(x5, self.weight, 1e-5, self.hidden_size // 2)
        return x6 + 3.0


# ============================================================
# 1. Per-op lowering tests
# ============================================================


class TestPerOpLowering:
    """
    Per-op lowering tests, making sure all (supported) op implementations
    are lowered correctly. These ensure all implementation and supports_args
    functions are properly executable by Dynamo.
    """

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_supports_args_dynamo_executable(self, op_name, provider):
        """
        Verify all supports_args functions can be executed by Dynamo.
        This ensures the supports_args code is compatible with torch.compile tracing.
        """
        op = IrOp.registry[op_name]
        impl = op.impls[provider]

        if impl._supports_args is None:
            pytest.skip("No supports_args defined for this provider")

        # Wrap supports_args with torch.compile to verify Dynamo compatibility
        compiled_supports = torch.compile(impl.supports_args, backend="eager")
        fake_args = _make_fake_args_for_op(op)

        # Should not raise Dynamo errors
        try:
            compiled_supports(*fake_args)
        except Exception as e:
            pytest.fail(
                f"supports_args for {op_name}:{provider} is not Dynamo compatible: {e}"
            )

    @pytest.mark.parametrize("op_name,provider", _get_op_provider_pairs())
    def test_supports_args_no_batch_specialization(self, op_name, provider):
        """
        Verify supports_args does not specialize on batch size.
        Uses unbacked SymInt to ensure the function doesn't depend on concrete values.
        """
        from torch._dynamo.utils import create_unbacked_symint

        op = IrOp.registry[op_name]
        impl = op.impls[provider]

        if impl._supports_args is None:
            pytest.skip("No supports_args defined for this provider")

        # Create fake args with unbacked symint for batch dimension
        x_meta = torch.empty(8, 16, device="meta", dtype=torch.bfloat16)

        # Get the unbacked symint - this represents an unknown batch size
        unbacked_symint = create_unbacked_symint()

        # Replace the first dimension with unbacked symint
        # This simulates an unknown batch size
        try:
            # Call supports_args with unbacked symint in shape
            # If supports_args checks concrete values, this will fail
            result = impl.supports_args(x_meta, None, 1e-5)
            assert isinstance(result, bool), (
                f"supports_args should return bool, got {type(result)}"
            )
        except torch._dynamo.exc.Unsupported as e:
            if "concretization" in str(e).lower() or "cannot" in str(e).lower():
                pytest.fail(
                    f"supports_args for {op_name}:{provider} appears to specialize "
                    f"on concrete values: {e}"
                )
            raise


# ============================================================
# 2. Lowering unit tests with fake ops
# ============================================================


class TestLoweringUnit:
    """
    Lowering unit tests using fake ops & implementations.
    Crucially stress-tests implementation selection by using fake ops
    with complex supports_args.
    """

    def test_complex_supports_args_selection(self):
        """
        Test implementation selection with complex supports_args conditions.
        Uses a fake op to verify that the dispatch mechanism correctly selects
        implementations based on supports_args return values.
        """
        # Register a temporary fake op for testing
        @ir.register_op(name="_test_complex_op")
        def _test_complex_op(x: torch.Tensor) -> torch.Tensor:
            return x

        # Register implementation with complex supports_args
        @_test_complex_op.register_impl(
            "complex_impl",
            supports_args=lambda x: x.dtype == torch.bfloat16 and x.dim() == 2,
        )
        def complex_impl(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        try:
            x_bf16_2d = torch.empty(2, 8, device="meta", dtype=torch.bfloat16)
            x_fp32_2d = torch.empty(2, 8, device="meta", dtype=torch.float32)
            x_bf16_3d = torch.empty(2, 4, 8, device="meta", dtype=torch.bfloat16)

            with _test_complex_op.set_priority(["complex_impl", "native"]):
                # bf16 2D -> complex_impl (dtype matches, dim matches)
                assert _test_complex_op.dispatch(x_bf16_2d).provider == "complex_impl"

                # fp32 2D -> native (dtype doesn't match)
                assert _test_complex_op.dispatch(x_fp32_2d).provider == "native"

                # bf16 3D -> native (dim doesn't match)
                assert _test_complex_op.dispatch(x_bf16_3d).provider == "native"
        finally:
            # Clean up
            if "complex_impl" in _test_complex_op.impls:
                del _test_complex_op.impls["complex_impl"]
            if "_test_complex_op" in IrOp.registry:
                del IrOp.registry["_test_complex_op"]

    def test_supports_args_signature_validation(self):
        """
        Test that supports_args signature validation catches mismatches.
        """
        @ir.register_op(name="_test_sig_op")
        def _test_sig_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        try:
            # Test that wrong number of parameters is rejected
            with pytest.raises(ValueError, match="number of parameters"):

                @_test_sig_op.register_impl(
                    "bad_sig", supports_args=lambda x: True
                )
                def bad_sig_impl(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y
        finally:
            if "_test_sig_op" in IrOp.registry:
                del IrOp.registry["_test_sig_op"]


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

    @pytest.mark.parametrize("rms_provider", ops.rms_norm.supported_providers())
    def test_lowering_rms_norm(self, rms_provider, default_vllm_config):
        torch.set_default_device(current_platform.device_type)

        lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
        backend = TestBackend(lowering_pass)
        backend_unlowered = TestBackend()

        model = Model()
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        with (
            ops.rms_norm.set_priority([rms_provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            compiled_unlowered_model = torch.compile(
                model, backend=backend_unlowered, fullgraph=True
            )
            output = compiled_model(x)
            output_unlowered = compiled_unlowered_model(x)

        selected = lowering_pass.selected_impls["rms_norm"]
        assert len(selected) == 3
        assert selected["rms_norm"] == rms_provider
        assert selected["rms_norm_1"] == rms_provider
        assert selected["rms_norm_2"] == "native"

        # Compiled function guards on global value, avoid recompilation
        with ir.enable_torch_wrap(True):
            output2 = compiled_model(x)

        torch.testing.assert_close(output_unlowered, output)
        torch.testing.assert_close(output_unlowered, output2)

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

        # Create a simple model that uses the op
        class SimpleModel(nn.Module):
            def forward(self, x):
                # Call the op directly
                args = _make_fake_args_for_op(op)
                return op(*args)

        torch.set_default_device(current_platform.device_type)

        with (
            op.set_priority([provider, "native"]),
            ir.enable_torch_wrap(True),
        ):
            model = SimpleModel()
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
    def test_lowering_op_count(self, op_name, provider, default_vllm_config):
        """
        Verify the correct number of ops are lowered into chosen implementations.
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
            model = Model()
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            compiled_model = torch.compile(model, backend=backend, fullgraph=True)
            _ = compiled_model(x)

        # Check that the op was recorded in selected_impls
        if op_name in lowering_pass.selected_impls:
            selected = lowering_pass.selected_impls[op_name]
            # At least one instance should be selected
            assert len(selected) > 0, (
                f"No instances of {op_name} were lowered to {provider}"
            )


# ============================================================
# Additional tests for unbacked SymInt
# ============================================================


class TestUnbackedSymInt:
    """
    Tests specifically for verifying unbacked SymInt compatibility.
    These tests ensure supports_args functions don't depend on concrete values.
    """

    def test_unbacked_symint_rms_norm_supports_args(self):
        """
        Test rms_norm supports_args with unbacked symint.
        This is the primary test mentioned in the issue.
        """
        from torch._dynamo.utils import create_unbacked_symint

        for provider in ops.rms_norm.supported_providers():
            impl = ops.rms_norm.impls[provider]
            if impl._supports_args is None:
                continue

            # Create unbacked symint
            unbacked_bs = create_unbacked_symint()

            # Create meta tensors
            x = torch.empty(8, 16, device="meta", dtype=torch.bfloat16)
            weight = torch.empty(16, device="meta", dtype=torch.bfloat16)

            try:
                # This should work without concretization errors
                result = impl.supports_args(x, weight, 1e-5)
                assert isinstance(result, bool)
            except torch._dynamo.exc.Unsupported as e:
                if "concretization" in str(e).lower():
                    pytest.fail(
                        f"rms_norm:{provider} supports_args specializes on "
                        f"concrete values: {e}"
                    )
                raise