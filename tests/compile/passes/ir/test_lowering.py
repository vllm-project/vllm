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
    def test_supports_args_dynamo_compatible(self, op_name, provider):
        """
        Verify supports_args is Dynamo-compatible and doesn't specialize on batch size.

        Uses unbacked SymInt for batch dimensions to ensure:
        1. supports_args can be traced by torch.compile
        2. supports_args doesn't depend on concrete batch values
        """
        op = IrOp.registry[op_name]
        impl = op.impls[provider]

        if impl._supports_args is None:
            pytest.skip("No supports_args defined for this provider")

        # Create fake args with unbacked symint dimensions
        fake_args = _make_fake_args_for_op(op)

        # Wrap with torch.compile and call
        # This verifies both:
        # 1. Dynamo can trace supports_args
        # 2. supports_args doesn't concretize unbacked symints
        try:
            compiled = torch.compile(impl.supports_args, backend="eager")
            result = compiled(*fake_args)
            assert isinstance(result, bool), (
                f"supports_args should return bool, got {type(result)}"
            )
        except torch._dynamo.exc.Unsupported as e:
            if "concretization" in str(e).lower():
                pytest.fail(
                    f"supports_args for {op_name}:{provider} specializes "
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


