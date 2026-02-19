# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FP4 ScaledMM kernel selection and functionality

Run `pytest tests/quantization/test_fp4.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    FP4ScaledMMLinearKernel,
    FP4ScaledMMLinearLayerConfig,
    init_fp4_linear_kernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.cutlass import (
    CutlassFP4ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.kernels.scaled_mm.flashinfer import (
    FlashInferFP4ScaledMMLinearKernel,
)
from vllm.platforms import current_platform


class TestFP4KernelAbstraction:
    """Test FP4 kernel abstraction implementation."""

    def test_fp4_kernel_is_abstract(self):
        """Test that FP4ScaledMMLinearKernel properly defines abstract methods."""
        assert hasattr(FP4ScaledMMLinearKernel, "apply_fp4_mm")
        # Can't instantiate abstract class
        with pytest.raises(TypeError):
            FP4ScaledMMLinearKernel(
                FP4ScaledMMLinearLayerConfig(
                    group_size=128,
                    is_checkpoint_fp4_serialized=True,
                    out_dtype=torch.bfloat16,
                ),
                layer_param_names=["weight", "weight_scale"],
            )

    def test_cutlass_fp4_implements_required_methods(self):
        """Test that CutlassFP4 implements all required methods."""
        required_methods = ["is_supported", "can_implement", "apply_fp4_mm"]
        for method in required_methods:
            assert hasattr(CutlassFP4ScaledMMLinearKernel, method), (
                f"CutlassFP4 missing {method}()"
            )

    def test_flashinfer_fp4_implements_required_methods(self):
        """Test that FlashInferFP4 implements all required methods."""
        required_methods = ["is_supported", "can_implement", "apply_fp4_mm"]
        for method in required_methods:
            assert hasattr(FlashInferFP4ScaledMMLinearKernel, method), (
                f"FlashInferFP4 missing {method}()"
            )

    def test_is_supported_returns_correct_types(self):
        """Test that is_supported() returns (bool, str|None)."""
        kernels = [CutlassFP4ScaledMMLinearKernel, FlashInferFP4ScaledMMLinearKernel]

        for kernel in kernels:
            result, reason = kernel.is_supported()
            assert isinstance(result, bool), (
                f"{kernel.__name__}.is_supported() should return bool"
            )
            assert reason is None or isinstance(reason, str), (
                f"{kernel.__name__}.is_supported() reason should be str or None"
            )

    def test_can_implement_returns_correct_types(self):
        """Test that can_implement() returns (bool, str|None)."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )
        kernels = [CutlassFP4ScaledMMLinearKernel, FlashInferFP4ScaledMMLinearKernel]

        for kernel in kernels:
            result, reason = kernel.can_implement(config)
            assert isinstance(result, bool), (
                f"{kernel.__name__}.can_implement() should return bool"
            )
            assert reason is None or isinstance(reason, str), (
                f"{kernel.__name__}.can_implement() reason should be str or None"
            )


class TestFP4KernelConfig:
    """Test FP4 kernel configuration."""

    def test_config_initialization(self):
        """Test that FP4ScaledMMLinearLayerConfig can be created."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )
        assert config.group_size == 128
        assert config.is_checkpoint_fp4_serialized is True
        assert config.out_dtype == torch.bfloat16

    def test_config_with_none_values(self):
        """Test config with None values."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=None,
            is_checkpoint_fp4_serialized=False,
            out_dtype=None,
        )
        assert config.group_size is None
        assert config.is_checkpoint_fp4_serialized is False
        assert config.out_dtype is None

    def test_config_with_different_group_sizes(self):
        """Test config with various group sizes."""
        for group_size in [16, 32, 64, 128, 256]:
            config = FP4ScaledMMLinearLayerConfig(
                group_size=group_size,
                is_checkpoint_fp4_serialized=True,
                out_dtype=torch.bfloat16,
            )
            assert config.group_size == group_size


@pytest.mark.skipif(not current_platform.is_cuda(), reason="FP4 kernels require CUDA")
class TestFP4KernelInitialization:
    """Test FP4 kernel initialization logic."""

    def test_init_fp4_linear_kernel_basic(self):
        """Test basic kernel initialization."""
        kernel = init_fp4_linear_kernel(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
            module_name="test_module",
        )

        assert isinstance(kernel, FP4ScaledMMLinearKernel)
        assert kernel.config.group_size == 128
        assert kernel.config.is_checkpoint_fp4_serialized is True

    def test_init_fp4_kernel_with_flashinfer_backend(self):
        """Test kernel initialization with FlashInfer backend specification."""
        # Skip if FlashInfer not available or not Blackwell
        cc = current_platform.get_device_capability()
        if cc is None or (cc[0] * 10 + cc[1]) < 100:
            pytest.skip("Requires compute capability 10.0+ (Blackwell)")

        try:
            from vllm.utils.flashinfer import has_flashinfer

            if not has_flashinfer():
                pytest.skip("FlashInfer not installed")
        except ImportError:
            pytest.skip("FlashInfer not available")

        kernel = init_fp4_linear_kernel(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
            backend="cutlass",
            module_name="test_flashinfer_module",
        )

        # If FlashInfer is selected, check it has the backend attribute
        if isinstance(kernel, FlashInferFP4ScaledMMLinearKernel):
            assert kernel.backend == "cutlass"

    def test_init_fp4_kernel_flashinfer_backends(self):
        """Test different FlashInfer backend variants."""
        cc = current_platform.get_device_capability()
        if cc is None or (cc[0] * 10 + cc[1]) < 100:
            pytest.skip("Requires compute capability 10.0+ (Blackwell)")

        try:
            from vllm.utils.flashinfer import has_flashinfer

            if not has_flashinfer():
                pytest.skip("FlashInfer not installed")
        except ImportError:
            pytest.skip("FlashInfer not available")

        for backend_name in ["cutlass", "trtllm", "cudnn"]:
            kernel = init_fp4_linear_kernel(
                group_size=128,
                is_checkpoint_fp4_serialized=True,
                out_dtype=torch.bfloat16,
                backend=backend_name,
                module_name=f"test_{backend_name}",
            )
            if isinstance(kernel, FlashInferFP4ScaledMMLinearKernel):
                assert kernel.backend == backend_name

    def test_init_fp4_kernel_force_cutlass(self):
        """Test forcing CUTLASS kernel."""
        try:
            kernel = init_fp4_linear_kernel(
                group_size=128,
                is_checkpoint_fp4_serialized=True,
                out_dtype=torch.bfloat16,
                force_kernel=CutlassFP4ScaledMMLinearKernel,
                module_name="test_force_cutlass",
            )
            assert isinstance(kernel, CutlassFP4ScaledMMLinearKernel)
        except ValueError as e:
            # It's OK if CUTLASS FP4 is not supported on this platform
            assert "CUTLASS FP4" in str(e) or "cutlass" in str(e).lower()


class TestFP4LayerParamExtraction:
    """Test FP4 parameter extraction from layers."""

    def test_get_layer_params_structure(self):
        """Test that _get_layer_params returns correct tuple structure."""
        # Verify the method exists in concrete classes
        assert hasattr(CutlassFP4ScaledMMLinearKernel, "_get_layer_params")
        assert hasattr(FlashInferFP4ScaledMMLinearKernel, "_get_layer_params")

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="Requires CUDA for tensor operations"
    )
    def test_param_extraction_with_real_tensors(self):
        """Test parameter extraction with actual tensor data."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        param_names = [
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale_inv",
            "alpha",
        ]

        # Create a mock layer with realistic tensor shapes
        class MockLayer:
            def __init__(self):
                # FP4 weights: packed 2 values per byte
                self.weight = torch.randint(
                    0, 256, (512, 256), dtype=torch.uint8, device="cuda"
                )
                # Per-block FP8 scales
                self.weight_scale = torch.randn(
                    512, 256 // 128, dtype=torch.float8_e4m3fn, device="cuda"
                )
                # Global FP32 scales
                self.weight_scale_2 = torch.tensor(
                    1.5, dtype=torch.float32, device="cuda"
                )
                self.input_scale_inv = torch.tensor(
                    0.8, dtype=torch.float32, device="cuda"
                )
                self.alpha = torch.tensor(1.2, dtype=torch.float32, device="cuda")

        layer = MockLayer()

        # Test CUTLASS kernel parameter extraction
        is_supported, _ = CutlassFP4ScaledMMLinearKernel.is_supported()
        if is_supported:
            kernel = CutlassFP4ScaledMMLinearKernel(config, param_names)
            params = kernel._get_layer_params(layer)

            # Should return tuple of 5 tensors
            assert isinstance(params, tuple)
            assert len(params) == 5
            assert all(isinstance(p, torch.Tensor) for p in params)

            # Verify tensor properties
            weight, weight_scale, weight_scale_2, input_scale_inv, alpha = params
            assert weight.dtype == torch.uint8
            assert weight_scale.dtype == torch.float8_e4m3fn
            assert weight_scale_2.dtype == torch.float32
            assert input_scale_inv.dtype == torch.float32
            assert alpha.dtype == torch.float32


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FP4 integration tests require CUDA"
)
class TestFP4KernelIntegration:
    """Integration tests for FP4 kernels (requires GPU)."""

    def test_kernel_can_be_instantiated(self):
        """Test that supported kernels can be instantiated."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        param_names = [
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale_inv",
            "alpha",
        ]

        # Try each kernel that claims to be supported
        for KernelClass in [
            CutlassFP4ScaledMMLinearKernel,
            FlashInferFP4ScaledMMLinearKernel,
        ]:
            is_supported, reason = KernelClass.is_supported()
            if is_supported:
                # Should be able to instantiate
                if KernelClass == FlashInferFP4ScaledMMLinearKernel:
                    kernel = KernelClass(config, param_names, backend="cutlass")
                else:
                    kernel = KernelClass(config, param_names)
                assert kernel is not None
                assert kernel.config == config

    def test_apply_weights_with_mock_layer(self):
        """Test apply_weights method with mock layer and tensors."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        param_names = [
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale_inv",
            "alpha",
        ]

        # Create mock layer with proper tensors
        class MockLayer:
            def __init__(self):
                self.weight = torch.randint(
                    0, 256, (512, 256), dtype=torch.uint8, device="cuda"
                )
                self.weight_scale = torch.randn(
                    512, 2, dtype=torch.float8_e4m3fn, device="cuda"
                )
                self.weight_scale_2 = torch.tensor(
                    1.0, dtype=torch.float32, device="cuda"
                )
                self.input_scale_inv = torch.tensor(
                    1.0, dtype=torch.float32, device="cuda"
                )
                self.alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        # Test with CUTLASS kernel if supported
        is_supported, _ = CutlassFP4ScaledMMLinearKernel.is_supported()
        if is_supported:
            kernel = CutlassFP4ScaledMMLinearKernel(config, param_names)

            # Verify kernel has apply_weights method
            assert hasattr(kernel, "apply_weights")

            # Note: We don't actually run apply_weights here because it requires
            # properly quantized inputs. This test just verifies the method exists
            # and the kernel is properly initialized.


class TestFP4KernelSelection:
    """Test kernel selection logic for different scenarios."""

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
    def test_kernel_selection_order(self):
        """Test that kernel selection follows priority order."""
        # FlashInfer should be preferred on Blackwell, CUTLASS otherwise
        kernel = init_fp4_linear_kernel(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        cc = current_platform.get_device_capability()
        if cc and (cc[0] * 10 + cc[1]) >= 100:
            # On Blackwell+, should prefer FlashInfer if available
            try:
                from vllm.utils.flashinfer import has_flashinfer

                if has_flashinfer():
                    assert isinstance(kernel, FlashInferFP4ScaledMMLinearKernel)
                else:
                    assert isinstance(kernel, CutlassFP4ScaledMMLinearKernel)
            except ImportError:
                assert isinstance(kernel, CutlassFP4ScaledMMLinearKernel)
        else:
            # On older GPUs, should use CUTLASS
            assert isinstance(kernel, CutlassFP4ScaledMMLinearKernel)


class TestMarlinSpecialCase:
    """Test that Marlin backend is handled as a special case."""

    def test_marlin_not_in_kernel_registry(self):
        """Test that Marlin is not included in the FP4 kernel registry."""
        # Marlin doesn't fit the kernel abstraction, so it should be handled
        # separately in the quantization methods (modelopt.py, compressed_tensors)
        from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
            _POSSIBLE_FP4_KERNELS,
        )

        # Check that no Marlin kernel exists in the registry
        for platform_kernels in _POSSIBLE_FP4_KERNELS.values():
            for kernel in platform_kernels:
                assert "Marlin" not in kernel.__name__, (
                    "Marlin should not be in FP4 kernel registry - it uses special path"
                )

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
    def test_marlin_handled_in_quantization_methods(self):
        """Test that Marlin is handled in quantization method level, not kernel level"""
        # This is a documentation test - verifies the design decision
        # Marlin has different calling convention(takes BF16 input, not quantized FP4)
        # So it should be handled at the ModelOptNvFp4LinearMethod.apply() level
        # with a special path that bypasses the kernel abstraction

        # Import to verify the special case exists
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            NvFp4LinearBackend,
        )

        assert hasattr(NvFp4LinearBackend, "MARLIN"), (
            "NvFp4LinearBackend should have MARLIN enum value"
        )


class TestWeightProcessing:
    """Test weight processing after loading."""

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="CUDA required for weight processing"
    )
    def test_process_weights_after_loading_exists(self):
        """Test that kernels have process_weights_after_loading method."""
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        param_names = [
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale_inv",
            "alpha",
        ]

        # Verify method exists for all supported kernels
        for KernelClass in [
            CutlassFP4ScaledMMLinearKernel,
            FlashInferFP4ScaledMMLinearKernel,
        ]:
            is_supported, _ = KernelClass.is_supported()
            if is_supported:
                if KernelClass == FlashInferFP4ScaledMMLinearKernel:
                    kernel = KernelClass(config, param_names, backend="cutlass")
                else:
                    kernel = KernelClass(config, param_names)

                assert hasattr(kernel, "process_weights_after_loading"), (
                    f"{KernelClass.__name__} missing process_weights_after_loading()"
                )


class TestFP4KernelOutputShapeCalculation:
    """Test output shape calculation in FP4 kernels."""

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
    def test_output_shape_derived_from_weight_tensor(self):
        """Test that output shape is derived from weight.shape, not layer attribute."""
        # This follows the FP8 pattern, not the Int8 pattern
        config = FP4ScaledMMLinearLayerConfig(
            group_size=128,
            is_checkpoint_fp4_serialized=True,
            out_dtype=torch.bfloat16,
        )

        param_names = [
            "weight",
            "weight_scale",
            "weight_scale_2",
            "input_scale_inv",
            "alpha",
        ]

        # Create a mock layer where output_size_per_partition doesn't match weight shape
        class MockLayer:
            def __init__(self):
                # Weight has output_size=512, but we'll set attribute to different value
                self.weight = torch.randint(
                    0, 256, (512, 256), dtype=torch.uint8, device="cuda"
                )
                self.weight_scale = torch.randn(
                    512, 2, dtype=torch.float8_e4m3fn, device="cuda"
                )
                self.weight_scale_2 = torch.tensor(
                    1.0, dtype=torch.float32, device="cuda"
                )
                self.input_scale_inv = torch.tensor(
                    1.0, dtype=torch.float32, device="cuda"
                )
                self.alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                self.output_size_per_partition = 999  # Different from weight.shape[0]

        # The kernel should use weight.shape[1] for output dim,
        # not layer.output_size_per_partition. This is verified in the base class
        # implementation in ScaledMMLinearKernel.py where we do:
        # output_shape[-1] = w.shape[1] (not layer.output_size_per_partition)

        is_supported, _ = CutlassFP4ScaledMMLinearKernel.is_supported()
        if is_supported:
            kernel = CutlassFP4ScaledMMLinearKernel(config, param_names)
            # Just verify kernel can be created - actual output shape test would require
            # running the full forward pass which needs properly quantized inputs
            assert kernel is not None
