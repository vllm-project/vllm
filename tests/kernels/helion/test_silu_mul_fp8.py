# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_mul_fp8 import (
    pick_silu_mul_fp8_config,
    silu_mul_fp8,
    silu_mul_fp8_baseline,
)


def skip_if_platform_unsupported():
    try:
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        platform = get_canonical_gpu_name()

        try:
            config_manager = ConfigManager.get_instance()
        except RuntimeError:
            config_manager = ConfigManager()

        configs = config_manager.get_platform_configs("silu_mul_fp8", platform)
        if len(configs) == 0:
            pytest.skip("Current GPU platform not supported for silu_mul_fp8 kernel")

    except (ImportError, RuntimeError, KeyError):
        pytest.skip("Error detecting platform support for silu_mul_fp8 kernel")


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluMulFp8ConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "intermediate_2048_batchsize_256",
            "intermediate_4096_batchsize_256",
        ]

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        args = (input_tensor, scale)

        selected_key = pick_silu_mul_fp8_config(args, config_keys)
        assert selected_key == "intermediate_2048_batchsize_256"

    def test_config_picker_closest_match(self):
        config_keys = [
            "intermediate_2048_batchsize_256",
            "intermediate_4096_batchsize_256",
        ]
        # Use 7000 (intermediate_size=3500) which is closer to 4096 than 2048
        input_tensor = torch.randn(32, 7000, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        args = (input_tensor, scale)

        selected_key = pick_silu_mul_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_batchsize_256"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default", "some_other_key"]

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        args = (input_tensor, scale)

        selected_key = pick_silu_mul_fp8_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        args = (input_tensor, scale)

        selected_key = pick_silu_mul_fp8_config(args, config_keys)
        assert selected_key is None

    @pytest.mark.parametrize("intermediate_size", [2048, 4096, 5120])
    def test_config_picker_different_sizes(self, intermediate_size):
        config_keys = [
            "intermediate_2048_batchsize_256",
            "intermediate_4096_batchsize_256",
            "intermediate_5120_batchsize_256",
        ]

        input_tensor = torch.randn(
            32, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda"
        )
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        args = (input_tensor, scale)

        selected_key = pick_silu_mul_fp8_config(args, config_keys)
        expected_key = f"intermediate_{intermediate_size}_batchsize_256"
        assert selected_key == expected_key


class TestSiluMulFp8Correctness:
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    @pytest.mark.parametrize("intermediate_size", [2048, 3000, 3500, 4096, 5000])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu_mul_fp8_correctness(self, batch_size, intermediate_size, dtype):
        skip_if_platform_unsupported()

        input_size = 2 * intermediate_size
        input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        reference_output = silu_mul_fp8_baseline(input_tensor, scale)
        helion_output = silu_mul_fp8(input_tensor, scale)

        assert helion_output.shape == reference_output.shape
        assert helion_output.dtype == torch.float8_e4m3fn
        assert reference_output.dtype == torch.float8_e4m3fn

        ref_f32 = reference_output.to(torch.float32)
        helion_f32 = helion_output.to(torch.float32)
        # FP8 E4M3 has limited precision. Values near quantization boundaries
        # can round differently due to intermediate precision differences.
        torch.testing.assert_close(
            helion_f32,
            ref_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Mismatch at batch={batch_size}, size={intermediate_size}",
        )

    def test_silu_mul_fp8_shape_inference(self):
        skip_if_platform_unsupported()
        batch_size, input_size = 32, 8192
        intermediate_size = input_size // 2

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        output = silu_mul_fp8(input_tensor, scale)

        expected_shape = (batch_size, intermediate_size)
        assert output.shape == expected_shape
        assert output.dtype == torch.float8_e4m3fn

    def test_silu_mul_fp8_scale_variations(self):
        skip_if_platform_unsupported()
        batch_size, input_size = 16, 4096

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )

        scales = [0.1, 0.5, 1.0, 2.0, 10.0]

        for scale_val in scales:
            scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda")

            reference_output = silu_mul_fp8_baseline(input_tensor, scale)
            helion_output = silu_mul_fp8(input_tensor, scale)
            ref_f32 = reference_output.to(torch.float32)
            helion_f32 = helion_output.to(torch.float32)

            torch.testing.assert_close(
                helion_f32,
                ref_f32,
                atol=0.05,
                rtol=0.05,
                msg=f"Mismatch for scale={scale_val}",
            )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 4096),
            (16, 4096),
            (128, 4096),
            (1024, 4096),
            (1, 8192),
            (16, 8192),
            (128, 8192),
        ],
    )
    def test_silu_mul_fp8_various_shapes(self, shape):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        reference_output = silu_mul_fp8_baseline(input_tensor, scale)
        helion_output = silu_mul_fp8(input_tensor, scale)

        assert helion_output.shape == reference_output.shape

        ref_f32 = reference_output.to(torch.float32)
        helion_f32 = helion_output.to(torch.float32)

        torch.testing.assert_close(
            helion_f32, ref_f32, atol=0.05, rtol=0.05, msg=f"Mismatch for shape={shape}"
        )


def silu_mul_fp8_pytorch(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference using F.silu.

    This matches vLLM's SiluAndMul.forward_native exactly:
    F.silu(x[..., :d]) * x[..., d:]
    """
    d = input.shape[-1] // 2
    result = F.silu(input[..., :d]) * input[..., d:]
    return (result.to(torch.float32) / scale).to(torch.float8_e4m3fn)


class TestSiluMulFp8PytorchReference:
    """Tests comparing Helion kernel against pure PyTorch implementation.

    Uses tighter tolerance since both use PyTorch's FP8 conversion
    (same rounding mode), unlike the vLLM C++ baseline which uses
    NVIDIA's hardware FP8 conversion with different rounding.
    """

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 256])
    @pytest.mark.parametrize("intermediate_size", [1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu_mul_fp8_vs_pytorch(self, batch_size, intermediate_size, dtype):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(
            batch_size, 2 * intermediate_size, dtype=dtype, device="cuda"
        )
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        pytorch_output = silu_mul_fp8_pytorch(input_tensor, scale)
        helion_output = silu_mul_fp8(input_tensor, scale)

        assert helion_output.shape == pytorch_output.shape
        assert helion_output.dtype == torch.float8_e4m3fn

        pytorch_f32 = pytorch_output.to(torch.float32)
        helion_f32 = helion_output.to(torch.float32)

        # Tolerance accounts for FP8 quantization boundary effects
        torch.testing.assert_close(
            helion_f32,
            pytorch_f32,
            atol=0.05,
            rtol=0.05,
            msg=(
                f"Mismatch at batch={batch_size}, size={intermediate_size}, "
                f"dtype={dtype}"
            ),
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 4096),  # 3D input
            (2, 4, 2048),  # 3D input
            (1, 1, 1, 8192),  # 4D input
        ],
    )
    def test_silu_mul_fp8_multidim_vs_pytorch(self, shape):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        pytorch_output = silu_mul_fp8_pytorch(input_tensor, scale)
        helion_output = silu_mul_fp8(input_tensor, scale)

        assert helion_output.shape == pytorch_output.shape

        pytorch_f32 = pytorch_output.to(torch.float32)
        helion_f32 = helion_output.to(torch.float32)

        torch.testing.assert_close(
            helion_f32,
            pytorch_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Mismatch for shape={shape}",
        )


class TestSiluMulFp8Integration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_mul_fp8" in registered_kernels

        kernel_wrapper = registered_kernels["silu_mul_fp8"]
        assert kernel_wrapper.op_name == "silu_mul_fp8"
        assert kernel_wrapper._config_picker is not None

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported()
        from vllm.kernels.helion.register import get_registered_kernels

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_mul_fp8"]
        fake_impl = kernel_wrapper._fake_impl

        fake_output = fake_impl(input_tensor, scale)

        expected_shape = (32, 2048)
        assert fake_output.shape == expected_shape
        assert fake_output.dtype == torch.float8_e4m3fn
        assert fake_output.device == input_tensor.device
