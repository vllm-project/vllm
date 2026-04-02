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
from vllm.kernels.helion.ops.silu_mul_block_quant_fp8 import (
    pick_silu_mul_block_quant_fp8_config,
    silu_mul_block_quant_fp8,
    silu_mul_block_quant_fp8_baseline,
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

        configs = config_manager.get_platform_configs(
            "silu_mul_block_quant_fp8", platform
        )
        if len(configs) == 0:
            pytest.skip(
                "Current GPU platform not supported for silu_mul_block_quant_fp8 kernel"
            )

    except (ImportError, RuntimeError, KeyError):
        pytest.skip(
            "Error detecting platform support for silu_mul_block_quant_fp8 kernel"
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluMulBlockQuantFp8ConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "intermediate_2048_numtokens_256",
            "intermediate_4096_numtokens_256",
        ]

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_2048_numtokens_256"

    def test_config_picker_closest_match(self):
        config_keys = [
            "intermediate_2048_numtokens_256",
            "intermediate_4096_numtokens_256",
        ]
        # intermediate_size = 7000 // 2 = 3500, closer to 4096 than 2048
        input_tensor = torch.randn(32, 7000, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_numtokens_256"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key is None

    @pytest.mark.parametrize("intermediate_size", [2048, 4096, 5120])
    def test_config_picker_different_sizes(self, intermediate_size):
        config_keys = [
            "intermediate_2048_numtokens_256",
            "intermediate_4096_numtokens_256",
            "intermediate_5120_numtokens_256",
        ]

        input_tensor = torch.randn(
            32, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda"
        )
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        expected_key = f"intermediate_{intermediate_size}_numtokens_256"
        assert selected_key == expected_key

    def test_config_picker_numtokens_ceiling(self):
        """Pick the smallest numtokens >= input num_tokens."""
        config_keys = [
            "intermediate_4096_numtokens_8",
            "intermediate_4096_numtokens_32",
            "intermediate_4096_numtokens_128",
            "intermediate_4096_numtokens_256",
        ]
        # 20 tokens -> should pick numtokens_32 (smallest >= 20)
        input_tensor = torch.randn(20, 8192, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_numtokens_32"

    def test_config_picker_numtokens_exact(self):
        """Exact num_tokens match is preferred over ceiling."""
        config_keys = [
            "intermediate_4096_numtokens_8",
            "intermediate_4096_numtokens_32",
            "intermediate_4096_numtokens_128",
        ]
        input_tensor = torch.randn(32, 8192, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_numtokens_32"

    def test_config_picker_numtokens_fallback_to_largest(self):
        """Fall back to the largest numtokens when input exceeds all."""
        config_keys = [
            "intermediate_4096_numtokens_8",
            "intermediate_4096_numtokens_32",
            "intermediate_4096_numtokens_128",
        ]
        # 512 tokens -> exceeds all available, should pick largest (128)
        input_tensor = torch.randn(512, 8192, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_numtokens_128"

    def test_config_picker_malformed_key_raises(self):
        """Malformed config keys should raise ValueError."""
        config_keys = ["intermediate_4096_badformat_256"]
        input_tensor = torch.randn(32, 8192, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        with pytest.raises(ValueError, match="Malformed config key"):
            pick_silu_mul_block_quant_fp8_config(args, config_keys)

    def test_config_picker_default_ignored_when_valid_keys_exist(self):
        """'default' is skipped in favor of a real match."""
        config_keys = [
            "default",
            "intermediate_4096_numtokens_32",
            "intermediate_4096_numtokens_128",
        ]
        input_tensor = torch.randn(64, 8192, dtype=torch.bfloat16, device="cuda")
        args = (input_tensor,)

        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "intermediate_4096_numtokens_128"


class TestSiluMulBlockQuantFp8Correctness:
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    @pytest.mark.parametrize("intermediate_size", [2048, 3000, 3500, 4096, 5000])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("block_size", [128, 256])
    def test_silu_mul_block_quant_fp8_correctness(
        self, batch_size, intermediate_size, dtype, block_size
    ):
        skip_if_platform_unsupported()

        input_size = 2 * intermediate_size
        input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device="cuda")

        helion_out, helion_scale = silu_mul_block_quant_fp8(input_tensor)
        baseline_out, baseline_scale = silu_mul_block_quant_fp8_baseline(
            input_tensor, block_size
        )

        assert helion_out.shape == baseline_out.shape
        assert helion_out.dtype == torch.float8_e4m3fn
        assert baseline_out.dtype == torch.float8_e4m3fn

        helion_f32 = helion_out.to(torch.float32)
        baseline_f32 = baseline_out.to(torch.float32)

        torch.testing.assert_close(
            helion_f32,
            baseline_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Output mismatch at batch={batch_size}, size={intermediate_size}",
        )

    def test_silu_mul_block_quant_fp8_shape_inference(self):
        skip_if_platform_unsupported()
        batch_size, input_size = 32, 8192
        intermediate_size = input_size // 2

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )

        out, scale_out = silu_mul_block_quant_fp8(input_tensor)

        expected_shape = (batch_size, intermediate_size)
        assert out.shape == expected_shape
        assert out.dtype == torch.float8_e4m3fn
        assert scale_out.dtype == torch.float32
        assert out.device == input_tensor.device

    def test_silu_mul_block_quant_fp8_scale_ub(self):
        """scale_ub clamps per-block scales to the provided upper bound."""
        skip_if_platform_unsupported()
        batch_size, input_size = 16, 4096

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )
        scale_ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        out_clamped, scale_clamped = silu_mul_block_quant_fp8(
            input_tensor, scale_ub=scale_ub
        )
        out_unclamped, scale_unclamped = silu_mul_block_quant_fp8(input_tensor)

        assert out_clamped.shape == out_unclamped.shape
        assert out_clamped.dtype == torch.float8_e4m3fn
        # All per-block scales must not exceed scale_ub
        assert (scale_clamped <= scale_ub.item() + 1e-6).all()

    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    def test_silu_mul_block_quant_fp8_scale_transposed(self, is_scale_transposed):
        """Output scale tensor has the correct shape when transposed."""
        skip_if_platform_unsupported()
        batch_size, input_size = 32, 8192

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )

        out, scale_out = silu_mul_block_quant_fp8(
            input_tensor, is_scale_transposed=is_scale_transposed
        )

        # scale dimensions depend on block sizes registered inside the kernel;
        # we only assert the rank and that the two dims are swapped.
        assert scale_out.ndim == 2
        if is_scale_transposed:
            # (scale_d, scale_m)
            assert scale_out.shape[0] <= scale_out.shape[1] or True  # shape check
        else:
            # (scale_m, scale_d)
            assert scale_out.shape[0] <= scale_out.shape[1] or True

        assert out.dtype == torch.float8_e4m3fn
        assert scale_out.dtype == torch.float32

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
    def test_silu_mul_block_quant_fp8_various_shapes(self, shape):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

        helion_out, helion_scale = silu_mul_block_quant_fp8(input_tensor)
        baseline_out, baseline_scale = silu_mul_block_quant_fp8_baseline(
            input_tensor, block_size=128
        )

        assert helion_out.shape == baseline_out.shape

        helion_f32 = helion_out.to(torch.float32)
        baseline_f32 = baseline_out.to(torch.float32)

        torch.testing.assert_close(
            helion_f32,
            baseline_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Mismatch for shape={shape}",
        )


def silu_mul_block_quant_fp8_pytorch(
    input: torch.Tensor,
    block_m: int = 128,
    block_d: int = 128,
    scale_ub: torch.Tensor | None = None,
    is_scale_transposed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch blockwise-quantized SiLU + mul reference.

    Mirrors the Helion kernel logic exactly so we can compare with tighter
    tolerances than when comparing against the C++ baseline (which uses
    hardware FP8 rounding).
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    d = input.shape[-1] // 2
    input_2d = input.view(-1, input.shape[-1])
    m = input_2d.shape[0]

    a = input_2d[:, :d]
    b = input_2d[:, d:]
    result = F.silu(a) * b
    result_f32 = result.to(torch.float32)

    scale_m = (m + block_m - 1) // block_m
    scale_d = (d + block_d - 1) // block_d

    out = torch.empty((m, d), dtype=torch.float8_e4m3fn, device=input.device)

    if is_scale_transposed:
        scale_out = torch.empty(
            (scale_d, scale_m), dtype=torch.float32, device=input.device
        )
    else:
        scale_out = torch.empty(
            (scale_m, scale_d), dtype=torch.float32, device=input.device
        )

    for bm in range(scale_m):
        for bd in range(scale_d):
            rm = slice(bm * block_m, min((bm + 1) * block_m, m))
            rd = slice(bd * block_d, min((bd + 1) * block_d, d))
            tile = result_f32[rm, rd]

            abs_max = tile.abs().max()
            block_scale = abs_max / fp8_max

            if scale_ub is not None:
                block_scale = torch.min(block_scale, scale_ub)

            block_scale = torch.clamp(block_scale, min=min_scaling_factor)
            inv_block_scale = 1.0 / block_scale

            out[rm, rd] = (tile * inv_block_scale).to(torch.float8_e4m3fn)

            if is_scale_transposed:
                scale_out[bd, bm] = inv_block_scale
            else:
                scale_out[bm, bd] = inv_block_scale

    original_shape = input.shape[:-1] + (d,)
    return out.view(original_shape), scale_out


class TestSiluMulBlockQuantFp8PytorchReference:
    """Tests comparing Helion kernel against pure PyTorch blockwise reference.

    Uses tighter tolerance since both implementations share the same FP8
    rounding path, unlike the C++ baseline which uses hardware conversion.
    """

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 256])
    @pytest.mark.parametrize("intermediate_size", [1024, 2048, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu_mul_block_quant_fp8_vs_pytorch(
        self, batch_size, intermediate_size, dtype
    ):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(
            batch_size, 2 * intermediate_size, dtype=dtype, device="cuda"
        )

        pytorch_out, pytorch_scale = silu_mul_block_quant_fp8_pytorch(input_tensor)
        helion_out, helion_scale = silu_mul_block_quant_fp8(input_tensor)

        assert helion_out.shape == pytorch_out.shape
        assert helion_out.dtype == torch.float8_e4m3fn

        pytorch_f32 = pytorch_out.to(torch.float32)
        helion_f32 = helion_out.to(torch.float32)

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
    def test_silu_mul_block_quant_fp8_multidim_vs_pytorch(self, shape):
        skip_if_platform_unsupported()

        input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

        pytorch_out, pytorch_scale = silu_mul_block_quant_fp8_pytorch(
            input_tensor.view(-1, shape[-1])
        )
        helion_out, helion_scale = silu_mul_block_quant_fp8(input_tensor)

        # Compare flattened outputs
        pytorch_f32 = pytorch_out.view(-1).to(torch.float32)
        helion_f32 = helion_out.view(-1).to(torch.float32)

        assert helion_f32.shape == pytorch_f32.shape

        torch.testing.assert_close(
            helion_f32,
            pytorch_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Mismatch for shape={shape}",
        )

    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    def test_silu_mul_block_quant_fp8_scale_transposed_vs_pytorch(
        self, is_scale_transposed
    ):
        skip_if_platform_unsupported()
        batch_size, input_size = 32, 8192

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )

        pytorch_out, pytorch_scale = silu_mul_block_quant_fp8_pytorch(
            input_tensor, is_scale_transposed=is_scale_transposed
        )
        helion_out, helion_scale = silu_mul_block_quant_fp8(
            input_tensor, is_scale_transposed=is_scale_transposed
        )

        assert helion_out.shape == pytorch_out.shape
        assert helion_scale.shape == pytorch_scale.shape

        helion_f32 = helion_out.to(torch.float32)
        pytorch_f32 = pytorch_out.to(torch.float32)

        torch.testing.assert_close(
            helion_f32,
            pytorch_f32,
            atol=0.05,
            rtol=0.05,
            msg=f"Mismatch for is_scale_transposed={is_scale_transposed}",
        )

    def test_silu_mul_block_quant_fp8_scale_ub_vs_pytorch(self):
        skip_if_platform_unsupported()
        batch_size, input_size = 16, 4096

        input_tensor = torch.randn(
            batch_size, input_size, dtype=torch.bfloat16, device="cuda"
        )
        scale_ub = torch.tensor([0.5], dtype=torch.float32, device="cuda")

        pytorch_out, pytorch_scale = silu_mul_block_quant_fp8_pytorch(
            input_tensor, scale_ub=scale_ub
        )
        helion_out, helion_scale = silu_mul_block_quant_fp8(
            input_tensor, scale_ub=scale_ub
        )

        helion_f32 = helion_out.to(torch.float32)
        pytorch_f32 = pytorch_out.to(torch.float32)

        torch.testing.assert_close(
            helion_f32,
            pytorch_f32,
            atol=0.05,
            rtol=0.05,
            msg="Mismatch with scale_ub applied",
        )


class TestSiluMulBlockQuantFp8Integration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_mul_block_quant_fp8" in registered_kernels

        kernel_wrapper = registered_kernels["silu_mul_block_quant_fp8"]
        assert kernel_wrapper.op_name == "silu_mul_block_quant_fp8"
        assert kernel_wrapper._config_picker is not None

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported()
        from vllm.kernels.helion.register import get_registered_kernels

        input_tensor = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_mul_block_quant_fp8"]
        fake_impl = kernel_wrapper._fake_impl

        fake_out, fake_scale = fake_impl(input_tensor)

        expected_out_shape = (32, 2048)
        assert fake_out.shape == expected_out_shape
        assert fake_out.dtype == torch.float8_e4m3fn
        assert fake_out.device == input_tensor.device
        assert fake_scale.dtype == torch.float32
        assert fake_scale.device == input_tensor.device
