# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_mul_block_quant_fp8 import (
    pick_silu_mul_block_quant_fp8_config,
    silu_mul_block_quant_fp8,
    silu_mul_block_quant_fp8_baseline,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
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


def _generate_fake_input(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[Any, ...]:
    """Generate fake (FakeTensor) inputs matching the silu kernel signature."""
    with FakeTensorMode():
        # input has shape [num_tokens, 2 * hidden_size]
        input = torch.randn(
            (num_tokens, 2 * hidden_size), device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE)
        scales = torch.empty(
            (num_tokens, hidden_size // group_size),
            device="cuda",
            dtype=torch.float32,
        )
        scale_ub = torch.mean(input.abs()).to(torch.float32)
        args = (input, out, scales, group_size, scale_ub)
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluMulBlockQuantFp8ConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_size_64_num_tokens_16",
            "hidden_size_4096_group_size_128_num_tokens_16",
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "hidden_size_4096_group_size_128_num_tokens_16"

    def test_config_picker_closest_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_size_64_num_tokens_16",
            "hidden_size_2048_group_size_64_num_tokens_32",
            "hidden_size_2048_group_size_128_num_tokens_16",
            "hidden_size_2048_group_size_128_num_tokens_32",
            "hidden_size_4096_group_size_64_num_tokens_16",
            "hidden_size_4096_group_size_64_num_tokens_32",
            "hidden_size_4096_group_size_128_num_tokens_16",
            "hidden_size_4096_group_size_128_num_tokens_32",
        ]

        # hidden_size=3000 is between 2048 and 4096 -> closer to 2048 (diff 952 vs 1096)
        # group_size=70 is between 64 and 128 -> closer to 64 (diff 6 vs 58)
        # num_tokens=20 -> smallest available >= 20 is 32
        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "hidden_size_2048_group_size_64_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_size_64_num_tokens_16",
            "hidden_size_2048_group_size_64_num_tokens_32",
            "hidden_size_2048_group_size_128_num_tokens_16",
            "hidden_size_2048_group_size_128_num_tokens_32",
            "hidden_size_4096_group_size_64_num_tokens_16",
            "hidden_size_4096_group_size_64_num_tokens_32",
            "hidden_size_4096_group_size_128_num_tokens_16",
            "hidden_size_4096_group_size_128_num_tokens_32",
        ]

        # num_tokens=64 exceeds largest available (32) -> fall back to largest
        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "hidden_size_4096_group_size_128_num_tokens_32"

    def test_config_picker_malformed_key_raises(self):
        config_keys = ["bad_key"]

        args = _generate_fake_input(16, 4096, 128)
        with pytest.raises(ValueError):
            pick_silu_mul_block_quant_fp8_config(args, config_keys)

    def test_config_picker_only_default_no_parameterised_keys(self):
        """Only 'default' present and no parameterised keys -> returns 'default'."""
        config_keys = ["default"]

        args = _generate_fake_input(8, 2048, 64)
        selected_key = pick_silu_mul_block_quant_fp8_config(args, config_keys)
        assert selected_key == "default"


DTYPES = [torch.bfloat16, torch.float16]
# num_tokens x hidden_size pairs (hidden_size is the `output` half-width)
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, 128, 1024, 4096]],
    *[(16, i) for i in [64, 1024]],
    *[(256, i) for i in [64]],
]
SCALE_UBS = [True, False]
GROUP_SIZES = [64, 128]
SEEDS = [0]


class TestSiluMulBlockQuantFp8Correctness:
    @pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
    @pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("group_size", GROUP_SIZES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_silu_mul_block_quant_fp8(
        self,
        num_tokens: int,
        hidden_size: int,
        has_scale_ub: bool,
        dtype: torch.dtype,
        group_size: int,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported()

        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        _, fp8_max = get_fp8_min_max()
        scale = 1.0 / hidden_size
        # input has shape [num_tokens, 2 * hidden_size]: first half = gate, second = up
        input = (
            torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda") * scale
        )

        mean_val = torch.mean(input.abs())

        if has_scale_ub:
            scale_ub = (mean_val / fp8_max).to(dtype=torch.float32, device="cuda")
        else:
            scale_ub = None

        groups_per_row = hidden_size // group_size

        ref_out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE)
        ops_out = ref_out.clone()

        ref_scales = torch.empty(
            (num_tokens, groups_per_row), device="cuda", dtype=torch.float32
        )
        ops_scales = ref_scales.clone()

        silu_mul_block_quant_fp8_baseline(
            ref_out,
            input,
            ref_scales,
            group_size,
            scale_ub,
            False,
        )

        silu_mul_block_quant_fp8(
            input,
            ops_out,
            ops_scales,
            group_size,
            scale_ub,
        )

        ref_scales_c = ref_scales.contiguous()
        ops_scales_c = ops_scales.contiguous()

        torch.testing.assert_close(ref_scales_c, ops_scales_c)
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1

    @pytest.mark.parametrize("num_tokens", [1, 8, 32])
    def test_scale_ub_clamps_scales(self, num_tokens: int) -> None:
        # When scale_ub is set, all per-block scales must be <= scale_ub.
        skip_if_platform_unsupported()

        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        hidden_size = 128
        group_size = 64
        torch.manual_seed(0)
        input = torch.randn(
            num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE)
        scales = torch.empty(
            (num_tokens, hidden_size // group_size), device="cuda", dtype=torch.float32
        )
        _, fp8_max = get_fp8_min_max()
        mean_val = input.abs().mean()
        scale_ub_factor = (mean_val / fp8_max).to(dtype=torch.float32, device="cuda")

        silu_mul_block_quant_fp8(
            input, out, scales, group_size, scale_ub=scale_ub_factor
        )

        assert (scales <= scale_ub_factor.item() + 1e-6).all(), (
            f"Some scales {scales.max().item()}exceed scale_ub {scale_ub_factor.item()}"
        )


class TestSiluMulBlockQuantFp8Integration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_mul_block_quant_fp8" in registered_kernels

        kernel_wrapper = registered_kernels["silu_mul_block_quant_fp8"]
        assert kernel_wrapper.op_name == "silu_mul_block_quant_fp8"
        assert kernel_wrapper._config_picker is not None
        assert set(kernel_wrapper._mutates_args) == {"out", "scales"}

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported()
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_mul_block_quant_fp8"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        # fake impl should execute without error under FakeTensorMode
        result = fake_impl(*args)
        # kernel returns (out, scales) tuple; fake impl may return None or a tuple
        assert result is None or isinstance(result, tuple)

    def test_output_shapes(self):
        # Kernel must produce outputs with the expected shapes.
        skip_if_platform_unsupported()

        num_tokens, hidden_size, group_size = 8, 256, 64
        input = torch.randn(
            num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        scale_ub = torch.mean(input.abs()).to(dtype=torch.float32, device="cuda")
        out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE)
        scales = torch.empty(
            (num_tokens, hidden_size // group_size), device="cuda", dtype=torch.float32
        )

        silu_mul_block_quant_fp8(input, out, scales, group_size, scale_ub)

        assert out.shape == (num_tokens, hidden_size)
        assert scales.shape == (num_tokens, hidden_size // group_size)
        assert out.dtype == FP8_DTYPE
        assert scales.dtype == torch.float32

    def test_output_dtype_is_fp8(self):
        # Output tensor must be written in FP8 dtype.
        skip_if_platform_unsupported()

        num_tokens, hidden_size, group_size = 4, 128, 64
        input = torch.randn(
            num_tokens, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE)
        scales = torch.empty(
            (num_tokens, hidden_size // group_size), device="cuda", dtype=torch.float32
        )

        silu_mul_block_quant_fp8(input, out, scales, group_size)
        assert out.dtype == FP8_DTYPE
