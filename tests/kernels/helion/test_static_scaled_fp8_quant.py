# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_static_scaled_fp8_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.static_scaled_fp8_quant import (
    pick_config,
    static_scaled_fp8_quant_dispatch,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize,
)
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(
    num_tokens: int,
    hidden_size: int,
    group_shape_m: int,
    group_shape_n: int,
) -> tuple[Any, ...]:
    with FakeTensorMode():
        input = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        result = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
        group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
        group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
        num_group_m = num_tokens // group_m
        num_group_n = hidden_size // group_n
        scale = torch.randn(
            num_group_m, num_group_n, device=input.device, dtype=torch.float32
        )

        args = (result, input, scale, group_m, group_n)
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestStaticScaledFp8QuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_shape_m_-1_group_shape_n_1_num_tokens_16",
            "hidden_size_4096_group_shape_m_-1_group_shape_n_1_num_tokens_16",
            "hidden_size_2048_group_shape_m_1_group_shape_n_-1_num_tokens_16",
            "hidden_size_4096_group_shape_m_1_group_shape_n_-1_num_tokens_16",
        ]

        args = _generate_fake_input(16, 4096, 1, -1)
        selected_key = pick_config(args, config_keys)
        assert (
            selected_key
            == "hidden_size_4096_group_shape_m_1_group_shape_n_-1_num_tokens_16"
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_shape_m_32_group_shape_n_64_num_tokens_64",
            "hidden_size_4096_group_shape_m_32_group_shape_n_64_num_tokens_64",
            "hidden_size_2048_group_shape_m_256_group_shape_n_512_num_tokens_512",
            "hidden_size_4096_group_shape_m_256_group_shape_n_512_num_tokens_512",
        ]

        args = _generate_fake_input(128, 2560, 64, 128)
        selected_key = pick_config(args, config_keys)
        assert (
            selected_key
            == "hidden_size_2048_group_shape_m_32_group_shape_n_64_num_tokens_64"
        )

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_fake_input(16, 4096, 1, -1)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_fake_input(16, 4096, 1, -1)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_shape_m_64_group_shape_n_64_num_tokens_16",
            "hidden_size_4096_group_shape_m_64_group_shape_n_64_num_tokens_16",
            "hidden_size_2048_group_shape_m_128_group_shape_n_128_num_tokens_16",
            "hidden_size_4096_group_shape_m_128_group_shape_n_128_num_tokens_16",
        ]

        args = _generate_fake_input(32, 8192, 256, 256)
        selected_key = pick_config(args, config_keys)
        assert (
            selected_key
            == "hidden_size_4096_group_shape_m_128_group_shape_n_128_num_tokens_16"
        )

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key",
        ]

        args = _generate_fake_input(16, 4096, 1, -1)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


DTYPES = [torch.bfloat16, torch.float]
SEEDS = [0]

# Test static FP8 quantization with 2D group scales
GROUP_SHAPES_2D = [
    (-1, -1),  # Per-tensor
    (-1, 1),  # Per-channel
    (1, -1),  # Per-token
    (-1, 128),  # Per-head quantization
    (1, 128),  # DeepSeek-style per-token-per-group (group_m=1, group_n=128)
    (128, 128),  # DeepSeek-style block quantization
    (1, 64),  # Smaller group size
    (1, 16),  # Small group (scalar path in kernel)
    (4, 256),  # Non-trivial both dimensions
]
# Use sizes divisible by all group shapes
NUM_TOKENS_GROUP = [128, 512]
HIDDEN_SIZES_GROUP = [256, 1024, 2048]


class TestStaticScaledFp8QuantCorrectness:
    @pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
    @pytest.mark.parametrize("group_shape", GROUP_SHAPES_2D)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_static_fp8_quant_group_2d(
        self,
        num_tokens: int,
        hidden_size: int,
        group_shape: tuple[int, int],
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        """Test static FP8 quantization with 2D group scales using scaled_quantize."""
        skip_if_platform_unsupported("static_scaled_fp8_quant")
        # Normalize group_shape (-1 means full extent)
        norm_group_m = num_tokens if group_shape[0] == -1 else group_shape[0]
        norm_group_n = hidden_size if group_shape[1] == -1 else group_shape[1]

        # Skip if sizes are not divisible by group shape
        if num_tokens % norm_group_m != 0 or hidden_size % norm_group_n != 0:
            pytest.skip(
                f"Skipping: ({num_tokens}, {hidden_size}) not divisible by "
                f"group_shape ({group_shape[0]}, {group_shape[1]})"
            )

        set_random_seed(seed)

        input = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
        ref_out, scale = scaled_quantize(
            input,
            group_shape,
            FP8_DTYPE,
            compute_dtype=torch.float32,
        )
        ops_out = torch.empty(ref_out.shape, device=ref_out.device, dtype=ref_out.dtype)
        static_scaled_fp8_quant_dispatch(ops_out, input, scale, group_shape)

        torch.testing.assert_close(
            ref_out.float(), ops_out.float(), rtol=1.2e-1, atol=1e-5
        )

    @pytest.mark.parametrize("num_tokens", NUM_TOKENS_GROUP)
    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES_GROUP)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize(
        "group_shape", [(1, -1), (-1, 1)]
    )  # per-token, per-channel
    def test_static_fp8_quant_1d_scale(
        self,
        num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        seed: int,
        group_shape: tuple[int, int],
    ) -> None:
        """Test static FP8 quantization with 1D scale (per-token or per-channel)."""
        skip_if_platform_unsupported("static_scaled_fp8_quant")
        set_random_seed(seed)

        input = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
        ref_out, scale_2d = scaled_quantize(
            input, group_shape, FP8_DTYPE, compute_dtype=torch.float32
        )

        # Flatten scale to 1D for testing 1D scale path
        scale_1d = scale_2d.flatten()
        ops_out = torch.empty(ref_out.shape, device=ref_out.device, dtype=ref_out.dtype)
        static_scaled_fp8_quant_dispatch(ops_out, input, scale_1d, group_shape)

        torch.testing.assert_close(
            ref_out.float(), ops_out.float(), rtol=0.12, atol=0.0
        )


class TestStaticScaledFp8QuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "static_scaled_fp8_quant" in registered_kernels

        kernel_wrapper = registered_kernels["static_scaled_fp8_quant"]
        assert kernel_wrapper.op_name == "static_scaled_fp8_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("static_scaled_fp8_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["static_scaled_fp8_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 1, -1)
        assert fake_impl(*args) is None
