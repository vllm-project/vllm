# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scaled_mm helion kernel

Run `pytest tests/kernels/helion/test_scaled_mm.py`.
"""

from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.scaled_mm import (
    baseline,
    pick_config,
    scaled_mm,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_input(
    num_tokens: int, hidden_size: int, feature_size: int
) -> tuple[Any, ...]:
    in_dtype = current_platform.fp8_dtype()
    a = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device="cuda").to(
        in_dtype
    )
    b = torch.randn(feature_size, hidden_size, dtype=torch.float32, device="cuda").to(
        in_dtype
    )
    b = b.t()
    scale_a = torch.randn(num_tokens, 1, dtype=torch.float32, device="cuda")
    scale_b = torch.randn(feature_size, 1, dtype=torch.float32, device="cuda")
    bias = torch.randn(feature_size, dtype=torch.bfloat16, device="cuda")
    out_dtype = torch.bfloat16

    args = (a, b, scale_a, scale_b, out_dtype, bias)
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestScaledMmConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
        ]

        args = _generate_input(16, 4096, 6144)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_feature_size_6144_num_tokens_16"

    def test_config_picker_closest_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_2048_feature_size_4096_num_tokens_32",
            "hidden_size_2048_feature_size_6144_num_tokens_16",
            "hidden_size_2048_feature_size_6144_num_tokens_32",
            "hidden_size_4096_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_4096_num_tokens_32",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_32",
        ]

        args = _generate_input(20, 3000, 500)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_feature_size_4096_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_2048_feature_size_4096_num_tokens_32",
            "hidden_size_2048_feature_size_6144_num_tokens_16",
            "hidden_size_2048_feature_size_6144_num_tokens_32",
            "hidden_size_4096_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_4096_num_tokens_32",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_32",
        ]

        args = _generate_input(64, 8192, 7000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_feature_size_6144_num_tokens_32"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key",
        ]

        args = _generate_input(16, 4096, 4096)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


def _get_8bit_types():
    types = [torch.int8]
    if current_platform.supports_fp8():
        types.append(current_platform.fp8_dtype())
    return types


MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


class TestScaledMmCorrectness:
    @pytest.mark.parametrize("M,N,K", MNK_FACTORS)
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("in_dtype", _get_8bit_types())
    @pytest.mark.parametrize("use_scalar_scale_a", [True, False])
    @pytest.mark.parametrize("use_scalar_scale_b", [True, False])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_scaled_mm(
        self,
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        use_scalar_scale_a,
        use_scalar_scale_b,
        use_bias,
    ):
        skip_if_platform_unsupported("scaled_mm")
        is_floating_point_type = lambda t: torch.tensor(
            [1, 1], dtype=t
        ).is_floating_point()

        set_random_seed(0)

        # NOTE: There are cases, where if the matrix is large enough, an output
        # like 65504.4 can be produced, and can easily turn into inf when
        # multiplied when using float16/bfloat16.  This means one function, e.g.,
        # testing function, and another function, e.g. golden function, can
        # produce a non-inf value while the other produces an inf value, and
        # will cause assert_close/allclose to fail, even though if overflow
        # wouldn't have occurred, the values would have been "close."
        #
        # So, the values here are kept small enough to avoid this situation.
        if is_floating_point_type(in_dtype):
            a = (0.25 * torch.rand((M, K), dtype=torch.float32, device="cuda")).to(
                in_dtype
            )
            b = (0.25 * torch.rand((K, N), dtype=torch.float32, device="cuda")).to(
                in_dtype
            )
        else:
            a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device="cuda")
            b = torch.randint(-32, 32, (K, N), dtype=in_dtype, device="cuda")

        if use_scalar_scale_a:
            scale_a = torch.rand((1, 1), device=a.device)
        else:
            scale_a = 0.25 * torch.rand((M, 1), device=a.device)

        if use_scalar_scale_b:
            scale_b = torch.rand((1, 1), device=b.device)
        else:
            scale_b = 0.25 * torch.rand((N, 1), device=b.device)

        bias = None
        if use_bias:
            bias = torch.rand((N,), device=a.device, dtype=out_dtype)

        c_check = scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

        c_actual = baseline(a, b, scale_a, scale_b, out_dtype, bias)

        torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)


class TestScaledMmIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "scaled_mm" in registered_kernels

        kernel_wrapper = registered_kernels["scaled_mm"]
        assert kernel_wrapper.op_name == "scaled_mm"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args is None

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("scaled_mm")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["scaled_mm"]
        fake_impl = kernel_wrapper._fake_impl

        a, b, scale_a, scale_b, out_dtype, bias = _generate_input(16, 4096, 4096)
        fake_output = fake_impl(a, b, scale_a, scale_b, out_dtype, bias)

        assert fake_output.shape[0] == a.shape[0]
        assert fake_output.shape[1] == b.shape[1]
        assert fake_output.dtype == out_dtype
        assert fake_output.device == a.device
