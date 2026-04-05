# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_dynamic_per_token_scaled_fp8_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.dynamic_per_token_scaled_fp8_quant import (
    baseline,
    dynamic_per_token_scaled_fp8_quant,
    pick_config,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(num_tokens: int, hidden_size: int) -> tuple[Any, ...]:
    with FakeTensorMode():
        input = torch.randn(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        result = torch.empty(
            input.shape, device=input.device, dtype=current_platform.fp8_dtype()
        )
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=torch.float32)
        scale_ub = torch.mean(input).to(torch.float32)
        args = (result, input, scale, scale_ub)
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestDynamicPerTokenScaledFp8QuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_num_tokens_16",
            "hidden_size_4096_num_tokens_16",
        ]

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_num_tokens_16"

    def test_config_picker_closest_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_num_tokens_16",
            "hidden_size_2048_num_tokens_32",
            "hidden_size_4096_num_tokens_16",
            "hidden_size_4096_num_tokens_32",
        ]

        args = _generate_fake_input(20, 3000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_num_tokens_16",
            "hidden_size_4096_num_tokens_16",
        ]

        args = _generate_fake_input(32, 8192)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_num_tokens_16"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key_4096_bad_key_16",
        ]

        args = _generate_fake_input(16, 4096)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


class TestDynamicPerTokenScaledFp8QuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("hidden_size", [17, 1024, 1025, 1026, 5137, 8193])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_dynamic_per_token_fp8_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("dynamic_per_token_scaled_fp8_quant")
        set_random_seed(seed)

        x = (
            torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") + 1e-6
        )  # avoid nans

        scale_ub = (
            torch.mean(x).to(dtype=torch.float32, device="cuda")
            if has_scale_ub
            else None
        )

        ref_out = torch.empty(x.shape, device="cuda", dtype=FP8_DTYPE)
        ref_scales = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
        baseline(ref_out, x, ref_scales, scale_ub)

        ops_out = torch.empty(x.shape, device="cuda", dtype=FP8_DTYPE)
        ops_scales = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
        dynamic_per_token_scaled_fp8_quant(ops_out, x, ops_scales, scale_ub)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestDynamicPerTokenScaledFp8QuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "dynamic_per_token_scaled_fp8_quant" in registered_kernels

        kernel_wrapper = registered_kernels["dynamic_per_token_scaled_fp8_quant"]
        assert kernel_wrapper.op_name == "dynamic_per_token_scaled_fp8_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "scale"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("dynamic_per_token_scaled_fp8_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["dynamic_per_token_scaled_fp8_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096)
        assert fake_impl(*args) is None
