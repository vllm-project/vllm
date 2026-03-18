# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_dynamic_per_token_quant helion kernel

Run `pytest tests/kernels/helion/test_rms_norm_dynamic_per_token_quant.py`.
"""

from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.rms_norm_dynamic_per_token_quant import (
    baseline,
    pick_config,
    rms_norm_dynamic_per_token_quant,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_input(num_tokens: int, hidden_size: int) -> tuple[Any, ...]:
    input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    result = torch.empty(
        input.shape, device=input.device, dtype=current_platform.fp8_dtype()
    )
    scale = torch.empty((num_tokens, 1), device=input.device, dtype=torch.float32)
    scale_ub = torch.mean(input).to(torch.float32)
    residual = torch.randn_like(input)
    weight = torch.normal(
        mean=1.0,
        std=1.0,
        size=(hidden_size,),
        dtype=input.dtype,
        device=input.device,
    )
    epsilon = 1e-6
    args = (result, input, weight, scale, epsilon, scale_ub, residual)
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestRmsNormDynamicPerTokenQuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_num_tokens_16",
            "hidden_size_4096_num_tokens_16",
        ]

        args = _generate_input(16, 4096)
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

        args = _generate_input(20, 3000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_num_tokens_16",
            "hidden_size_4096_num_tokens_16",
        ]

        args = _generate_input(32, 8192)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_num_tokens_16"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key_4096_bad_key_16",
        ]

        args = _generate_input(16, 4096)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8, current_platform.fp8_dtype()]
VEC_HIDDEN_SIZES = [1024, 1025, 1027, 1029]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5120, 5137]],
    *[(2048, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5137]],
    *[(4096, i) for i in [1, 64, 5137]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
SEEDS = [0]

EPS = 1e-6


class TestRmsNormDynamicPerTokenQuantCorrectness:
    @pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
    @pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
    @pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_rms_norm_dynamic_per_token_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        add_residual: bool,
        has_scale_ub: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("rms_norm_dynamic_per_token_quant")

        set_random_seed(seed)

        if has_scale_ub and quant_dtype != current_platform.fp8_dtype():
            # skip
            return

        scale = 1 / (hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        weight = torch.normal(
            mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=x.device
        )
        residual = torch.randn_like(x) * scale if add_residual else None
        scale_ub = (
            torch.mean(x).to(dtype=torch.float32, device="cuda")
            if has_scale_ub
            else None
        )

        ref_out = torch.empty(x.shape, device=x.device, dtype=quant_dtype)
        ref_scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        ref_residual = residual.clone() if residual is not None else None
        baseline(ref_out, x, weight, ref_scales, EPS, scale_ub, ref_residual)

        ops_out = torch.empty(x.shape, device=x.device, dtype=quant_dtype)
        ops_scales = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        ops_residual = residual.clone() if residual is not None else None
        rms_norm_dynamic_per_token_quant(
            ops_out, x, weight, ops_scales, EPS, scale_ub, ops_residual
        )

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1

        if add_residual:
            torch.testing.assert_close(ref_residual, ops_residual)


class TestRmsNormDynamicPerTokenQuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "rms_norm_dynamic_per_token_quant" in registered_kernels

        kernel_wrapper = registered_kernels["rms_norm_dynamic_per_token_quant"]
        assert kernel_wrapper.op_name == "rms_norm_dynamic_per_token_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "scale", "residual"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("rms_norm_dynamic_per_token_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["rms_norm_dynamic_per_token_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_input(16, 4096)
        assert fake_impl(*args) is None
