# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_dynamic_per_token_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_and_mul_dynamic_per_token_quant import (
    _pick_cache,
    baseline,
    pick_config,
    silu_and_mul_dynamic_per_token_quant,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(num_tokens: int, intermediate_size: int) -> tuple[Any, ...]:
    with FakeTensorMode():
        in_dtype: torch.dtype = torch.bfloat16
        out_dtype: torch.dtype = current_platform.fp8_dtype()
        scale_dtype: torch.dtype = torch.float32
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=scale_dtype)
        scale_ub = torch.mean(input).to(scale_dtype)
        args = (result, input, scale, scale_ub)
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluAndMulDynamicPerTokenQuantConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"intermediate_size": 4096, "num_tokens": 16})

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"intermediate_size": 2048, "num_tokens": 32})

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "num_tokens": 16}),
        ]

        args = _generate_fake_input(32, 8192)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"intermediate_size": 4096, "num_tokens": 16})


class TestSiluAndMulDynamicPerTokenQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("intermediate_size", [17, 1024, 1025, 1026, 5137, 8193])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_dynamic_per_token_quant(
        self,
        num_tokens: int,
        intermediate_size: int,
        dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_dynamic_per_token_quant")
        set_random_seed(seed)

        ref_x = (
            torch.rand(num_tokens, 2 * intermediate_size, dtype=dtype, device="cuda")
            + 1e-6
        )
        ops_x = ref_x.clone()

        scale_ub = (
            torch.mean(SiluAndMul.forward_native(ref_x)).to(
                dtype=torch.float32, device="cuda"
            )
            if has_scale_ub
            else None
        )

        ref_out = torch.empty(
            num_tokens, intermediate_size, device="cuda", dtype=FP8_DTYPE
        )
        ref_scales = torch.empty(
            (ref_x.shape[0], 1), device="cuda", dtype=torch.float32
        )
        baseline(ref_out, ref_x, ref_scales, scale_ub)

        ops_out = torch.empty(
            num_tokens, intermediate_size, device="cuda", dtype=FP8_DTYPE
        )
        ops_scales = torch.empty(
            (ref_x.shape[0], 1), device="cuda", dtype=torch.float32
        )
        silu_and_mul_dynamic_per_token_quant(ops_out, ops_x, ops_scales, scale_ub)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestSiluAndMulDynamicPerTokenQuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_and_mul_dynamic_per_token_quant" in registered_kernels

        kernel_wrapper = registered_kernels["silu_and_mul_dynamic_per_token_quant"]
        assert kernel_wrapper.op_name == "silu_and_mul_dynamic_per_token_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "input", "scale"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("silu_and_mul_dynamic_per_token_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_and_mul_dynamic_per_token_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096)
        assert fake_impl(*args) is None
