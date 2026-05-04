# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_per_block_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.kernels.helion.case_key import CaseKey

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_and_mul_per_block_quant import (
    _pick_cache,
    baseline,
    pick_config,
    silu_and_mul_per_block_quant,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(
    num_tokens: int, intermediate_size: int, group_size: int
) -> tuple[Any, ...]:
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
        scale = torch.empty(
            (num_tokens, intermediate_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        scale_ub = torch.mean(input).to(scale_dtype)
        args = (
            result,
            input,
            scale,
            group_size,
            scale_ub,
            False,
        )
        return args


class TestSiluAndMulPerBlockQuantConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluAndMulPerBlockQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("hidden_size", [1024, 2048, 5120])
    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("quant_dtype", [current_platform.fp8_dtype(), torch.int8])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_per_block_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        is_scale_transposed: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        scale = 1 / hidden_size
        x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda") * scale

        ref_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)
        ref_scales = torch.empty(
            (x.shape[0], hidden_size // group_size), device="cuda", dtype=torch.float32
        )
        baseline(ref_out, x, ref_scales, group_size, None, False)

        ops_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)
        ops_scales = torch.empty(
            (x.shape[0], hidden_size // group_size), device="cuda", dtype=torch.float32
        )
        silu_and_mul_per_block_quant(ops_out, x, ops_scales, group_size, None, False)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestSiluAndMulPerBlockQuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_and_mul_per_block_quant" in registered_kernels

        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant"]
        assert kernel_wrapper.op_name == "silu_and_mul_per_block_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["out", "scales"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
