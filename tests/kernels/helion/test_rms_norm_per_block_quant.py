# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_per_block_quant helion kernel

Run `pytest tests/kernels/helion/test_rms_norm_per_block_quant.py`.
"""

import itertools
from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.rms_norm_per_block_quant import (
    baseline,
    pick_config,
    rms_norm_per_block_quant,
)
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_input(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[Any, ...]:
    input = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    result = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
    scale = torch.empty(
        (num_tokens, hidden_size // group_size),
        device=input.device,
        dtype=torch.float32,
    )
    scale_ub = torch.mean(input).to(scale.dtype)
    residual = torch.randn_like(input)
    weight = torch.normal(
        mean=1.0,
        std=1.0,
        size=(hidden_size,),
        dtype=input.dtype,
        device=input.device,
    )
    epsilon = 1e-6
    args = (
        result,
        input,
        weight,
        scale,
        epsilon,
        scale_ub,
        residual,
        group_size,
        False,
    )
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestRmsNormPerBlockQuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_size_64_num_tokens_16",
            "hidden_size_4096_group_size_128_num_tokens_16",
        ]

        args = _generate_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
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

        args = _generate_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_group_size_64_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
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

        args = _generate_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_group_size_128_num_tokens_32"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key",
        ]

        args = _generate_input(16, 4096, 128)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8, FP8_DTYPE]
VEC_HIDDEN_SIZES = [64, 1024]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, 128, 1024, 5120]],
    *[(2048, i) for i in [64, 1024]],
    *[(4096, i) for i in [64]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
GROUP_SIZES = [64, 128]
TMA_ALIGNMENTS = [0, 4]
SEEDS = [0]
EPS = 1e-6


class TestRmsNormPerBlockQuantCorrectness:
    @pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
    @pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
    @pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    @pytest.mark.parametrize(
        "group_size, tma_alignment",
        [*itertools.product(GROUP_SIZES, TMA_ALIGNMENTS)],
    )
    @pytest.mark.parametrize("seed", SEEDS)
    def test_rms_norm_per_block_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        add_residual: bool,
        has_scale_ub: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        is_scale_transposed: bool,
        group_size: int,
        tma_alignment: int,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("rms_norm_per_block_quant")

        set_random_seed(seed)

        if hidden_size % group_size != 0:
            # skip
            return

        if tma_alignment != 0 and hidden_size // group_size % tma_alignment == 0:
            # Skip tests where TMA alignment doesn't create extra padding to save time
            return

        if has_scale_ub and quant_dtype != FP8_DTYPE:
            # skip
            return

        scale = 1 / (hidden_size)
        input = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        weight = torch.normal(
            mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=input.device
        )
        residual = torch.randn_like(input) * scale if add_residual else None
        scale_ub = (
            torch.mean(input).to(dtype=torch.float32, device="cuda")
            if has_scale_ub
            else None
        )
        groups_per_row = hidden_size // group_size

        ref_residual = residual.clone() if residual is not None else None
        ops_residual = residual.clone() if residual is not None else None
        ref_out = torch.empty(input.shape, device=input.device, dtype=quant_dtype)
        ops_out = ref_out.clone()

        if is_scale_transposed:
            if tma_alignment == 0:
                ref_scales = torch.empty(
                    (groups_per_row, num_tokens),
                    device=input.device,
                    dtype=torch.float32,
                ).transpose(0, 1)
            else:
                tma_aligned_m = (
                    (num_tokens + tma_alignment - 1) // tma_alignment * tma_alignment
                )
                shape = (num_tokens, groups_per_row)
                stride = (1, tma_aligned_m)
                ref_scales = torch.empty_strided(
                    shape, stride, device=input.device, dtype=torch.float32
                )
        else:
            ref_scales = torch.empty(
                (num_tokens, groups_per_row), device=input.device, dtype=torch.float32
            )

        ops_scales = ref_scales.clone()

        baseline(
            ref_out,
            input,
            weight,
            ref_scales,
            EPS,
            scale_ub,
            ref_residual,
            group_size,
            is_scale_transposed,
        )
        ref_scales = ref_scales.contiguous()

        rms_norm_per_block_quant(
            ops_out,
            input,
            weight,
            ops_scales,
            EPS,
            scale_ub,
            ops_residual,
            group_size,
            is_scale_transposed,
        )
        ops_scales = ops_scales.contiguous()

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1

        if add_residual:
            torch.testing.assert_close(ref_residual, ops_residual)


class TestRmsNormPerBlockQuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "rms_norm_per_block_quant" in registered_kernels

        kernel_wrapper = registered_kernels["rms_norm_per_block_quant"]
        assert kernel_wrapper.op_name == "rms_norm_per_block_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "scale", "residual"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("rms_norm_per_block_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["rms_norm_per_block_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_input(16, 4096, 128)
        assert fake_impl(*args) is None
