# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_dynamic_per_token_quant helion kernel

Run `pytest tests/kernels/helion/test_per_token_group_fp8_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.per_token_group_fp8_quant import (
    baseline,
    per_token_group_fp8_quant,
    pick_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[Any, ...]:
    with FakeTensorMode():
        input = torch.randn(
            (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        output_q = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
        output_s = torch.empty(
            (num_tokens, hidden_size // group_size),
            device=input.device,
            dtype=torch.float32,
        )
        use_ue8m0 = False
        column_major = False
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10
        args = (
            input,
            output_q,
            output_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major,
        )
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestPerTokenGroupFp8QuantConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_group_size_64_num_tokens_16",
            "hidden_size_4096_group_size_128_num_tokens_16",
        ]

        args = _generate_fake_input(16, 4096, 128)
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

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_group_size_64_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_fake_input(16, 4096, 128)
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

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_group_size_128_num_tokens_32"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key",
        ]

        args = _generate_fake_input(16, 4096, 128)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


class TestPerTokenGroupFp8QuantCorrectness:
    @pytest.mark.parametrize(
        "shape", [(31, 128), (32, 128), (63, 256), (64, 256), (16, 512), (2048, 5120)]
    )
    @pytest.mark.parametrize("column_major", [False, True])
    @pytest.mark.parametrize("tma_aligned", [False, True])
    @pytest.mark.parametrize("scale_ue8m0", [False, True])
    @pytest.mark.parametrize("group_size", [64, 128])
    def test_per_token_group_fp8_quant(
        self,
        shape,
        column_major: bool,
        tma_aligned: bool,
        scale_ue8m0: bool,
        group_size: int,
    ):
        skip_if_platform_unsupported("per_token_group_fp8_quant")

        torch.manual_seed(42)
        num_tokens, hidden_size = shape
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10
        input = (
            torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
            * 8
        )
        ref_q = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
        ops_q = ref_q.clone()

        groups_per_row = hidden_size // group_size
        if column_major:
            if tma_aligned:
                tma_alignment = 4
                tma_aligned_m = (
                    (num_tokens + tma_alignment - 1) // tma_alignment * tma_alignment
                )
                shape = (num_tokens, groups_per_row)
                stride = (1, tma_aligned_m)
                ref_s = torch.empty_strided(
                    shape, stride, device=input.device, dtype=torch.float32
                )
            else:
                ref_s = torch.empty(
                    (groups_per_row, num_tokens),
                    device=input.device,
                    dtype=torch.float32,
                ).transpose(0, 1)
        else:
            ref_s = torch.empty(
                (num_tokens, groups_per_row), device=input.device, dtype=torch.float32
            )

        ops_s = ref_s.clone()

        baseline(
            input,
            ref_q,
            ref_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            column_major,
            tma_aligned,
        )
        per_token_group_fp8_quant(
            input,
            ops_q,
            ops_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            scale_ue8m0,
            column_major,
            tma_aligned,
        )

        assert torch.allclose(ref_s, ops_s)
        # allow 1 ULP difference
        assert (
            ref_q.view(torch.uint8).to(torch.int16)
            - ops_q.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestPerTokenGroupFp8QuantIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "per_token_group_fp8_quant" in registered_kernels

        kernel_wrapper = registered_kernels["per_token_group_fp8_quant"]
        assert kernel_wrapper.op_name == "per_token_group_fp8_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["output_q", "output_s"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("per_token_group_fp8_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["per_token_group_fp8_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
