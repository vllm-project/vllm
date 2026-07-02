# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scaled_mm helion kernel

Run `pytest tests/kernels/helion/test_scaled_mm_blockwise.py`.
"""

from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.scaled_mm_blockwise import (
    _pick_cache,
    baseline,
    pick_config,
    scaled_mm_blockwise,
)
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_input(M: int, K: int, N: int) -> tuple[Any, ...]:
    in_dtype = FP8_DTYPE
    a = torch.randn(M, K, dtype=torch.float32, device="cuda").to(in_dtype)
    b = torch.randn(N, K, dtype=torch.float32, device="cuda").to(in_dtype)
    b = b.t()
    group_m = 1
    group_k = 128
    group_n = 128
    num_group_m = M // group_m
    num_group_k = K // group_k
    num_group_n = N // group_n

    scale_a = torch.randn(num_group_m, num_group_k, dtype=torch.float32, device="cuda")
    scale_b = torch.randn(num_group_k, num_group_n, dtype=torch.float32, device="cuda")
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    out_dtype = torch.bfloat16

    args = (a, b, scale_a, scale_b, group_m, group_k, group_n, out_dtype, bias)
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestScaledMmBlockwiseConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16}),
            CaseKey({"K": 4096, "N": 6144, "M": 16}),
        ]

        args = _generate_input(16, 4096, 6144)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"K": 4096, "N": 6144, "M": 16})

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16}),
            CaseKey({"K": 2048, "N": 4096, "M": 32}),
            CaseKey({"K": 2048, "N": 6144, "M": 16}),
            CaseKey({"K": 2048, "N": 6144, "M": 32}),
            CaseKey({"K": 4096, "N": 4096, "M": 16}),
            CaseKey({"K": 4096, "N": 4096, "M": 32}),
            CaseKey({"K": 4096, "N": 6144, "M": 16}),
            CaseKey({"K": 4096, "N": 6144, "M": 32}),
        ]

        args = _generate_input(20, 3000, 500)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"K": 2048, "N": 4096, "M": 32})

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16}),
            CaseKey({"K": 2048, "N": 4096, "M": 32}),
            CaseKey({"K": 2048, "N": 6144, "M": 16}),
            CaseKey({"K": 2048, "N": 6144, "M": 32}),
            CaseKey({"K": 4096, "N": 4096, "M": 16}),
            CaseKey({"K": 4096, "N": 4096, "M": 32}),
            CaseKey({"K": 4096, "N": 6144, "M": 16}),
            CaseKey({"K": 4096, "N": 6144, "M": 32}),
        ]

        args = _generate_input(64, 8192, 7000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey({"K": 4096, "N": 6144, "M": 32})


MNK_FACTORS = [
    (1, 256, 128),
    (1, 16384, 1024),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 16384, 1024),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]


class TestScaledMmBlockwiseCorrectness:
    @pytest.mark.parametrize("M,N,K", MNK_FACTORS)
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("in_dtype", [FP8_DTYPE, torch.int8])
    @pytest.mark.parametrize(
        "scale_a_group_shape,scale_b_group_shape", [((1, 128), (128, 128))]
    )
    @pytest.mark.parametrize("use_bias", [False])
    def test_scaled_mm_blockwise_fp8(
        self,
        M,
        N,
        K,
        out_dtype,
        in_dtype,
        scale_a_group_shape,
        scale_b_group_shape,
        use_bias,
    ):
        skip_if_platform_unsupported("scaled_mm_blockwise")
        set_random_seed(0)
        group_m, group_k = scale_a_group_shape
        group_n = scale_b_group_shape[1]

        if K % scale_b_group_shape[0] != 0 or N % scale_b_group_shape[1] != 0:
            return
        if M % scale_a_group_shape[0] != 0 or K % scale_a_group_shape[1] != 0:
            return

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
                FP8_DTYPE
            )
            b = (0.25 * torch.rand((N, K), dtype=torch.float32, device="cuda")).to(
                FP8_DTYPE
            )
        else:
            a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device="cuda")
            b = torch.randint(-32, 32, (N, K), dtype=in_dtype, device="cuda")

        b = b.t()

        num_group_m = M // group_m
        num_group_k = K // group_k
        num_group_n = N // group_n

        scale_a = 0.25 * torch.rand(
            (num_group_m, num_group_k), dtype=torch.float32, device=a.device
        )
        scale_b = 0.25 * torch.rand(
            (num_group_k, num_group_n), dtype=torch.float32, device=b.device
        )

        # make scales M-major for blockwise quant
        scale_a = scale_a.t().contiguous().t()
        # make scales K-major for blockwise quant
        scale_b = scale_b.t().contiguous().t()

        bias = None
        if use_bias:
            bias = torch.rand((N,), device=a.device, dtype=out_dtype)

        c_check = scaled_mm_blockwise(
            a, b, scale_a, scale_b, group_m, group_k, group_n, out_dtype, bias
        )

        c_actual = baseline(
            a, b, scale_a, scale_b, group_m, group_k, group_n, out_dtype, bias
        )

        if is_floating_point_type(in_dtype):
            torch.testing.assert_close(c_check, c_actual, rtol=1e-1, atol=1e-1)
        else:
            torch.testing.assert_close(c_check, c_actual, rtol=2e-1, atol=7e-1)


class TestScaledMmBlockwiseIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "scaled_mm_blockwise" in registered_kernels

        kernel_wrapper = registered_kernels["scaled_mm_blockwise"]
        assert kernel_wrapper.op_name == "scaled_mm_blockwise"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args is None

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("scaled_mm_blockwise")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["scaled_mm_blockwise"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_input(16, 4096, 4096)
        fake_output = fake_impl(*args)
        a = args[0]
        b = args[1]
        out_dtype = args[-2]

        assert fake_output.shape[0] == a.shape[0]
        assert fake_output.shape[1] == b.shape[1]
        assert fake_output.dtype == out_dtype
        assert fake_output.device == a.device
