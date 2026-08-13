# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the block_scaled_mm helion kernel

Run `pytest tests/kernels/helion/test_block_scaled_mm.py`.
"""

from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.utils import opcheck
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.block_scaled_mm import (
    _pick_cache,
    baseline,
    block_scaled_mm,
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

GROUP_M = 1
GROUP_K = 128
GROUP_N = 128


def _generate_input(
    M: int,
    K: int,
    N: int,
    in_dtype: torch.dtype | None = None,
) -> tuple[Any, ...]:
    if in_dtype is None:
        in_dtype = current_platform.fp8_dtype()

    if in_dtype.is_floating_point:
        a = (0.25 * torch.rand((M, K), dtype=torch.float32, device="cuda")).to(in_dtype)
        b = (0.25 * torch.rand((N, K), dtype=torch.float32, device="cuda")).to(in_dtype)
    else:
        a = torch.randint(-32, 32, (M, K), dtype=in_dtype, device="cuda")
        b = torch.randint(-32, 32, (N, K), dtype=in_dtype, device="cuda")
    b = b.t()

    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)

    num_group_m = M // GROUP_M
    num_group_k = K // GROUP_K
    num_group_n = N // GROUP_N

    a_scales = 0.25 * torch.rand(
        num_group_m, num_group_k, dtype=torch.float32, device="cuda"
    )
    b_scales = 0.25 * torch.rand(
        num_group_k, num_group_n, dtype=torch.float32, device="cuda"
    )
    # make scales M-major / K-major for blockwise quant
    a_scales = a_scales.t().contiguous().t()
    b_scales = b_scales.t().contiguous().t()

    return out, a, b, a_scales, b_scales


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestBlockScaledMmConfigPicker:
    def setup_method(self):
        _pick_cache.clear()
        self.fp8 = str(current_platform.fp8_dtype())
        self.int8 = str(torch.int8)

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 4096, "M": 16, "in_dtype": self.fp8}),
        ]

        args = _generate_input(16, 4096, 6144)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 4096, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 6144, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 4096, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 4096, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 6144, "M": 32, "in_dtype": self.fp8}),
        ]

        args = _generate_input(20, 3072, 512)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"K": 2048, "N": 4096, "M": 32, "in_dtype": self.fp8}
        )

    def test_config_picker_matches_in_dtype(self):
        config_keys = [
            CaseKey({"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 4096, "M": 16, "in_dtype": self.int8}),
        ]

        args = _generate_input(16, 4096, 6144, in_dtype=torch.int8)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"K": 2048, "N": 4096, "M": 16, "in_dtype": self.int8}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_no_matching_in_dtype(self):
        config_keys = [
            CaseKey({"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}),
        ]

        args = _generate_input(16, 4096, 6144, in_dtype=torch.int8)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"K": 2048, "N": 4096, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 4096, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 2048, "N": 6144, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 4096, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 4096, "M": 32, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 6144, "M": 16, "in_dtype": self.fp8}),
            CaseKey({"K": 4096, "N": 6144, "M": 32, "in_dtype": self.fp8}),
        ]

        args = _generate_input(64, 8192, 7040)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"K": 4096, "N": 6144, "M": 32, "in_dtype": self.fp8}
        )


# N and K must be multiples of the block scale group sizes (128).
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


class TestBlockScaledMmCorrectness:
    @pytest.mark.parametrize("M,N,K", MNK_FACTORS)
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("in_dtype", [current_platform.fp8_dtype()])
    def test_block_scaled_mm(self, M, N, K, out_dtype, in_dtype):
        skip_if_platform_unsupported("block_scaled_mm")

        set_random_seed(0)

        out, a, b, a_scales, b_scales = _generate_input(M, K, N, in_dtype=in_dtype)
        out = out.to(out_dtype)
        c_actual = torch.empty_like(out)

        block_scaled_mm(out, a, b, a_scales, b_scales)
        baseline(c_actual, a, b, a_scales, b_scales)

        if in_dtype.is_floating_point:
            torch.testing.assert_close(out, c_actual, rtol=1e-1, atol=1e-1)
        else:
            torch.testing.assert_close(out, c_actual, rtol=2e-1, atol=7e-1)


class TestBlockScaledMmIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "block_scaled_mm" in registered_kernels

        kernel_wrapper = registered_kernels["block_scaled_mm"]
        assert kernel_wrapper.op_name == "block_scaled_mm"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["out"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("block_scaled_mm")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["block_scaled_mm"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_input(16, 4096, 4096)
        assert fake_impl(*args) is None

    def test_customop_opcheck(self):
        skip_if_platform_unsupported("block_scaled_mm")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["block_scaled_mm"]

        # opcheck if registered as custom op
        if hasattr(torch.ops.vllm_helion, kernel_wrapper.op_name):
            fn = getattr(torch.ops.vllm_helion, kernel_wrapper.op_name)
            args = _generate_input(16, 4096, 4096)
            opcheck(fn, args)
