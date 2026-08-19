# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the silu_and_mul_per_block_quant helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_per_block_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
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
        if current_platform.is_rocm():
            args = (input, group_size)
        else:
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

    def test_config_picker_single_config(self):
        config_key = CaseKey.default()

        assert pick_config((), [config_key]) is config_key

    @pytest.mark.skipif(
        not current_platform.is_rocm(), reason="ROCm-specific config portfolio"
    )
    @pytest.mark.parametrize(
        "intermediate_size,expected_config_id",
        [(256, 1), (2304, 1), (3072, 0), (25600, 0)],
    )
    def test_rocm_config_id_picker(
        self, intermediate_size: int, expected_config_id: int
    ) -> None:
        config_keys = [CaseKey({"config_id": i}) for i in range(2)]
        args = _generate_fake_input(128, intermediate_size, 128)

        assert pick_config(args, config_keys) == CaseKey(
            {"config_id": expected_config_id}
        )

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
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_per_block_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        is_scale_transposed: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        if has_scale_ub and quant_dtype != FP8_DTYPE:
            # skip
            return

        scale = 1 / hidden_size
        x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda") * scale

        if current_platform.is_rocm():
            if (
                group_size != 128
                or is_scale_transposed
                or dtype not in (torch.bfloat16, torch.float16)
                or quant_dtype != current_platform.fp8_dtype()
                or has_scale_ub
            ):
                pytest.skip("ROCm functional op matches AITER's serving contract")
            from vllm._aiter_ops import rocm_aiter_ops

            ref_out, ref_scales = rocm_aiter_ops.get_act_mul_fused_fp8_group_quant_op()(
                x, group_size
            )
            ops_out, ops_scales = silu_and_mul_per_block_quant(x, group_size)
            torch.testing.assert_close(ref_scales, ops_scales, rtol=0.02, atol=1e-4)
            assert (
                ref_out.view(torch.uint8).to(torch.int16)
                - ops_out.view(torch.uint8).to(torch.int16)
            ).abs().max() <= 1
            return

        if has_scale_ub:
            act = torch.nn.functional.silu(x[:, :hidden_size]) * x[:, hidden_size:]
            act_abs = act.abs().float()
            scale_ub = 0.5 * (act_abs.mean() + act_abs.amax())
        else:
            scale_ub = None

        ref_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)

        if is_scale_transposed:
            ref_scales = torch.empty(
                (hidden_size // group_size, x.shape[0]),
                device="cuda",
                dtype=torch.float32,
            ).t()
        else:
            ref_scales = torch.empty(
                (x.shape[0], hidden_size // group_size),
                device="cuda",
                dtype=torch.float32,
            )

        ops_out = ref_out.clone()
        ops_scales = ref_scales.clone()

        baseline(ref_out, x, ref_scales, group_size, scale_ub, is_scale_transposed)
        silu_and_mul_per_block_quant(
            ops_out, x, ops_scales, group_size, scale_ub, is_scale_transposed
        )

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
        expected_mutations = (
            None
            if current_platform.is_rocm()
            else [
                "out",
                "scales",
            ]
        )
        assert kernel_wrapper._mutates_args == expected_mutations

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        result = fake_impl(*args)
        if current_platform.is_rocm():
            assert result[0].shape == (16, 4096)
            assert result[1].shape == (16, 32)
        else:
            assert result is None
