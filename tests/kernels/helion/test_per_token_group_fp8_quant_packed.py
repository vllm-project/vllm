# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per_token_group_fp8_quant_packed Helion kernel."""

from itertools import product
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.per_token_group_fp8_quant_packed import (
    _pick_cache,
    baseline,
    per_token_group_fp8_quant_packed,
    pick_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install helion",
        allow_module_level=True,
    )


_GRID_SHAPES = list(
    product(
        [4, 16, 64, 256, 1024, 2048, 8192],
        [2048, 4096, 5120],
    )
)


def _make_outputs(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    output_q_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_q_num_tokens = output_q_num_tokens or num_tokens
    output_q = torch.empty(
        (output_q_num_tokens, hidden_size), device="cuda", dtype=FP8_DTYPE
    )
    groups_per_row = hidden_size // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
    output_s_packed = torch.empty_strided(
        (num_tokens, k_num_packed),
        (1, tma_aligned_num_tokens),
        device="cuda",
        dtype=torch.int32,
    )
    return output_q, output_s_packed


def _make_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int = 128,
    output_q_num_tokens: int | None = None,
) -> tuple[Any, ...]:
    input = (
        torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) * 8
    )
    output_q, output_s_packed = _make_outputs(
        num_tokens, hidden_size, group_size, output_q_num_tokens
    )
    fp8_min, fp8_max = get_fp8_min_max()
    return (
        input,
        output_q,
        output_s_packed,
        group_size,
        1e-10,
        fp8_min,
        fp8_max,
    )


def _physical_scale_storage(scale: torch.Tensor) -> torch.Tensor:
    num_tokens, k_num_packed = scale.shape
    numel = num_tokens + (k_num_packed - 1) * scale.stride(1)
    return torch.as_strided(scale, (numel,), (1,))


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestPerTokenGroupFp8QuantPackedConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_buckets_tokens_after_matching_dimensions(self):
        config_keys = [
            CaseKey({"hidden_size": hidden, "group_size": 128, "num_tokens": tokens})
            for hidden in [2048, 4096]
            for tokens in [16, 32]
        ]

        with FakeTensorMode():
            args = _make_args(20, 4096)

        assert pick_config(args, config_keys) == CaseKey(
            {"hidden_size": 4096, "group_size": 128, "num_tokens": 32}
        )


class TestPerTokenGroupFp8QuantPackedCorrectness:
    @pytest.mark.parametrize(("num_tokens", "hidden_size"), _GRID_SHAPES)
    def test_registered_shape_matches_torch_reference(
        self, num_tokens: int, hidden_size: int
    ):
        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")
        torch.manual_seed(42)
        args = _make_args(num_tokens, hidden_size)
        ref_q, ref_s = _make_outputs(num_tokens, hidden_size, 128)
        ref_args = (args[0], ref_q, ref_s, *args[3:])

        baseline(*ref_args)
        per_token_group_fp8_quant_packed(*args)

        torch.testing.assert_close(args[2], ref_s, rtol=0, atol=0)
        quant_diff = (
            args[1].view(torch.uint8).to(torch.int16)
            - ref_q.view(torch.uint8).to(torch.int16)
        ).abs()
        assert quant_diff.max() <= 1

    def test_zero_fills_padded_output_and_scale_storage(self):
        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")
        torch.manual_seed(42)
        num_tokens, hidden_size, output_q_num_tokens = 3, 640, 4
        args = _make_args(num_tokens, hidden_size, 128, output_q_num_tokens)
        ref_q, ref_s = _make_outputs(num_tokens, hidden_size, 128, output_q_num_tokens)
        ref_args = (args[0], ref_q, ref_s, *args[3:])

        args[1].view(torch.uint8).fill_(0xFF)
        ref_q.view(torch.uint8).fill_(0xFF)
        _physical_scale_storage(args[2]).fill_(0x7F7F7F7F)
        _physical_scale_storage(ref_s).fill_(0x7F7F7F7F)

        baseline(*ref_args)
        per_token_group_fp8_quant_packed(*args)

        torch.testing.assert_close(
            _physical_scale_storage(args[2]),
            _physical_scale_storage(ref_s),
            rtol=0,
            atol=0,
        )
        quant_diff = (
            args[1].view(torch.uint8).to(torch.int16)
            - ref_q.view(torch.uint8).to(torch.int16)
        ).abs()
        assert quant_diff.max() <= 1


class TestPerTokenGroupFp8QuantPackedIntegration:
    def test_registration_and_fake_impl(self):
        from vllm.kernels.helion.register import get_registered_kernels

        wrapper = get_registered_kernels()["per_token_group_fp8_quant_packed"]
        assert wrapper._mutates_args == ["output_q", "output_s_packed"]

        with FakeTensorMode():
            args = _make_args(16, 4096)
            assert wrapper._fake_impl(*args) is None
