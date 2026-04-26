# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ROCm AITER grouped FP8 quantization path."""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_mi3xx

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and on_mi3xx()),
    reason="ROCm MI300/MI350 grouped-quant tests",
)

GROUP_SIZE = 128
GROUP_QUANT_SHAPES = [(32, 1024), (64, 2048), (128, 4096)]


def _assert_aiter_group_quant_supported() -> None:
    assert is_aiter_found_and_supported(), (
        "AITER grouped FP8 quantization is expected on MI300/MI350 ROCm tests"
    )


def _native_group_quant_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant_op = QuantFP8(
        static=False,
        group_shape=GroupShape(1, GROUP_SIZE),
        column_major_scales=False,
        use_ue8m0=False,
    )
    return quant_op.forward_native(x)


def test_rocm_aiter_group_fp8_quant_fake_implementation():
    _assert_aiter_group_quant_supported()

    x = torch.randn((128, 4096), dtype=torch.bfloat16, device="cuda")
    torch.library.opcheck(
        torch.ops.vllm.rocm_aiter_group_fp8_quant,
        (x, GROUP_SIZE),
        test_utils=("test_faketensor",),
    )


@pytest.mark.parametrize("shape", GROUP_QUANT_SHAPES)
def test_rocm_aiter_group_fp8_quant_matches_native_reference(
    default_vllm_config,
    shape,
):
    _assert_aiter_group_quant_supported()

    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    x_fp8, scales = rocm_aiter_ops.group_fp8_quant(x, GROUP_SIZE)
    ref_fp8, ref_scales = _native_group_quant_reference(x)

    assert x_fp8.shape == x.shape
    assert scales.shape == (shape[0], shape[1] // GROUP_SIZE)
    assert x_fp8.dtype == current_platform.fp8_dtype()
    assert scales.dtype == torch.float32
    torch.testing.assert_close(x_fp8, ref_fp8, atol=0.0, rtol=0.0)
    torch.testing.assert_close(scales, ref_scales, atol=0.0, rtol=0.0)


def test_rocm_aiter_group_fp8_quant_torch_compile_matches_eager():
    _assert_aiter_group_quant_supported()

    def group_fp8_quant_fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return rocm_aiter_ops.group_fp8_quant(x, GROUP_SIZE)

    compiled_fn = torch.compile(
        group_fp8_quant_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False,
    )

    x = torch.randn((128, 4096), dtype=torch.bfloat16, device="cuda")
    eager_fp8, eager_scales = group_fp8_quant_fn(x)
    compiled_fp8, compiled_scales = compiled_fn(x)

    torch.testing.assert_close(compiled_fp8, eager_fp8, atol=0.0, rtol=0.0)
    torch.testing.assert_close(compiled_scales, eager_scales, atol=0.0, rtol=0.0)

    x_2 = torch.randn((128, 4096), dtype=torch.bfloat16, device="cuda")
    eager_fp8_2, eager_scales_2 = group_fp8_quant_fn(x_2)
    compiled_fp8_2, compiled_scales_2 = compiled_fn(x_2)

    torch.testing.assert_close(compiled_fp8_2, eager_fp8_2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(compiled_scales_2, eager_scales_2, atol=0.0, rtol=0.0)


def test_rocm_aiter_group_fp8_quant_rejects_non_128_group_size():
    _assert_aiter_group_quant_supported()

    x = torch.randn((32, 1024), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="Group size must be 128"):
        rocm_aiter_ops.group_fp8_quant(x, 64)


def test_rocm_aiter_group_fp8_quant_rejects_non_divisible_hidden_size():
    _assert_aiter_group_quant_supported()

    x = torch.randn((32, 1025), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="divisible by group size"):
        rocm_aiter_ops.group_fp8_quant(x, GROUP_SIZE)
