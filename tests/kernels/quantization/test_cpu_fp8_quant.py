# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU static and dynamic per-token FP8 quant ops.

Run `pytest tests/kernels/quantization/test_cpu_fp8_quant.py -v`.
"""

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import FP8_DTYPE, ref_dynamic_per_token_quant
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
NUM_TOKENS = [1, 7, 128]
HIDDEN_SIZES = [17, 256, 1025]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("do_scale_ub", [True, False])
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, do_scale_ub: bool
) -> None:
    set_random_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype) + 1e-6
    scale_ub = torch.mean(x).to(torch.float32) if do_scale_ub else None

    ref_out, ref_scales = ref_dynamic_per_token_quant(x, FP8_DTYPE, scale_ub)
    out, scales = ops.scaled_fp8_quant(
        x, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )

    torch.testing.assert_close(ref_scales, scales)
    torch.testing.assert_close(ref_out.float(), out.float())


@pytest.mark.parametrize("hidden_size", [256, 1025])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant_token_padding(
    hidden_size: int, dtype: torch.dtype
) -> None:
    """Padded output rows must not disturb the quantized real rows."""
    set_random_seed(0)
    x = torch.rand(5, hidden_size, dtype=dtype) + 1e-6
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, FP8_DTYPE)
    out, scales = ops.scaled_fp8_quant(
        x, use_per_token_if_dynamic=True, num_token_padding=16
    )

    assert out.shape[0] == 16
    torch.testing.assert_close(ref_scales, scales[:5])
    torch.testing.assert_close(ref_out.float(), out[:5].float())


@pytest.mark.parametrize("num_tokens", [1, 7, 128])
@pytest.mark.parametrize("hidden_size", [256, 1024])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "group_shape", [(-1, -1), (-1, 1), (1, -1), (1, 128), (-1, 128)]
)
@torch.inference_mode()
def test_static_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    group_shape: tuple[int, int],
) -> None:
    set_random_seed(0)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype)
    ref_out, scale = scaled_quantize(
        x, GroupShape(*group_shape), FP8_DTYPE, compute_dtype=torch.float32
    )
    if group_shape in [(-1, 1), (1, -1)]:
        scale = scale.flatten()

    out, out_scale = ops.scaled_fp8_quant(x, scale=scale, group_shape=group_shape)

    torch.testing.assert_close(scale, out_scale)
    torch.testing.assert_close(ref_out.float(), out.float(), rtol=0.12, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_static_fp8_quant_saturates_per_tensor(dtype: torch.dtype) -> None:
    """Values beyond the FP8 range must clamp rather than become NaN."""
    x = torch.tensor([[1000.0, -1000.0, 1.0, 0.0]], dtype=dtype)
    scale = torch.tensor([1.0], dtype=torch.float32)

    out, _ = ops.scaled_fp8_quant(x, scale=scale)

    fp8_max = torch.finfo(FP8_DTYPE).max
    assert out.float().tolist() == [[fp8_max, -fp8_max, 1.0, 0.0]]
