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


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_fp8_quant_rejects_empty_hidden_size(dtype: torch.dtype) -> None:
    x = torch.empty(4, 0, dtype=dtype)
    scale = torch.ones(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-empty hidden size"):
        ops.scaled_fp8_quant(x, scale=scale)
    with pytest.raises(RuntimeError, match="non-empty hidden size"):
        ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)


@pytest.mark.parametrize("scale_shape", [(0, 32), (1, 0), (0, 0)])
@torch.inference_mode()
def test_static_fp8_quant_rejects_empty_2d_scale(
    scale_shape: tuple[int, int],
) -> None:
    x = torch.rand(4, 32)
    scale = torch.empty(scale_shape, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="empty dimension"):
        ops.scaled_fp8_quant(x, scale=scale)


@pytest.mark.parametrize("group_shape", [(0, 1), (1, 0)])
@torch.inference_mode()
def test_static_fp8_quant_rejects_zero_group_shape(
    group_shape: tuple[int, int],
) -> None:
    x = torch.rand(4, 32)
    scale = torch.ones(32, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="group_shape entries"):
        ops.scaled_fp8_quant(x, scale=scale, group_shape=group_shape)


@torch.inference_mode()
def test_static_fp8_quant_rejects_full_extent_group_shape_on_empty_batch() -> None:
    x = torch.empty(0, 32)
    scale = torch.ones(32, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="group_shape entries"):
        ops.scaled_fp8_quant(x, scale=scale, group_shape=(-1, 1))


@torch.inference_mode()
def test_static_fp8_quant_rejects_non_collapsible_leading_dims() -> None:
    scale = torch.ones(1, dtype=torch.float32)

    # A batch stride of 0 makes rows 3..5 lie past the 3-row storage.
    x = torch.rand(1, 3, 32).expand(2, 3, 32)
    out = torch.empty(6, 32, dtype=FP8_DTYPE)
    with pytest.raises(RuntimeError, match="uniformly strided"):
        torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)

    out = torch.empty(1, 3, 32, dtype=FP8_DTYPE).expand(2, 3, 32)
    with pytest.raises(RuntimeError, match="uniformly strided"):
        torch.ops._C.static_scaled_fp8_quant(out, torch.rand(2, 3, 32), scale, None)


@torch.inference_mode()
def test_static_fp8_quant_collapsible_leading_dims_control() -> None:
    x = torch.rand(2, 3, 32)
    scale = torch.tensor([0.5], dtype=torch.float32)
    out = torch.empty(2, 3, 32, dtype=FP8_DTYPE)

    torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)

    ref, _ = ops.scaled_fp8_quant(x.reshape(6, 32), scale=scale)
    assert torch.equal(out.reshape(6, 32).float(), ref.float())


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("permuted_input", [False, True])
@pytest.mark.parametrize("permuted_output", [False, True])
@torch.inference_mode()
def test_fp8_quant_singleton_stride_control(
    dynamic: bool, permuted_input: bool, permuted_output: bool
) -> None:
    set_random_seed(0)
    x = torch.rand(2, 3, 32, 1).permute(0, 3, 1, 2)
    if not permuted_input:
        x = x.reshape(2, 1, 3, 32).clone(memory_format=torch.contiguous_format)
    out = torch.empty(2, 3, 32, 1, dtype=FP8_DTYPE).permute(0, 3, 1, 2)
    if not permuted_output:
        out = torch.empty(2, 1, 3, 32, dtype=FP8_DTYPE)

    if dynamic:
        scale = torch.empty(6, 1)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
        ref, ref_scale = ops.scaled_fp8_quant(
            x.reshape(6, 32), use_per_token_if_dynamic=True
        )
        torch.testing.assert_close(scale, ref_scale)
    else:
        scale = torch.tensor([0.5])
        torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)
        ref, _ = ops.scaled_fp8_quant(x.reshape(6, 32), scale=scale)
    torch.testing.assert_close(out.reshape(6, 32).float(), ref.float())


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("padded_output", [False, True])
@torch.inference_mode()
def test_fp8_quant_empty_batch_control(dynamic: bool, padded_output: bool) -> None:
    x = torch.empty(2, 0, 32)
    shape = (2, 3, 32) if padded_output else (2, 0, 32)
    out = torch.ones(shape, dtype=FP8_DTYPE)
    scale = torch.ones(6, 1) if dynamic else torch.ones(1)

    if dynamic:
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
    else:
        torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)

    assert torch.all(out.float() == 1)
    assert torch.all(scale == 1)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("row_stride", [0, 16])
@torch.inference_mode()
def test_fp8_quant_rejects_overlapping_output_rows(
    dynamic: bool, row_stride: int
) -> None:
    x = torch.rand(4, 32)
    out = torch.empty(128, dtype=FP8_DTYPE).as_strided((4, 32), (row_stride, 1))
    scale = torch.empty(4, 1) if dynamic else torch.ones(1)

    with pytest.raises(RuntimeError, match="must not overlap"):
        if dynamic:
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
        else:
            torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)


@torch.inference_mode()
def test_scaled_fp8_quant_rejects_expanded_output() -> None:
    x = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1).expand(4, 32)
    out = torch.empty((1, 32), dtype=FP8_DTYPE).expand(4, 32)
    scale = torch.ones(1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="must not overlap"):
        ops.scaled_fp8_quant(x.contiguous(), scale=scale, output=out)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("row_stride", [32, 48])
@torch.inference_mode()
def test_fp8_quant_padded_output_rows_control(dynamic: bool, row_stride: int) -> None:
    set_random_seed(0)
    x = torch.rand(4, 32)
    out = torch.zeros(4 * 48, dtype=FP8_DTYPE).as_strided((4, 32), (row_stride, 1))

    if dynamic:
        scale = torch.empty(4, 1)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
        ref, ref_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        torch.testing.assert_close(scale, ref_scale)
    else:
        scale = torch.tensor([0.5])
        torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)
        ref, _ = ops.scaled_fp8_quant(x, scale=scale)
    assert torch.equal(out.float(), ref.float())


@pytest.mark.parametrize("dynamic", [False, True])
@torch.inference_mode()
def test_fp8_quant_single_token_output_stride_control(dynamic: bool) -> None:
    set_random_seed(0)
    x = torch.rand(1, 32)
    out = torch.empty(32, dtype=FP8_DTYPE).as_strided((1, 32), (0, 1))

    if dynamic:
        scale = torch.empty(1, 1)
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(out, x, scale, None)
        ref, ref_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        torch.testing.assert_close(scale, ref_scale)
    else:
        scale = torch.tensor([0.5])
        torch.ops._C.static_scaled_fp8_quant(out, x, scale, None)
        ref, _ = ops.scaled_fp8_quant(x, scale=scale)
    assert torch.equal(out.float(), ref.float())
