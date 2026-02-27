# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.nvfp4.batch_invariant import (
    linear_batch_invariant_nvfp4,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Batch-invariant NVFP4 linear requires Blackwell (sm100+) support.",
        allow_module_level=True,
    )

if not hasattr(tl, "dot_scaled"):
    pytest.skip(
        reason="Installed Triton build does not expose tl.dot_scaled.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SEEDS = [42]
DEVICES = ["cuda:0"]


@pytest.fixture(autouse=True)
def _triton_allocator_for_tma_kernels():
    """tl.make_tensor_descriptor in grouped NVFP4 GEMM needs triton.set_allocator."""
    for device in DEVICES:
        set_triton_allocator(torch.device(device))
    yield


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max()).to(torch.float32)


def _input_global_scale(x: torch.Tensor) -> torch.Tensor:
    return _global_scale(x.reshape(-1, x.shape[-1]))


def _alpha(
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> torch.Tensor:
    return (1.0 / (input_global_scale * weight_global_scale)).to(torch.float32)


def _quantize_weight_for_cutlass(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    weight_global_scale = _global_scale(weight)
    weight_fp4, weight_scale = ops.scaled_fp4_quant(weight, weight_global_scale)
    weight_scale = swizzle_blockscale(weight_scale)
    weight_fp4, weights_padding_cols = pad_nvfp4_weight_for_cutlass(weight_fp4)
    return weight_fp4, weight_scale, weight_global_scale, weights_padding_cols


@dataclass
class _PreparedNvFp4LinearCase:
    weight_fp4: torch.Tensor
    weight_scale: torch.Tensor
    input_global_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    alpha: torch.Tensor
    output_size: int
    weights_padding_cols: int


def _prepare_nvfp4_linear_case(
    x_for_scale: torch.Tensor,
    weight: torch.Tensor,
) -> _PreparedNvFp4LinearCase:
    input_global_scale = _input_global_scale(x_for_scale)
    (
        weight_fp4,
        weight_scale,
        weight_global_scale,
        weights_padding_cols,
    ) = _quantize_weight_for_cutlass(weight)
    return _PreparedNvFp4LinearCase(
        weight_fp4=weight_fp4,
        weight_scale=weight_scale,
        input_global_scale=input_global_scale,
        weight_global_scale=weight_global_scale,
        alpha=_alpha(input_global_scale, weight_global_scale),
        output_size=weight.shape[0],
        weights_padding_cols=weights_padding_cols,
    )


def _run_prepared_nvfp4_linear(
    x: torch.Tensor,
    prepared: _PreparedNvFp4LinearCase,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return linear_batch_invariant_nvfp4(
        input=x,
        weight=prepared.weight_fp4,
        weight_scale=prepared.weight_scale,
        input_global_scale_inv=prepared.input_global_scale,
        alpha=prepared.alpha,
        output_size=prepared.output_size,
        bias=bias,
        weights_padding_cols=prepared.weights_padding_cols,
        quant_backend="cutlass",
    )


def _dequant_ref_output(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    output_size: int,
    weights_padding_cols: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    x_fp4, x_scale = ops.scaled_fp4_quant(x_2d, input_global_scale)
    if weights_padding_cols > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()

    x_dq = dequantize_nvfp4_to_dtype(
        x_fp4,
        x_scale,
        input_global_scale,
        dtype=x.dtype,
        device=x.device,
        block_size=16,
    )
    w_dq = dequantize_nvfp4_to_dtype(
        weight_fp4,
        weight_scale,
        weight_global_scale,
        dtype=x.dtype,
        device=x.device,
        block_size=16,
    )

    out_2d = torch.matmul(x_dq, w_dq.t())
    out_2d = out_2d[:, :output_size].contiguous()
    out = out_2d.reshape(*x.shape[:-1], output_size)
    if bias is not None:
        out = out + bias
    return out


def _dequant_ref_output_for_case(
    x: torch.Tensor,
    prepared: _PreparedNvFp4LinearCase,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return _dequant_ref_output(
        x=x,
        weight_fp4=prepared.weight_fp4,
        weight_scale=prepared.weight_scale,
        input_global_scale=prepared.input_global_scale,
        weight_global_scale=prepared.weight_global_scale,
        output_size=prepared.output_size,
        weights_padding_cols=prepared.weights_padding_cols,
        bias=bias,
    )


def _assert_close_nvfp4(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    atol, rtol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (5e-2, 5e-2)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _run_batch_invariant_case(
    x_single: torch.Tensor,
    x_batch: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[_PreparedNvFp4LinearCase, torch.Tensor, torch.Tensor]:
    prepared = _prepare_nvfp4_linear_case(x_batch, weight)
    out_single = _run_prepared_nvfp4_linear(x_single, prepared, bias)
    out_batch = _run_prepared_nvfp4_linear(x_batch, prepared, bias)
    return prepared, out_single, out_batch


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@pytest.mark.parametrize("out_features", [96, 97])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_matches_dequant_reference(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
    out_features: int,
) -> None:
    set_random_seed(seed)
    in_features = 128

    if input_ndim == 2:
        x = torch.randn((17, in_features), dtype=dtype, device=device)
    else:
        x = torch.randn((4, 5, in_features), dtype=dtype, device=device)

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    prepared = _prepare_nvfp4_linear_case(x, w)

    if out_features % 32 != 0:
        assert prepared.weight_fp4.shape[0] > out_features

    out = _run_prepared_nvfp4_linear(x, prepared, bias)
    ref = _dequant_ref_output_for_case(x, prepared, bias)
    _assert_close_nvfp4(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_large_dims(
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test with dimensions that exercise multiple swizzle blocks (128-row,
    4-column) and multiple K-tiles (BLOCK_SIZE_K=256)."""
    set_random_seed(seed)
    out_features = 256
    in_features = 512
    m = 33

    x = torch.randn((m, in_features), dtype=dtype, device=device)
    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    prepared = _prepare_nvfp4_linear_case(x, w)
    out = _run_prepared_nvfp4_linear(x, prepared, bias)
    ref = _dequant_ref_output_for_case(x, prepared, bias)
    _assert_close_nvfp4(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_batch_size_invariance(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    set_random_seed(seed)
    out_features = 64
    in_features = 128

    if input_ndim == 2:
        x_single = torch.randn((1, in_features), dtype=dtype, device=device)
        x_batch = torch.cat(
            [x_single, torch.randn((7, in_features), dtype=dtype, device=device)], dim=0
        )
    else:
        x_single = torch.randn((1, 4, in_features), dtype=dtype, device=device)
        x_batch = torch.cat(
            [x_single, torch.randn((3, 4, in_features), dtype=dtype, device=device)],
            dim=0,
        )

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)
    prepared, out_single, out_batch = _run_batch_invariant_case(
        x_single=x_single,
        x_batch=x_batch,
        weight=w,
        bias=bias,
    )
    ref_single = _dequant_ref_output_for_case(x_single, prepared, bias)
    ref_batch = _dequant_ref_output_for_case(x_batch, prepared, bias)

    assert torch.equal(out_single[0], out_batch[0])
    _assert_close_nvfp4(out_single, ref_single, dtype)
    _assert_close_nvfp4(out_batch, ref_batch, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_batch_order_invariance(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    set_random_seed(seed)
    out_features = 64
    in_features = 128

    if input_ndim == 2:
        x_batch = torch.randn((17, in_features), dtype=dtype, device=device)
    else:
        x_batch = torch.randn((4, 5, in_features), dtype=dtype, device=device)
    perm = torch.roll(torch.arange(x_batch.shape[0], device=device), shifts=-1)
    x_permuted = x_batch[perm].contiguous()

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    prepared = _prepare_nvfp4_linear_case(x_batch, w)
    out = _run_prepared_nvfp4_linear(x_batch, prepared, bias)
    out_permuted = _run_prepared_nvfp4_linear(x_permuted, prepared, bias)
    ref = _dequant_ref_output_for_case(x_batch, prepared, bias)
    ref_permuted = _dequant_ref_output_for_case(x_permuted, prepared, bias)

    assert torch.equal(out[perm], out_permuted)
    _assert_close_nvfp4(out, ref, dtype)
    _assert_close_nvfp4(out_permuted, ref_permuted, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@pytest.mark.parametrize("in_features", [128, 272])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_crosses_first_m_tile_boundary(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
    in_features: int,
) -> None:
    """Exercise the case where the first prompt moves from a partial M tile
    alone to a full 128-row tile when mixed into a larger batch."""
    set_random_seed(seed)
    out_features = 64

    if input_ndim == 2:
        x_batch = torch.randn((129, in_features), dtype=dtype, device=device)
    else:
        x_batch = torch.randn((3, 64, in_features), dtype=dtype, device=device)
    x_single = x_batch[:1].clone()

    assert x_batch.reshape(-1, in_features).shape[0] > 128

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    prepared, out_single, out_batch = _run_batch_invariant_case(
        x_single=x_single,
        x_batch=x_batch,
        weight=w,
        bias=bias,
    )
    ref_single = _dequant_ref_output_for_case(x_single, prepared, bias)
    ref_batch = _dequant_ref_output_for_case(x_batch, prepared, bias)

    assert torch.equal(out_single[0], out_batch[0])
    _assert_close_nvfp4(out_single, ref_single, dtype)
    _assert_close_nvfp4(out_batch, ref_batch, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_matches_dequant_reference_without_bias(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    set_random_seed(seed)
    out_features = 96
    in_features = 128

    if input_ndim == 2:
        x = torch.randn((17, in_features), dtype=dtype, device=device)
    else:
        x = torch.randn((4, 5, in_features), dtype=dtype, device=device)

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)

    prepared = _prepare_nvfp4_linear_case(x, w)
    out = _run_prepared_nvfp4_linear(x, prepared, bias=None)
    ref = _dequant_ref_output_for_case(x, prepared, bias=None)

    _assert_close_nvfp4(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("input_ndim", [2, 3])
@torch.inference_mode()
def test_linear_batch_invariant_nvfp4_fused_bias(
    dtype: torch.dtype,
    seed: int,
    device: str,
    input_ndim: int,
) -> None:
    """Verify that the fused HAS_BIAS kernel path matches an unfused
    matmul (bias=None) followed by a manual bias addition."""
    set_random_seed(seed)
    out_features = 96
    in_features = 128

    if input_ndim == 2:
        x = torch.randn((17, in_features), dtype=dtype, device=device)
    else:
        x = torch.randn((4, 5, in_features), dtype=dtype, device=device)

    w = torch.randn((out_features, in_features), dtype=dtype, device=device)
    bias = torch.randn((out_features,), dtype=dtype, device=device)

    prepared = _prepare_nvfp4_linear_case(x, w)

    out_fused = _run_prepared_nvfp4_linear(x, prepared, bias)
    out_no_bias = _run_prepared_nvfp4_linear(x, prepared, bias=None)
    ref_no_bias = _dequant_ref_output_for_case(x, prepared, bias=None)

    # The kernel adds bias in fp32 before the downcast, so build the
    # reference the same way to isolate matmul error from rounding diffs.
    ref_fused = (ref_no_bias.float() + bias.float()).to(dtype)

    _assert_close_nvfp4(out_no_bias, ref_no_bias, dtype)
    _assert_close_nvfp4(out_fused, ref_fused, dtype)
