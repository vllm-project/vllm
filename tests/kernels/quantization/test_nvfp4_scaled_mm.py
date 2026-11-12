# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
    get_nvfp4_global_scale,
)

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)
# M > 256 takes the large-M GEMM config.
SHAPES.append((512, 128, 64))

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_sf, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
    has_bias: bool,
) -> None:
    if has_bias and not current_platform.is_device_capability_family(120):
        pytest.skip("The bias epilogue is implemented for SM12x only.")
    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
    )
    bias = torch.randn(n, dtype=dtype, device=device) if has_bias else None
    if bias is not None:
        expected_out = expected_out + bias
    out = ops.cutlass_scaled_fp4_mm(
        a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype, bias=bias
    )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("dtype", DTYPES)
# m, n, packed k, one per config: M = 150 takes 128x64 tiles with N = 2080
# padding the last 4-column scale tile; M = 600 (128x128 tiles) and M = 2100
# (256x128) pad the last 128-row one.
@pytest.mark.parametrize(
    "shape", [(150, 2080, 256), (600, 1024, 2048), (2100, 1024, 2048)]
)
@torch.inference_mode()
def test_nvfp4_gemm_gelu_quant(dtype: torch.dtype, shape: tuple[int, int, int]) -> None:
    """The epilogue's NVFP4 of gelu(a @ b^T + bias) matches the GEMM, GELU and
    scaled_fp4_quant run apart, except where skipping the 16-bit rounding of the
    GEMM output moves a value to the neighboring FP4 code."""
    if not current_platform.is_device_capability_family(120):
        pytest.skip("The GELU + NVFP4 quant epilogue is implemented for SM12x only.")
    set_random_seed(42)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    device = "cuda:0"
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device) / k**0.5
    a_global_scale = get_nvfp4_global_scale(a_dtype)
    b_global_scale = get_nvfp4_global_scale(b_dtype)
    alpha = 1.0 / (a_global_scale * b_global_scale)
    a_fp4, a_scale = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale = ops.scaled_fp4_quant(b_dtype, b_global_scale)
    bias = torch.randn(n, dtype=dtype, device=device)

    y = torch.nn.functional.gelu(
        ops.cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, a_scale, b_scale, alpha, dtype, bias=bias
        )
    )
    y_global_scale = get_nvfp4_global_scale(y)
    ref, ref_scale = ops.scaled_fp4_quant(y, y_global_scale)

    # Into buffers holding stale bytes, as under torch.compile buffer reuse.
    out = torch.full_like(ref, 0xFF)
    out_scale = torch.empty_like(ref_scale)
    out_scale.view(torch.uint8).fill_(0xFF)
    torch.ops._C.cutlass_scaled_fp4_mm_gelu_nvfp4_quant(
        out, out_scale, a_fp4, b_fp4, a_scale, b_scale, alpha, bias, y_global_scale
    )

    # No stale scale survives in the 128x4 tile padding: the padded columns are
    # zero, and so are the padded rows except a few past M that the epilogue
    # quantizes from all-zero accumulators.
    sf_rows, sf_cols = out_scale.shape
    out_sf = convert_swizzled_to_linear(
        out_scale.view(torch.uint8), sf_rows, sf_cols * block_size, block_size
    )
    assert not out_sf[:, n // block_size :].any()
    assert not (out_sf[m:] == 0xFF).any()

    assert (out == ref).float().mean() > 0.97
    # No further from the exact result than the separate kernels.
    exact = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale,
        b_scale,
        a_global_scale,
        b_global_scale,
        m,
        n,
        torch.float32,
        block_size,
        device,
    )
    exact = torch.nn.functional.gelu(exact + bias.float())

    def error(data: torch.Tensor, scale: torch.Tensor) -> float:
        value = dequantize_nvfp4_to_dtype(
            data, scale, y_global_scale, torch.float32, device, block_size
        )
        return ((value - exact).norm() / exact.norm()).item()

    assert error(out, out_scale) <= error(ref, ref_scale) * 1.02 + 1e-3
